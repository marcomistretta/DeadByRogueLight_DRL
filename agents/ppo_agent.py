import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import reduce
from typing import Optional
import math
from utils import *
from torch.distributions import Categorical, Beta, Normal
import os

class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask.bool()
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype
            ).to(device)
            logits = torch.where(self.mask, logits, self.mask_value)
            for i, l in enumerate(logits[0]):
                print("{}: {}, m:{}".format(i, l, mask[0, i]))
            input('...')
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps = 1e-13

class Policy(nn.Module):
    def __init__(self, state_dim, embedding_arch, action_size=4, action_type='discrete', distribution_type='beta', max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(Policy, self).__init__()

        # Policy hyperparameters
        self.action_size = action_size
        self.state_dim = state_dim
        self.max_action_value = max_action_value
        self.min_action_value = min_action_value
        self.action_type = action_type
        self.distribution_type = distribution_type

        # Layers specification
        self.embedding_l = embedding_arch(state_dim)

        if self.action_type == 'discrete':
            self.action_l = nn.Linear(self.embedding_l.output_dim, self.action_size)
        elif self.action_type == 'continuous':
            if self.distribution_type == 'beta':
                self.alpha_l = nn.Linear(self.embedding_l.output_dim, self.action_size)
                self.beta_l = nn.Linear(self.embedding_l.output_dim, self.action_size)

    def forward(self, state):
        logits = self.embedding_l(state)
        if self.action_type == 'discrete':
            logits = self.action_l(logits)
            x = F.softmax(logits)
        elif self.action_type == 'continuous':
            alpha = F.softplus(self.alpha_l(logits)) + 1
            beta = F.softplus(self.beta_l(logits)) + 1
            x = torch.concat([alpha, beta], dim=1)
        return x, logits

class Critic(nn.Module):
    def __init__(self, state_dim, action_size, embedding_arch, action_masking=False, **kwargs):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_size = action_size
        # Layers specification
        self.embedding_l = embedding_arch(state_dim)
        self.q1_l = nn.Linear(self.embedding_l.output_dim, 1)
        self.action_masking = action_masking

    def forward(self, state):
        if self.action_masking:
            state = state[:, :-self.action_size]
        q1 = self.embedding_l(state)
        q1 = self.q1_l(q1)

        return q1

class PPOAgent(nn.Module):
    def __init__(self, state_dim, policy_embedding, critic_embedding, p_lr=7e-6, v_lr=7e-5, batch_fraction=0.5,
                 p_num_itr=20, v_num_itr=2, v_batch_fraction=0.5, previous_act=False,
                 # Actions
                 distribution='beta', action_type='continuous', action_size=2, action_min_value=-1,
                 action_max_value=1, frequency_mode='episodes',
                 epsilon=0.2, c1=0.5, c2=0.01, discount=0.99, lmbda=1.0, name='ppo', memory=10, norm_reward=False,
                 model_name='agent', action_masking=False,
                 **kwargs):
        super(PPOAgent, self).__init__()
        # Model parameters
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.batch_fraction = batch_fraction
        self.v_batch_fraction = v_batch_fraction
        self.p_num_itr = p_num_itr
        self.v_num_itr = v_num_itr
        self.name = name
        self.norm_reward = norm_reward
        self.model_name = model_name
        self.frequency_mode = frequency_mode
        # Functions that define input and network specifications
        # Whether to use the previous actions or not.
        # Typically this is done with LSTM
        self.previous_act = previous_act
        self.action_masking = action_masking
        self.state_dim = state_dim

        # PPO hyper-parameters
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.lmbda = lmbda
        # Action hyper-parameters
        # Types permitted: 'discrete' or 'continuous'. Default: 'discrete'
        self.action_type = action_type if action_type == 'continuous' or action_type == 'discrete' else 'discrete'
        self.action_size = action_size
        # min and max values for continuous actions
        self.action_min_value = action_min_value
        self.action_max_value = action_max_value
        # Distribution type for continuous actions
        self.distribution_type = distribution if distribution == 'gaussian' or distribution == 'beta' else 'gaussian'

        self.buffer = dict()
        self.clear_buffer()
        self.memory = memory

        self.policy = Policy(state_dim, policy_embedding, self.action_size, self.action_type, self.distribution_type,
                             self.action_max_value, self.action_min_value).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.p_lr)

        self.critic = Critic(state_dim, action_size, critic_embedding, action_masking=self.action_masking).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.v_lr)

    def forward(self, inputs, deterministic=False):
        if self.action_masking:
            # The mask is the first action_size part of the input
            mask = inputs[:, -self.action_size:]
            inputs = inputs[:, :-self.action_size]

        if self.action_type == 'discrete':
            if deterministic:
                # Get the most probable action
                probs, logits = self.policy(inputs)
                action = torch.argmax(probs)
            else:
                # Sample an action from probability distribution
                probs, logits = self.policy(inputs)
                if self.action_masking:
                    dist = CategoricalMasked(logits=logits, mask=mask)
                else:
                    dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
        elif self.action_type == 'continuous':
            if self.distribution_type == 'beta':
                probs, logits = self.policy(inputs)
                alpha = probs[:, :self.action_size]
                beta = probs[:, self.action_size:]

                dist = Beta(alpha, beta)
                # Sample an action from beta distribution
                action = dist.sample()
                logprob = dist.log_prob(action)
                # If there are more than 1 continuous actions, do the mean of log_probs
                if self.action_size > 1:
                    logprob = torch.sum(logprob, dim=1)
                # Standardize the action between min value and max value
                action = self.action_min_value + (
                        self.action_max_value - self.action_min_value) * action

        return action, logprob, probs, dist

    # Clear the memory buffer
    def clear_buffer(self):

        self.buffer['episode_lengths'] = []
        self.buffer['states'] = []
        self.buffer['actions'] = []
        self.buffer['old_probs'] = []
        self.buffer['states_n'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals):
        # If we store more than memory episodes, remove the last episode
        if self.frequency_mode == 'episodes':
            if len(self.buffer['episode_lengths']) + 1 >= self.memory + 1:
                idxs_to_remove = self.buffer['episode_lengths'][0]
                del self.buffer['states'][:idxs_to_remove]
                del self.buffer['actions'][:idxs_to_remove]
                del self.buffer['old_probs'][:idxs_to_remove]
                del self.buffer['states_n'][:idxs_to_remove]
                del self.buffer['rewards'][:idxs_to_remove]
                del self.buffer['terminals'][:idxs_to_remove]
                del self.buffer['episode_lengths'][0]

        # If we store more than memory timesteps, remove the last timestep
        elif self.frequency_mode == 'timesteps':
            if (len(self.buffer['states']) + 1 > self.memory):
                del self.buffer['states'][0]
                del self.buffer['actions'][0]
                del self.buffer['old_probs'][0]
                del self.buffer['states_n'][0]
                del self.buffer['rewards'][0]
                del self.buffer['terminals'][0]
                if self.recurrent:
                    del self.buffer['internal_states_c'][0]
                    del self.buffer['internal_states_h'][0]
                if self.recurrent_baseline:
                    del self.buffer['v_internal_states_c'][0]
                    del self.buffer['v_internal_states_h'][0]

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['old_probs'].append(old_prob)
        self.buffer['states_n'].append(state_n)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)

        # If its terminal, update the episode length count (all states - sum(previous episode lengths)
        if self.frequency_mode == 'episodes':
            if terminals == 1 or terminals == 2:
                self.buffer['episode_lengths'].append(
                    int(len(self.buffer['states']) - np.sum(self.buffer['episode_lengths'])))
        else:
            self.buffer['episode_lengths'] = []
            for i, t in enumerate(self.buffer['terminals']):
                if t == 1 or t == 2:
                    self.buffer['episode_lengths'].append(
                        int(i + 1 - np.sum(self.buffer['episode_lengths'])))

    # Change rewards in buffer to discounted rewards
    def compute_discounted_reward(self):

        discounted_rewards = []
        discounted_reward = 0
        # The discounted reward can be computed in reverse
        for (terminal, reward, i) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards']),
                                         reversed(range(len(self.buffer['rewards'])))):
            if terminal == 1:
                discounted_reward = 0
                # state = self.obs_to_state([self.buffer['states_n'][i]])
                # feed_dict = self.create_state_feed_dict(state)
                # discounted_reward = self.sess.run([self.value], feed_dict)[0]
            elif terminal == 2:
                state = self.buffer['states_n'][i]
                discounted_reward = self.critic(torch.from_numpy(state).to(device))

            discounted_reward = reward + (self.discount * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing reward
        if self.norm_reward:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                    np.std(discounted_rewards) + eps)

        return discounted_rewards

    # Change rewards in buffer to discounted rewards or GAE rewards (if lambda == 1, gae == discounted)
    def compute_gae(self, v_values):

        rewards = []
        gae = 0

        # The gae rewards can be computed in reverse
        for (terminal, reward, i) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards']),
                                         reversed(range(len(self.buffer['rewards'])))):
            m = 1
            if terminal == 1:
                m = 0
                gae = 0

            delta = reward + self.discount * v_values[i + 1] * m - v_values[i]
            gae = delta + self.discount * self.lmbda * m * gae
            discounted_reward = gae + v_values[i]

            rewards.insert(0, discounted_reward)

        # Normalizing
        if self.norm_reward:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)

        return rewards

    # Critic loss
    def critic_loss(self, q_values, rewards):
        return F.mse_loss(q_values, rewards)

    # Policy loss
    def policy_loss(self, rewards, actions, dist, baseline_values, oldprob):

        # Advantage (reward - baseline)
        advantages = (rewards - baseline_values).reshape(-1, 1)

        # L_clip loss
        if self.action_type == 'discrete':
            logprob_action = dist.log_prob(actions).reshape((-1, 1))
        else:
            # Inverse normalization actions between min_value and max_value
            # Beta Distribution
            if self.distribution_type == 'beta':
                actions = (actions - self.action_min_value) / (
                        self.action_max_value - self.action_min_value)
                actions = torch.clamp(actions, 0 + eps, 1 - eps)

                logprob_action = dist.log_prob(actions)
                if self.action_size > 1:
                    logprob_action = torch.sum(logprob_action, dim=1)
                logprob_action = logprob_action.reshape(-1, 1)

        ratio = torch.exp(logprob_action - oldprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        clip_loss = torch.minimum(surr1, surr2)

        # Entropy Bonus
        entr_loss = dist.entropy()
        # If there are more than 1 continuous actions, do the mean of entropies
        if self.action_size > 1 and self.action_type == 'continuous':
            entr_loss = torch.sum(entr_loss, dim=1)
        entr_loss = entr_loss.reshape(-1, 1)

        total_loss = - torch.mean(clip_loss + self.c2 * (entr_loss + eps))
        return total_loss

    # Update the model
    def update(self):
        self.train()
        losses = []
        v_losses = []

        # Get critic batch size based on v_batch_fraction
        batch_size = int(len(self.buffer['states']) * self.v_batch_fraction)

        # Before training, compute discounted reward
        discounted_rewards = self.compute_discounted_reward()

        # Train the value function
        for it in range(self.v_num_itr):
            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)
            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            states_mini_batch = torch.from_numpy(np.asarray(states_mini_batch)).to(device).float()
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
            rewards_mini_batch = torch.from_numpy(np.asarray(rewards_mini_batch)).to(device).float()
            rewards_mini_batch = rewards_mini_batch.reshape(-1, 1)

            v_values = self.critic(states_mini_batch)

            critic_loss = self.critic_loss(v_values, rewards_mini_batch)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            v_losses.append(critic_loss.detach().cpu())

        with torch.no_grad():
            # Compute GAE for rewards. If lambda == 1, they are discounted rewards
            # Compute values for each state
            num_batches = 1
            batch_size = int(np.ceil(len(self.buffer['states']) / num_batches))
            v_values = []
            for i in range(num_batches):
                states = self.buffer['states'][batch_size*i:batch_size*i + batch_size]
                states = torch.from_numpy(np.asarray(states)).to(device)

                v = self.critic(states).detach().cpu().numpy()
                v_values.extend(v)
        v_values = np.append(v_values, 0)
        discounted_rewards = self.compute_gae(v_values)

        # Get policy batch size based on batch_fraction
        batch_size = int(len(self.buffer['states']) * self.batch_fraction)
        # Train the policy
        for it in range(self.p_num_itr):
            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)

            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            states_mini_batch = torch.from_numpy(np.asarray(states_mini_batch)).to(device)
            actions_mini_batch = [self.buffer['actions'][id] for id in mini_batch_idxs]
            actions_mini_batch = torch.from_numpy(np.asarray(actions_mini_batch)).to(device)
            old_probs_mini_batch = [self.buffer['old_probs'][id] for id in mini_batch_idxs]
            old_probs_mini_batch = torch.from_numpy(np.asarray(old_probs_mini_batch)).to(device)
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
            rewards_mini_batch = torch.from_numpy(np.asarray(rewards_mini_batch)).to(device)
            # Get the baseline values
            v_values_mini_batch = [v_values[id] for id in mini_batch_idxs]
            v_values_mini_batch = torch.from_numpy(np.asarray(v_values_mini_batch)).to(device)

            pi_actions, logprob, probs, dist = self.forward(states_mini_batch)

            p_loss = self.policy_loss(rewards_mini_batch, actions_mini_batch, dist, v_values_mini_batch, old_probs_mini_batch)
            # Optimize the actor
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            self.policy_optimizer.step()
            losses.append(p_loss.detach().cpu())
        return np.mean(losses)

    def update_weights_op(self, weights):
        # Assign to model_a the weights of model_b. Use it for update the target networks weights.
        # For the policy
        # for a, b in zip(self.policy.parameters(), weights['policy'].parameters()):
        #     a.data.copy_(b.data)
        #
        # # For the critic
        # for a, b in zip(self.critic.parameters(), weights['critic'].parameters()):
        #     a.data.copy_(b.data)
        self.critic.load_state_dict(weights['critic'])
        self.policy.load_state_dict(weights['policy'])

    def get_weights(self):
        weights = dict()
        weights['policy'] = self.policy.state_dict()
        weights['critic'] = self.critic.state_dict()
        return weights

    def save_model(self, name=None, folder='saved', with_barracuda=True,
                   barracuda_folder="../AIGameJam/Assets/Scripts/ML/Models/"):

        torch.save(self.critic.state_dict(), '{}/{}_critic'.format(folder, name))
        torch.save(self.critic_optimizer.state_dict(), '{}/{}_critic_optimizer'.format(folder, name))

        torch.save(self.policy.state_dict(), '{}/{}_actor'.format(folder, name))
        torch.save(self.policy_optimizer.state_dict(), '{}/{}_actor_optimizer'.format(folder, name))
        if with_barracuda:
            # Input to the model
            x = torch.zeros(1, self.state_dim).to(device)

            # Export the model
            torch.onnx.export(self.policy,  # model being run
                              x,  # model input (or a tuple for multiple inputs)
                              "{}/{}.onnx".format(barracuda_folder, self.model_name),
                              # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=9,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['X'],  # the model's input names
                              output_names=['Y']  # the model's output names
                              )

    def load_model(self, name=None, folder='saved'):
        self.critic.load_state_dict(torch.load('{}/{}_critic'.format(folder, self.model_name)))
        self.critic_optimizer.load_state_dict(torch.load('{}/{}_critic_optimizer'.format(folder, self.model_name)))

        self.policy.load_state_dict(torch.load('{}/{}_actor'.format(folder, name)))
        self.policy_optimizer.load_state_dict(torch.load('{}/{}_actor_optimizer'.format(folder, name)))

