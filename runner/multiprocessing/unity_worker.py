from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
import torch
from envs.unity_env import UnityEnv as Env
import numpy as np
import ray
from architectures.victim_arch import *

ray.init()
@ray.remote
class RemoteWorker:
    # TODO: imrpove argument
    def __init__(self, init_queue, i, device, **kwargs):

        # Create environment in the memory of the remote worker
        # Get env arguments
        self.game_name = kwargs.pop('game_name', None)
        self.no_graphics = kwargs.pop('no_graphics', True)
        self.worker_id = kwargs.pop('worker_id', 10)
        self.max_episode_timesteps = kwargs.pop("max_episode_timesteps", 400)

        self.device = device
        # Create network in the memory of the remote worker
        # This net will be updated with the weights of the centralized net in the server
        # It has the same structure of the centralized net

        state_dim = kwargs.pop('state_dim', 0)
        action_type = kwargs.pop('action_type', 'discrete')
        action_size = kwargs.pop('action_size', 3)
        recurrent = kwargs.pop('recurrent', False)
        distribution = kwargs.pop('distribution', 'beta')
        action_max_value = kwargs.pop('action_max_value', 1)
        action_min_value = kwargs.pop('action_min_value', -1)

        self.env = Env(self.game_name, True, i, action_size=action_size, max_episode_timesteps=self.max_episode_timesteps)

        self.agent = PPOAgent(state_dim=state_dim,
                              policy_embedding=PolicyEmbedding,
                              critic_embedding=CriticEmbedding,
                              action_type=action_type,
                              action_size=action_size,
                              recurrent=recurrent,
                              distribution=distribution,
                              action_min_value=action_min_value,
                              action_max_value=action_max_value,
                              )
        init_queue.put('Created')

    def update_network_weight(self, weight):
        self.agent.update_weights_op(weight)

    def mp_run(self, queue, num_episode=1):
        env = self.env
        # This parallel buffer is local for each worker. Each
        # execution will put the copy of the buffer into the rollout queue.
        parallel_buffer = {
            'states': [],
            'states_n': [],
            'done': [],
            'reward': [],
            'action': [],
            'logprob': [],
            'internal': [],
            'v_internal': [],
            # Motivation
            'motivation': [],
            # Reward model
            'reward_model': [],
            # History
            'episode_rewards': [],
            'episode_timesteps': [],
            'mean_entropies': [],
            'std_entropies': [],
        }
        self.agent.eval()
        # Run each thread for num_episode episodes
        for i in range(num_episode):
            done = False
            step = 0
            # Reset the environment
            state = env.reset()

            # Total episode reward
            episode_reward = 0

            # Local entropies of the episode
            local_entropies = []

            while not done:
                # Evaluation - Execute step
                actions, logprobs, probs, dist = self.agent(torch.from_numpy(np.asarray([state])).to(self.device))
                actions = actions.detach().cpu().numpy()
                logprobs = logprobs.detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()
                actions = actions[0]
                state_n, reward, done = self.env.step(actions)

                episode_reward += reward
                local_entropies.append(0)
                if step >= self.max_episode_timesteps - 1:
                    done = True
                parallel_buffer['states'].append(state)
                parallel_buffer['states_n'].append(state_n)
                parallel_buffer['done'].append(done)
                parallel_buffer['reward'].append(reward)
                parallel_buffer['action'].append(actions)
                parallel_buffer['logprob'].append(logprobs)

                state = state_n
                step += 1

            # History statistics
            parallel_buffer['episode_rewards'].append(episode_reward)
            parallel_buffer['episode_timesteps'].append(step)
            parallel_buffer['mean_entropies'].append(np.mean(local_entropies))
            parallel_buffer['std_entropies'].append(np.std(local_entropies))
            queue.put(parallel_buffer)
