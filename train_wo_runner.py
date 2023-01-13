from agents.ppo_agent import PPOAgent
from envs.unity_env import UnityEnv
from architectures.victim_arch import *
import numpy as np

device = 'cuda'

# RL arguments
model_name = "agent"
game_name = None
max_episode_timestep = 500

# Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
curriculum = {
    'current_step': 0,
    "thresholds": [10000, 10000, 10000, 10000, 10000, 10000],
    "parameters": {
        "goalRadius": [2, 3, 5, 7, 10, 18, 28],
        "enemyRadius": [10, 10, 10, 10, 10, 10, 10],
    }
}

# Total episode of training
total_episode = 1e10
# Units of training (episodes or timesteps)
frequency_mode = 'timesteps'
# Frequency of training (in episode)
frequency = 10
# Memory of the agent (in episode)
memory = 10
# Action type of the policy
action_type = "discrete"
# Ignore this
random_steps = None
# Action space
action_size = 9

# Create agent
# The policy embedding and the critic embedding for the PPO agent are defined in the architecture file
# You can change those architectures, the PPOAgent will manage the action layers and the value layers
agent = PPOAgent(state_dim=121, batch_fraction=0.5, policy_embedding=PolicyEmbedding,
                 critic_embedding=CriticEmbedding, action_type="discrete", action_size=action_size,
                 model_name="agent", p_lr=7e-5, v_batch_fraction=0.5, v_num_itr=2, memory=memory,
                 c2=0.001, discount=0.99, v_lr=7e-5, frequency_mode=frequency_mode, distribution='beta',
                 action_min_value=-1, action_max_value=1, p_num_itr=10, lmbda=1, action_masking=True)


env = UnityEnv(game_name, True, 0, action_size=action_size, max_episode_timesteps=max_episode_timestep)
episode_count = 0
episode_rewards = []
# Runner loop
while True:
    # Episode loop

    # Evaluation
    state = env.reset()
    episode_count += 1
    episode_reward = 0
    # print("Now we evaluate the agent...")
    for step in range(max_episode_timestep):
        state = torch.from_numpy(np.asarray([state])).to(device)
        action, logprob, probs, dist = agent(state)
        action = action.detach().cpu()
        state_n, reward, done = env.step(action)
        agent.add_to_buffer(state, state_n, action, reward, probs, done)
        episode_reward += reward
        if done:
            break
        state = state_n

    # print("Reward got in episode {}: {}".format(episode_count, episode_reward))
    episode_rewards.append(episode_reward)
    if episode_count % 100 == 0:
        print("Mean episode rewards of 100 episodes at {}: {}".format(episode_count, np.mean(episode_rewards)))
        episode_rewards = []

    if episode_count % 10 == 0:
        # print("Now we train the agent at episode {}...".format(episode_count))
        agent.train()
        agent.clear_buffer()


