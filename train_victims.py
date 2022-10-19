from agents.ppo_agent import PPOAgent
from runner.parallel_runner import Runner
from runner.parallel_mp_runner import Runner as MPRunner
import os
import argparse
import torch
from envs.unity_env import UnityEnv
import os
from architectures.victim_arch import *
from runner.multiprocessing.unity_worker import RemoteWorker
from ray.util.queue import Queue

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the policy", default='agent')
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=1000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=100)
parser.add_argument('-mp', '--multiprocessing', default=False)
parser.add_argument('-pl', '--parallel', help="How many environments to simulate in parallel. Default is 1", type=int, default=1)

args = parser.parse_args()

eps = 1e-12

epis = 0

def callback(agent, env, runner):
    return

if __name__ == "__main__":

    # RL arguments
    model_name = args.model_name
    game_name = args.game_name
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = None

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'episodes'
    # Frequency of training (in episode)
    frequency = 10
    # Memory of the agent (in episode)
    memory = 10
    # Action type of the policy
    action_type = "discrete"
    # Ignore this
    random_steps = None
    # Action space
    action_size = 5

    # Create agent
    # The policy embedding and the critic embedding for the PPO agent are defined in the architecture file
    # You can change those architectures, the PPOAgent will manage the action layers and the value layers
    agent = PPOAgent(state_dim=100, batch_fraction=0.5, policy_embedding=PolicyEmbedding,
                     critic_embedding=CriticEmbedding, action_type=action_type, action_size=action_size,
                     model_name=model_name, p_lr=7e-5, v_batch_fraction=0.5, v_num_itr=2, memory=memory,
                     c2=0.001, discount=0.99, v_lr=7e-5, frequency_mode=frequency_mode, distribution='beta',
                     action_min_value=-1, action_max_value=1, p_num_itr=10, lmbda=0.99, action_masking=False)

    envs = []
    if args.multiprocessing:
        how_many_client = 5
        init_queue = Queue(how_many_client)
        envs = [RemoteWorker.remote(init_queue, ti, device=device,
                                    # Env argument
                                    game_name=game_name,
                                    state_dim=486,
                                    max_episode_timesteps=max_episode_timestep,
                                    action_type=agent.action_type,
                                    action_size=agent.action_size,
                                    ) for ti in range(how_many_client)]

        while not init_queue.full():
            continue
    else:
        for i in range(args.parallel):
            envs.append(UnityEnv(game_name, True, i, action_size=action_size, max_episode_timesteps=max_episode_timestep))

    # Create runner
    # This class manages the evaluation of the policy and the collection of experience in a parallel setting
    # (not vectorized)
    if args.multiprocessing:
        print("Start training with multiprocessing...")
        runner = MPRunner(agent=agent,
                        frequency=frequency, envs=envs, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback)
    else:
        runner = Runner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                            logging=logging, total_episode=total_episode, curriculum=curriculum,
                            frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback)

    try:
        runner.run()
    finally:
        for env in envs:
            env.close()
