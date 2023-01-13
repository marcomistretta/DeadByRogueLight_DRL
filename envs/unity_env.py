from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import logging as logs
import time
import matplotlib.pyplot as plt
from PIL import Image
import pickle

class UnityEnv:
    def __init__(self, game_name, no_graphics, worker_id, action_size, max_episode_timesteps, action_type='discrete'):

        self.no_graphics = no_graphics
        # Channel for passing the parameters
        self.curriculum_channel = EnvironmentParametersChannel()
        self.configuration_channel = EngineConfigurationChannel()
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=int(time.time()), worker_id=worker_id,
                                          side_channels=[self.curriculum_channel, self.configuration_channel])
        self.behavior_name = 'Victim?team=0'
        self.unity_env.reset()
        self.configuration_channel.set_configuration_parameters(time_scale=100, quality_level=0)
        self._max_episode_timesteps = max_episode_timesteps
        self.config = None
        self.current_timestep = 0
        self.discrete = action_type == 'discrete'
        self.action_size = action_size
        self.ep = 0

    def reset(self):

        # Change config to be fed to Unity (no list)
        unity_config = dict()
        if self.config is not None:
            for key in self.config.keys():
                unity_config[key] = self.config[key]

        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        self.current_timestep = 0

        # Set curriculum values
        for par, val in unity_config.items():
            self.curriculum_channel.set_float_parameter(par, val)

        decision_steps = None
        self.unity_env.reset()

        while decision_steps is None or len(decision_steps.obs[0]) <= 0:
            self.unity_env.step()
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        # state = decision_steps.obs[0][0, :]
        state = np.concatenate(decision_steps.obs, axis=-1)[0]
        self.ep += 1
        return state

    def step(self, actions, visualize=False):
        # actions = input("actions: ")
        if self.discrete:
            actions = np.asarray(actions)
            actions = np.reshape(actions, [1, 1])

            actions = actions
            actionsAT = ActionTuple()
            actionsAT.add_discrete(actions)
        else:
            actions = np.asarray(actions)
            actions = np.reshape(actions, [1, self.action_size])

            actionsAT = ActionTuple()
            actionsAT.add_continuous(actions)

        self.unity_env.set_actions(self.behavior_name, actionsAT)
        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        reward = None

        if(len(terminal_steps.interrupted) > 0):
            # state = terminal_steps.obs[1][0, :]
            state = np.concatenate(terminal_steps.obs, axis=-1)[0]

            done = True
            reward = terminal_steps.reward[0]
        else:
            # state = decision_steps.obs[1][0, :]
            state = np.concatenate(decision_steps.obs, axis=-1)[0]

            reward = decision_steps.reward[0]
            done = False

        if self.current_timestep > self._max_episode_timesteps:
            done = True

        self.current_timestep += 1
        # print(reward)
        return state, reward, done

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.unity_env.close()
