import gym
import numpy as np

class GameEnv:
    def __init__(self, game_name):
        self.__env = gym.make(game_name)
    
    def get_init_params(self):
        return self.__env.observation_space.shape, self.__env.action_space.n

    def env_reset(self):
        return self.__env.reset()
    
    def env_step(self, action):
        return self.__env.step(action)
    
    def env_render(self):
        self.__env.render()
