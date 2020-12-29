from r_utils import GameEnv
from nn_models import Q_Model
import torch.optim as optim
from torch.nn import MSELoss
from torch import Tensor, distributions
import numpy as np
import random
import matplotlib.pyplot as plt

class GameEmulator:
    def __init__(self, epochs, default_reward, batch_size):
        self.epochs = epochs
        self.max_reward = default_reward
        self.overall_reward = []
        self.batch_size = batch_size
        self.game = GameEnv("CartPole-v0")
        self.states_dim, self.actions_dim = self.game.get_init_params()

        self.q_model = Q_Model(self.actions_dim, self.states_dim[0])
        self.q_optim = optim.Adam(params=self.q_model.parameters(), lr=0.001)
        self.loss_func = MSELoss()

        # self.best_model = None

        self.sample_accumulator = []

        self.eps = 1
    
    def clear_sample(self):
        self.sample_accumulator.clear()
    
    def get_Q_score(self, state):
        return self.q_model(Tensor([state])).detach().numpy()

    def train(self):
        if len(self.sample_accumulator) > self.batch_size:
            batch = np.array(random.sample(self.sample_accumulator, self.batch_size))
        else:
            return
        prev_quality = self.q_model(Tensor([batch[:, 1]])).detach().numpy()[0]
        new_quality = self.q_model(Tensor([batch[:, 0]])).detach().numpy()[0]
        states = []
        Qs = []
        for inx, batch_el in enumerate(batch):
            if not batch_el[4]:
                Q = batch_el[3] + np.max(new_quality[inx])
            else:
                Q = batch_el[3]
            recounted_Q = prev_quality[inx]
            recounted_Q[batch_el[2]] = Q
            states.append(batch_el[1])
            Qs.append(recounted_Q)

        self.q_model.train()
        target = Tensor([Qs])
        self.q_optim.zero_grad()
        out = self.q_model(Tensor([states]))
        loss = self.loss_func(out, target)
        loss.backward()
        self.q_optim.step()

    def set_best(self, last_step=-3, eps_step=0.8):
        if self.max_reward < np.mean(self.overall_reward[last_step:]):
            if self.eps * eps_step > 0.001:
                self.eps *= eps_step
                print(self.eps)
            self.max_reward = np.mean(self.overall_reward[last_step:])
            # self.best_model = self.q_model
    
    def train_model(self, final_reward=None):
        
        for i in range(self.epochs):
            print(f"----------- Start {i + 1} train epoch -----------")
            
            done = False
            epoch_reward = 0
            prev_state = self.game.env_reset()
            while not done:
                if random.random() > self.eps:
                    action = np.argmax(self.get_Q_score(prev_state))
                else:
                    action = np.random.randint(0, self.actions_dim)

                state, reward, done, _ = self.game.env_step(action)
                epoch_reward += reward
                self.sample_accumulator.append((state, prev_state, action, reward, done))
                prev_state = state

            self.train()

            self.overall_reward.append(epoch_reward)
            self.set_best()
            if final_reward and epoch_reward > final_reward:
                print(f"----------- Model achieve the maximum best result - {epoch_reward} -----------")
                return

            print(f"----------- At the end of {i + 1} epoch reward = {epoch_reward} -----------")

    def play(self):
        print(self.max_reward)
        self.q_model.eval()
        while True:
            state = self.game.env_reset()
            done = False
            while not done:
                out = self.q_model(Tensor(state))
                action = np.argmax(out.detach().numpy())
                state, _, done, _ = self.game.env_step(action)
                self.game.env_render()
    
    def plot(self):
        plt.plot([i for i,v in enumerate(self.overall_reward)], self.overall_reward)
        plt.xlabel("Number of epochs")
        plt.ylabel("Reward")
        plt.show()

if __name__ == "__main__":
    game = GameEmulator(epochs=4000, default_reward=0, batch_size=100)
    game.train_model(final_reward=195)
    game.plot()
    game.play()

