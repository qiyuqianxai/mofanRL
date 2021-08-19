import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, RMSprop, AdamW
import random
import matplotlib.pyplot as plt
import gym
import torch.nn.functional as F

env = gym.make('MountainCar-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class get_config():
    n_action = env.action_space.n
    n_observation = env.observation_space.shape[0]
    lr = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_decay = 0.95

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0,0.3)
        nn.init.constant_(m.bias,0.1)

class net(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(net, self).__init__()
        self.layer1 = nn.Linear(s_dim,10)
        self.ativate= nn.ReLU()
        self.layer2 = nn.Linear(10, a_dim)
        self.softmax = nn.Softmax()
        self.apply(weights_init)

    def forward(self,x):
        x = self.layer1(x)
        x = self.ativate(x)
        x = self.layer2(x)
        prob = self.softmax(x)
        return x, prob

class PolicyGradient():
    def __init__(self):
        self.args = get_config()
        self.device = self.args.device
        self.net = net(self.args.n_observation,self.args.n_action).to(self.device)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.optimizer = Adam(self.net.parameters(), lr=self.args.lr)
        self.critisen = nn.CrossEntropyLoss()


    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        self.net.train()
        ep_obs = np.vstack(self.ep_obs)
        ep_obs = torch.tensor(ep_obs,dtype=torch.float).to(self.device)
        ep_as = torch.tensor(self.ep_as,dtype=torch.long).to(self.device)
        discounted_ep_rs_norm = torch.tensor(discounted_ep_rs_norm,dtype=torch.float).to(self.device)

        logits, _ = self.net(ep_obs)
        loss = self.critisen(logits, ep_as)
        loss = torch.mean(loss*discounted_ep_rs_norm)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm.cpu().numpy(), loss.item()

    def choose_action(self, observation):
        self.net.eval()
        with torch.no_grad():
            observation = observation[np.newaxis, :]
            observation = torch.tensor(observation,dtype=torch.float).to(self.device)
            _, prob_weights = self.net(observation)
        prob_weights = prob_weights.cpu().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.args.reward_decay + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

def train():
    DISPLAY_REWARD_THRESHOLD = 100 # renders environment if total episode reward is greater then this threshold
    render = False  # rendering wastes time
    max_episode = 3000
    RL = PolicyGradient()
    for i_episode in range(max_episode):
        observation = env.reset()
        while True:
            if render:
                env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    render = True  # rendering

                vt, loss = RL.learn()
                print("episode:", i_episode, "  reward:", int(running_reward)," loss:",loss," step:",vt.__len__())
                if i_episode == 0:
                    plt.plot(vt)  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            observation = observation_

if __name__ == '__main__':
    train()



