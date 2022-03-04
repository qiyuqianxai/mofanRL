import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, RMSprop, AdamW
import random
import matplotlib.pyplot as plt
import gym
import torch.nn.functional as F
import time

ENV_NAME = 'Pendulum-v1'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

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
    n_action = env.action_space.shape[0]
    n_observation = env.observation_space.shape[0]
    action_bound = env.action_space.high
    lr = 0.01
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    reward_decay = 0.95
    GAMMA = 0.9
    rep_iter_a = 600
    rep_iter_c = 500
    tau = 0.01

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0,0.3)
        nn.init.constant_(m.bias,0.1)

class actor_net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(actor_net, self).__init__()
        self.layer1 = nn.Linear(in_dim,30)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(30, out_dim)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.tanh(x)
        return x

class Actor(object):
    def __init__(self):
        args = get_config()
        self.a_dim = args.n_action
        self.s_dim = args.n_observation
        self.action_bound = args.action_bound
        self.lr = args.lr
        self.t_replace_counter = 0
        self.tau = args.tau
        self.rep_iter_a = args.rep_iter_a
        self.device = args.device
        self.eval_net = actor_net(self.s_dim,self.a_dim).to(self.device)
        self.target_net = actor_net(self.s_dim,self.a_dim).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.optimizer = Adam(self.eval_net.parameters(),lr=self.lr)
        self.t_params = self.target_net.parameters()

    def learn(self, s):   # batch update
        self.eval_net.train()
        s = torch.tensor(s,dtype=torch.float)
        logits = self.eval_net(s)
        action_bound = torch.tensor(self.action_bound,dtype=torch.float)
        scaled_a = torch.multiply(logits, action_bound)
        v = critic.target_net(s,scaled_a)
        loss = -torch.mean(v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update target_net
        for e_param, t_param in zip(self.eval_net.parameters(),self.target_net.parameters()):
            t_param.data = e_param.data*(1-self.tau) + t_param.data*self.tau

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        s = torch.tensor(s,dtype=torch.float).to(self.device)
        action_bound = torch.tensor(self.action_bound, dtype=torch.float)
        self.eval_net.eval()
        with torch.no_grad():
            logits = self.eval_net(s)
            scaled_a = torch.multiply(logits, action_bound)

        return scaled_a[0]  # single action

class critic_net(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(critic_net, self).__init__()
        self.s_layer = nn.Linear(s_dim,30)
        self.a_layer = nn.Linear(a_dim,30)
        self.v_layer = nn.Linear(30,1)
        self.relu = nn.ReLU()

    def forward(self,s, a):
        s = self.s_layer(s)
        a = self.a_layer(a)
        v = self.relu(s+a)
        v = self.v_layer(v)
        return v

class Critic(object):
    def __init__(self):
        args = get_config()
        self.a_dim = args.n_action
        self.s_dim = args.n_observation
        self.action_bound = args.action_bound
        self.lr = args.lr
        self.t_replace_counter = 0
        self.tau = args.tau
        self.gamma = args.GAMMA
        self.rep_iter_a = args.rep_iter_a
        self.device = args.device
        self.eval_net = critic_net(self.s_dim,self.a_dim).to(self.device)
        self.target_net = critic_net(self.s_dim,self.a_dim).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.optimizer = Adam(self.eval_net.parameters(),lr=self.lr)
        self.critisen = nn.MSELoss()

    def learn(self, b_s, b_a, b_r, b_s_):
        # train eval_net
        self.eval_net.train()
        b_s = torch.tensor(b_s,dtype=torch.float).to(self.device)
        b_a = torch.tensor(b_a,dtype=torch.float).to(self.device)
        b_r = torch.tensor(b_r,dtype=torch.float).to(self.device)
        b_s_ = torch.tensor(b_s_,dtype=torch.float).to(self.device)
        action_bound = torch.tensor(self.action_bound, dtype=torch.float)

        logits = actor.target_net(b_s_)
        scaled_a_ = torch.multiply(logits, action_bound)
        eval_v = self.eval_net(b_s, b_a)
        v_s_ = self.target_net(b_s_, scaled_a_)
        target_v = b_r + self.gamma*v_s_
        loss = self.critisen(eval_v, target_v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target_net
        for e_param,t_param in zip(self.eval_net.parameters(),self.target_net.parameters()):
            t_param.data = e_param.data*(1-self.tau) + t_param.data*self.tau

def train():
    MAX_EPISODES = 200
    MAX_EP_STEPS = 200
    MEMORY_CAPACITY = 10000
    BATCH_SIZE = 32
    RENDER = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    M = Memory(MEMORY_CAPACITY, dims=2 * env.observation_space.shape[0] + env.action_space.shape[0] + 1)
    var = 3  # control exploration

    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            M.store_transition(s, a, r / 10, s_)

            if M.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300:
                    RENDER = True
                break

    print('Running time: ', time.time() - t1)

if __name__ == '__main__':
    actor = Actor()
    critic = Critic()
    train()
