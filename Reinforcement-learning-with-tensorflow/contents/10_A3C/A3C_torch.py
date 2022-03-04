import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, RMSprop, AdamW
import random
import matplotlib.pyplot as plt
import gym
import torch.nn.functional as F
import time
import threading
import multiprocessing

GAME = 'CartPole-v0'
N_WORKERS = multiprocessing.cpu_count()
GAMMA = 0.9
ENTROPY_BETA = 0.001
MAX_GLOBAL_EP = 5000
GLOBAL_RUNNING_R = []
UPDATE_GLOBAL_ITER = 10
GLOBAL_EP = 0
env = gym.make(GAME)

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
    lr_a = 1e-4
    lr_c = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    GAMMA = 0.9

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0.,.1)
        # nn.init.constant_(m.bias,0.1)

class AC_Net(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(AC_Net, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(s_dim,200),
            nn.ReLU(),
            nn.Linear(200,a_dim),
        )
        self.softmax = nn.Softmax()
        self.critic = nn.Sequential(
            nn.Linear(s_dim,100),
            nn.ReLU(),
            nn.Linear(100,1),
        )
        self.apply(weights_init)

    def forward(self,s):
        a = self.actor(s)
        a_prob = self.softmax(a)
        v = self.critic(s)
        return a_prob, a, v

class A3C(object):
    def __init__(self,globalAC_Net):
        args = get_config()
        self.device = args.device
        self.globalAC = globalAC_Net.to(self.device)
        self.s_dim = args.n_observation
        self.a_dim = args.n_action
        self.ac_net = AC_Net(self.s_dim,self.a_dim).to(self.device)
        self.optimizer_a = RMSprop(self.globalAC.actor.parameters(),lr=args.lr_a,momentum=0.9)
        self.optimizer_c = RMSprop(self.globalAC.critic.parameters(),lr=args.lr_c,momentum=0.9)
        self.critisen_c = nn.MSELoss()
        self.critisen_a = nn.CrossEntropyLoss()

    def update_global(self,buffer_s, buffer_a, buffer_v_target):
        buffer_s = torch.tensor(buffer_s,dtype=torch.float).to(self.device)
        buffer_a = torch.tensor(buffer_a,dtype=torch.long).to(self.device)
        buffer_v_target = torch.tensor(buffer_v_target,dtype=torch.float).to(self.device)
        self.ac_net.train()

        a_prob, a, eval_v = self.ac_net(buffer_s)
        c_loss = self.critisen_c(eval_v,buffer_v_target)
        self.optimizer_c.zero_grad()
        c_loss.backward()
        for c_param,g_c_param in zip(self.ac_net.critic.parameters(),self.globalAC.critic.parameters()):
            if g_c_param.grad is not None:
                break
            g_c_param._grad = c_param.grad
        self.optimizer_c.step()

        a_loss = self.critisen_a(a, buffer_a)
        exp_v = a_loss * torch.subtract(eval_v.detach(), buffer_v_target.detach())
        entropy = -torch.sum(a_prob * torch.log(a_prob + 1e-5), dim=1)
        exp_v = ENTROPY_BETA * entropy + exp_v
        a_loss = torch.mean(-exp_v)
        self.optimizer_a.zero_grad()
        a_loss.backward()
        for a_param, g_a_param in zip(self.ac_net.actor.parameters(),self.globalAC.actor.parameters()):
            if g_a_param.grad is not None:
                break
            g_a_param._grad = a_param.grad
        self.optimizer_a.step()

    def pull_global(self):
        self.ac_net.load_state_dict(self.globalAC.state_dict())

    def chose_action(self,s):
        self.ac_net.eval()
        with torch.no_grad():
            s = s[np.newaxis, :]
            s = torch.tensor(s,dtype=torch.float).to(self.device)
            a_prob, _, _ = self.ac_net(s)
            a_prob = a_prob.cpu().numpy()
            action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())  # select action w.r.t the actions prob
            return action

    def get_s_v(self,s):
        self.ac_net.eval()
        s = s[np.newaxis, :]
        s = torch.tensor(s,dtype=torch.float).to(self.device)
        with torch.no_grad():
            v = self.ac_net.critic(s)
        return v[0]

class Worker(object):
    def __init__(self, name, globalAC_Net):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = A3C(globalAC_Net)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        buffer_s, buffer_a, buffer_r = [], [], []
        while GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            total_step = 1
            while True:
                if self.name == 'W_1':
                    self.env.render()
                a = self.AC.chose_action(s)
                s_, r, done, info = self.env.step(a)
                if done:
                    r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.AC.get_s_v(s_)
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_target = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_target)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)

                    global_lock.acquire(True)
                    self.AC.pull_global()
                    self.AC.update_global(buffer_s,buffer_a,buffer_v_target)
                    self.AC.pull_global()
                    global_lock.release()

                    buffer_s, buffer_a, buffer_r = [], [], []

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        "| step:", total_step
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    GLOBAL_AC_Net = AC_Net(get_config().n_observation,get_config().n_action)  # we only need its params
    global_lock = threading.Lock()
    workers = []
    # Create worker
    print(N_WORKERS)
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i   # worker name
        workers.append(Worker(i_name, GLOBAL_AC_Net))
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    [t.join() for t in worker_threads]

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()