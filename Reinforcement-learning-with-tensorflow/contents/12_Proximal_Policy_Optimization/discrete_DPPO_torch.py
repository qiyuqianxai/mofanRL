import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from torch.optim import Adam, RMSprop, AdamW
import itertools

GAME = 'CartPole-v0'
env = gym.make(GAME)

def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()

class config():
    EP_MAX = 1000
    EP_LEN = 500
    N_WORKER = 4  # parallel workers
    GAMMA = 0.9  # reward discount factor
    A_LR = 0.0001  # learning rate for actor
    C_LR = 0.0001  # learning rate for critic
    MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
    UPDATE_STEP = 15  # loop update operation n-steps
    EPSILON = 0.2  # for clipping surrogate objective
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0,0.1)
        nn.init.constant_(m.bias,0.1)

class a_net(nn.Module):
    def __init__(self,indim, outdim):
        super(a_net, self).__init__()
        self.layer1 = nn.Linear(indim,200)
        self.layer2 = nn.Linear(200,outdim)
        self.apply(weights_init)

    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        a_prob = F.softmax(x)
        return a_prob

class c_net(nn.Module):
    def __init__(self,indim):
        super(c_net, self).__init__()
        self.layer1 = nn.Linear(indim,200)
        self.layer2 = nn.Linear(200,1)
        self.apply(weights_init)

    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        v = self.layer2(x)
        return v

class PPO(object):
    def __init__(self,args):
        self.args = args
        self.indim = args.S_DIM
        self.outdim = args.A_DIM

        self.anet_c = a_net(self.indim, self.outdim).to(self.args.device)
        self.anet_t = a_net(self.indim, self.outdim).to(self.args.device)
        self.a_lr = args.A_LR
        self.a_opt = Adam(self.anet_c.parameters(),lr=self.a_lr)

        self.cnet = c_net(self.indim).to(self.args.device)
        self.c_lr = args.C_LR
        self.c_critisen = nn.MSELoss()
        self.c_opt = Adam(self.cnet.parameters(),lr=self.c_lr)

    def learn(self):
        global GLOBAL_UPDATE_COUNTER,Request_Stop,GLOBAL_EP
        while not Request_Stop:
            if GLOBAL_EP < self.args.EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.anet_t.load_state_dict(self.anet_c.state_dict())
                for a_t_params in self.anet_t.parameters():  # copy pi to old pi
                    a_t_params.requires_grad = False
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :self.indim], data[:, self.indim: self.indim + 1].ravel(), data[:, -1:]
                b_s = torch.tensor(s, dtype=torch.float).to(self.args.device)
                b_a = torch.tensor(a, dtype=torch.float).to(self.args.device)
                a_indices = torch.stack([torch.arange(b_a.shape[0], dtype=torch.int32).to(self.args.device), b_a], dim=1)
                b_r = torch.tensor(r, dtype=torch.float).to(self.args.device)
                self.cnet.eval()
                with torch.no_grad():
                    adv = b_r - self.cnet(b_s)
                # trian a_net
                self.anet_c.train()
                a_loss = []
                for i in range(self.args.UPDATE_STEP):
                    self.a_opt.zero_grad()
                    a_c_prob = self.anet_c(b_s)
                    a_c_prob = gather_nd(a_c_prob,a_indices)  # will do the trick
                    a_t_prob = self.anet_t(b_s)
                    a_t_prob = gather_nd(a_t_prob,a_indices)
                    ratio = a_c_prob / (a_t_prob + 1e-5)
                    surr = ratio*adv
                    loss = -torch.mean(torch.min(surr,torch.clamp(ratio, 1. - self.args.EPSILON, 1. + self.args.EPSILON)*adv))
                    a_loss.append(loss.item())
                    loss.backward()
                    self.a_opt.step()
                print(f"a loss:{sum(a_loss)/len(a_loss)}")

                # train c_net
                self.cnet.train()
                c_loss = []
                for i in range(self.args.UPDATE_STEP):
                    self.c_opt.zero_grad()
                    v = self.cnet(b_s)
                    loss = self.c_critisen(v,b_r)
                    c_loss.append(loss.item())
                    loss.backward()
                    self.c_opt.step()
                print(f"c loss:{sum(c_loss)/len(c_loss)}")

                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        s = torch.tensor(s, dtype=torch.float).to(self.args.device)
        self.anet_c.eval()
        with torch.no_grad():
            a_prob = self.anet_c(s)
            a_prob = a_prob.cpu().numpy()
            action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())  # select action w.r.t the actions prob
            return action

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.tensor(s, dtype=torch.float).to(self.args.device)
        self.cnet.eval()
        with torch.no_grad():
            v = self.cnet(s)
            v = v.item()
            return v


class Worker(object):
    def __init__(self, wid,args):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO
        self.args = args

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, Request_Stop
        while not Request_Stop:
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(self.args.EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if done: r = -10
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r - 1)  # 0 for not down, -11 for down. Reward engineering
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == self.args.EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= self.args.MIN_BATCH_SIZE or done:
                    if done:
                        v_s_ = 0  # end of episode
                    else:
                        v_s_ = self.ppo.get_v(s_)

                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.args.GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= self.args.MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= self.args.EP_MAX:  # stop training
                        Request_Stop = True
                        break

                    if done:
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print(GLOBAL_EP,'{0:.1f}%'.format(GLOBAL_EP / self.args.EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )

if __name__ == '__main__':
    args = config()
    GLOBAL_PPO = PPO(args)
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i,args=args) for i in range(args.N_WORKER)]
    Request_Stop = False
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []

    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.learn, ))
    threads[-1].start()
    # plot reward change and test
    env = gym.make('CartPole-v0')
    test_rs = []
    while not Request_Stop:
        s = env.reset()
        test_r = 0
        for t in range(args.EP_LEN):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            test_r+=r
            if done:
                break
        test_rs.append(test_r)
    print("end")
    [t.join() for t in threads]
    plt.figure()
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(test_rs)), test_rs,c="r")
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.show()
