import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, RMSprop
import random
import matplotlib.pyplot as plt
import gym

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

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0,0.3)
        nn.init.constant_(m.bias,0.1)

class DQN_Net(nn.Module):
    def __init__(self,s_dim, a_dim):
        super(DQN_Net, self).__init__()
        self.layer1 = nn.Linear(s_dim,20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20,a_dim)
        self.apply(weights_init)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # 返回的x是a_dim维，每个val代表采用该action得到的val
        return x

class config():
    lr = 0.005
    reward_decay = 0.9
    e_greedy = 0.9
    replace_target_iter = 200
    memory_size = 3000
    batch_size = 32
    e_greedy_increment = 0.001
    n_action = 11
    n_observation = 3

class Double_DQN():
    def __init__(self, args):
        self.s_dim = args.n_observation
        self.a_dim = args.n_action
        self.eval_net = DQN_Net(self.s_dim,self.a_dim)
        self.target_net = DQN_Net(self.s_dim,self.a_dim)
        self.args = args
        self.epsilon_max = self.args.e_greedy
        self.epsilon = 0 if self.args.e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.args.memory_size, self.s_dim * 2 + 2))
        self.critisen = nn.MSELoss()
        self.optimizer = RMSprop(self.eval_net.parameters(), lr=self.args.lr, momentum=0.9)

    def learn(self):
        # 更新target模型参数
        if self.learn_step_counter+1 % self.args.replace_target_iter == 0:
            self.target_net.load_state_dict(torch.load(self.eval_net.state_dict()))
            # 增加epsilon
            self.epsilon = self.epsilon + self.args.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # 取一个batch的数据对eval_net进行训练
        if self.memory_counter > self.args.memory_size:
            sample_index = np.random.choice(self.args.memory_size, size=self.args.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.args.batch_size)
        ############# train ###########################
        batch_memory = self.memory[sample_index,:]
        batch_memory = torch.tensor(batch_memory,dtype=torch.float)
        self.eval_net.eval()
        self.target_net.eval()

        # 输入s_到target预测下一s的val
        q_next = self.target_net(batch_memory[:, -self.s_dim:])
        # doubel dqn的改进：同时把s_输入到eval中
        q_next_eval = self.eval_net(batch_memory[:, -self.s_dim:])

        # 输入当前s到eval预测当前各个action的val
        self.eval_net.train()
        q_eval = self.eval_net(batch_memory[:,:self.s_dim])

        # 从memory中获取该s下实际采取的action以及其对应的r
        eval_act_index = batch_memory[:, self.s_dim].long()
        rewards = batch_memory[:, self.s_dim+1]
        q_target = q_eval.clone()

        # 从eval中找出在s_下得分最高的action
        max_act4next = torch.argmax(q_next_eval, dim=1)
        selected_q_next = q_next[:, max_act4next]
        # 将target中选中的最高分的action替换为实际score从而与预测的结果进行对比
        q_target[:,eval_act_index] = rewards + self.args.reward_decay*selected_q_next

        loss = self.critisen(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        return loss.item()

    def store_transition(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.args.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        observation = torch.tensor(observation,dtype=torch.float)
        self.eval_net.eval()
        with torch.no_grad():
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * torch.max(actions_value)
        self.q.append(self.running_q)
        # 随着模型的学习随机chose的概率降低
        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.a_dim)
        return action, self.running_q

def train(args):
    RL = Double_DQN(args)
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action,rq = RL.choose_action(observation)

        f_action = (action-(args.n_action-1)/2)/((args.n_action-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > args.memory_size:   # learning
            loss = RL.learn()
            print(f"step {total_steps}:loss:{loss}|running q:{rq}")

        if total_steps - args.memory_size > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1
        env.render()
    return RL.q

if __name__ == '__main__':
    args = config()
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    DQ = train(args)
    env.close()
    plt.plot(np.array(DQ), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
