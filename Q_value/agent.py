import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma # 衰减系数
        self.epsilon = 0
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        ####################### 智能体的决策函数，需要完成Q表格方法（需要完成）#######################
        self.sample_count += 1
        action = np.random.choice(self.action_dim)  #随机探索选取一个动作
        tmp=self.Q_table[state][action]
        for i in range(len(self.Q_table[state])):
            if(self.Q_table[state][i]>tmp):
                tmp=self.Q_table[state][i]
                action=i
        return action

    def update(self, state, action, reward, next_state, done):
        

        self.Q_table[state,action]=max(self.Q_table[next_state,:])+reward

        
        pass

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
