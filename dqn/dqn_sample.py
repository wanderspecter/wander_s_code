import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gym
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import imageio

# 设置随机种子以保证结果可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# 定义DQN网络结构
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()  # 调用父类构造函数初始化神经网络
        self.fc1 = nn.Linear(state_dim, 128)  # 输入层：状态维度 → 128维隐藏层
        self.fc2 = nn.Linear(128, 128)  # 隐藏层：128维 → 128维
        self.fc3 = nn.Linear(128, action_dim)  # 输出层：128维 → 动作维度（CartPole是2个动作）

    # 定义神经网络的前向传播过程
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层：全连接层 + ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层：全连接层 + ReLU激活函数
        return self.fc3(x)  # 输出层：直接返回原始Q值（未使用激活函数）


# 定义经验回放类
class ReplayBuffer:
    # 初始化经验回放缓冲区，设置最大容量
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 从缓冲区随机采样一个批次的训练数据
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


# 定义DQN算法类
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # 设备选择（优先使用GPU加速）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建策略网络（用于决策）和目标网络（用于稳定训练），并将它们复制到设备上
        self.policy_net = DQN(state_dim, action_dim).to(self.device)  # 训练用网络
        self.target_net = DQN(state_dim, action_dim).to(self.device)  # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 同步初始参数
        self.target_net.eval()  # 固定目标网络为评估模式（不计算梯度）

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)  # 优化器配置（Adam优化器，学习率0.001
        self.replay_buffer = ReplayBuffer(capacity=10000)  # 经验回放配置（容量10000条经验）
        self.batch_size = 64  # 每次训练的样本数量
        self.gamma = 0.99  # 未来奖励折扣因子
        self.tau = 0.005  # 目标网络软更新系数
        self.action_dim = action_dim  # 保存动作空间维度

    def select_action(self, state, epsilon):
        if random.random() < epsilon:  # ε-贪婪策略选择动作
            return random.randint(0, self.action_dim - 1)
        else:
            # 将numpy数组转换为PyTorch张量，并添加批次维度（unsqueeze(0)）
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 禁用梯度计算（提升推理效率，节省内存）
            with torch.no_grad():
                q_values = self.policy_net(state)  # 通过策略网络获取所有动作的Q值

            # 选择最大Q值对应的动作索引，并转换为Python标量
            return q_values.argmax().item()

    def update(self):
        # 检查经验回放缓冲区是否有足够样本
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区采样一个批次的训练数据
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # 将numpy数组转换为PyTorch张量，并将它们移动到设备上
        state = torch.FloatTensor(state).to(self.device)  # 形状：[batch_size, state_dim]
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)  # 形状：[batch_size, 1]
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # 形状：[batch_size, 1]
        next_state = torch.FloatTensor(next_state).to(self.device)  # 形状：[batch_size, state_dim]
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)  # 形状：[batch_size, 1]

        # 计算当前Q值
        current_q = self.policy_net(state).gather(1, action)

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)  # 取最大Q值，形状：[batch_size, 1]
            target_q = reward + self.gamma * next_q * (1 - done)  # Bellman方程，形状：[batch_size, 1]

        # 计算损失并优化
        loss = F.mse_loss(current_q, target_q)

        # 梯度清零 + 反向传播 + 参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络参数
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def save_model(self, path):
        # 保存策略网络参数到指定路径
        # 使用state_dict()只保存网络参数（而非整个模型），便于跨设备加载和版本控制
        torch.save(self.policy_net.state_dict(), path)  # 参数保存格式：PyTorch的二进制文件

    def load_model(self, path):
        # 从指定路径加载训练好的模型参数
        # 同时更新策略网络和目标网络，保持参数一致性
        self.policy_net.load_state_dict(torch.load(path))  # 训练中断后恢复模型
        self.target_net.load_state_dict(torch.load(path))  # 同步目标网络参数


def visualize_rewards_as_gif(episodes, rewards, filename='training_rewards.gif', interval=10):
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Training Reward Progress')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.grid(True)

    # 设置坐标轴范围
    ax.set_xlim(0, max(episodes))
    ax.set_ylim(min(rewards) * 0.9, max(rewards) * 1.1)

    # 生成帧数据
    frames = []
    for i in range(len(episodes)):
        line, = ax.plot(episodes[:i + 1], rewards[:i + 1], 'r-', lw=2)
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
        line.remove()

    # 保存为GIF
    imageio.mimsave(filename, frames, duration=interval)

    # 关闭图形
    plt.close()


# 测试DQN算法
def train_dqn():
    # 创建环境
    env = gym.make('CartPole-v1', render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建DQN智能体
    agent = DQNAgent(state_dim, action_dim)

    # 训练参数
    episodes = 300
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    # 数据记录
    episode_list = []
    reward_list = []

    # 训练循环
    for episode in range(episodes):
        state, _ = env.reset()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode * epsilon_decay)
        total_reward = 0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.update()

            if done:
                episode_list.append(episode)
                reward_list.append(total_reward)
                print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
                break

    # 保存模型
    agent.save_model("dqn_model.pth")

    # 关闭环境
    env.close()
    # visualize_rewards_as_gif(episode_list, reward_list)


def t_dqn(save_gif=False):
    # 初始化环境和智能体
    agent = DQNAgent(state_dim=4, action_dim=2)  # CartPole的固定维度
    # agent.load_model("dqn_model.pth")

    test_env = gym.make('CartPole-v1', render_mode='rgb_array' if save_gif else 'human')

    # GIF相关配置
    frames = []

    state, _ = test_env.reset()
    total_reward = 0

    while True:
        # 渲染并捕获帧
        frame = test_env.render()
        if save_gif:
            frames.append(Image.fromarray(frame))

        action = agent.select_action(state, epsilon=0)
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

        if done:
            print(f"Test DQN: Reward={total_reward}")
            # 保存GIF
            if save_gif and frames:
                titled_test = []
                for frame in frames:
                    draw = ImageDraw.Draw(frame)
                    draw.text((50, 10), "Final Strategy Testing", fill=(0, 255, 0))
                    titled_test.append(frame)

                titled_test[0].save('dqn_test_result.gif',
                                    save_all=True,
                                    append_images=titled_test[1:],
                                    duration=50,
                                    loop=0)
            break

    test_env.close()


if __name__ == "__main__":
    train_dqn()
    t_dqn(save_gif=True)
