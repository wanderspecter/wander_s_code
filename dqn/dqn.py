import warnings
import random
from collections import deque
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

class DQN(nn.Module):
    def __init__(
            self,
            stat_size: int,
            hidden_size: int,
            act_size: int
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_size)  # 输出的是act_size个最大Q值，不是概率，不能softmax
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_action(self, x: Tensor):
        y = self(x)
        return torch.argmax(y, dim=-1)  # 返回最大Q值对应的动作


def evaluate(agent: DQN, env, num_episodes=2) -> float:
    total_steps = 0
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=random.randint(0, 100))
        while True:
            with torch.no_grad():  # 评估时不计算梯度
                # 确定性选择动作
                action = agent.get_action(Tensor(obs))

            obs, _, terminated, truncated, _ = env.step(action.item())
            total_steps += 1
            if terminated or truncated:
                break
    return total_steps / num_episodes  # 返回平均步数


class QLearn:
    def __init__(
            self,
            dqn: DQN,
            target_dqn: DQN,
            device: torch.device,
            optim: torch.optim.Optimizer,
            alpha: float,
            env: gym.Env,
            eval_env: gym.Env
    ):
        super().__init__()
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.device = device
        self.optim = optim
        self.alpha = alpha
        self.env = env
        self.eval_env = eval_env

    def learn(
            self,
            batch: Sequence
    ):
        stats, acts, rews, nxt_stats, dones = batch

        stats = stats.to(self.device)
        acts = acts.to(self.device)
        rews = rews.to(self.device)  # 奖励也转换为列向量
        nxt_stats = nxt_stats.to(self.device)
        dones = dones.to(self.device)  # done也转换为列向量

        qv = self.dqn(stats).gather(dim=1, index=acts)
        with torch.no_grad():
            target_qv = rews + self.target_dqn(nxt_stats).max(dim=1, keepdim=True)[0] * (1 - dones)
        loss = F.mse_loss(qv, target_qv)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        for target_dqn_para, dqn_para in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_dqn_para.data.copy_(
                self.alpha * dqn_para + (1 - self.alpha) * target_dqn_para
            )

    def collect(self, buffer, n, t, eps):
        for _ in range(n):
            stat, _ = self.env.reset(seed=random.randint(0, 100))
            for _ in range(t):
                with torch.no_grad():
                    if random.random() < eps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.dqn.get_action(Tensor(stat)).item()
                nxt_stat, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                buffer.push(stat, action, rew, nxt_stat, done)

                if done:
                    break
                stat = nxt_stat


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()
        self.length = 0

    def clear(self):
        self.buffer.clear()
        self.length = 0

    def push(
            self,
            stat,
            action,
            rew,
            nxt_stat,
            done
    ):
        self.buffer.append((
            stat, action, rew, nxt_stat, done
        ))
        self.length += 1

    def sample(self, batch_size):
        index = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        batch = [self.buffer[i] for i in index]
        stat, action, rew, nxt_stat, done = zip(*batch)
        return (
            torch.Tensor(np.array(stat)),
            torch.LongTensor(np.array(action)).unsqueeze(1),  # 动作转换为列向量
            torch.Tensor(np.array(rew)).unsqueeze(1),  # 奖励转换为列向量
            torch.Tensor(np.array(nxt_stat)),
            torch.Tensor(np.array(done)).unsqueeze(1)  # done转换为列向量
        )


def trainer(
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        env_name: str = "CartPole-v1",
        hidden_size: int = 256,
        lr: float = 1e-4,
        alpha: float = 0.001,
        epos: int = 20,
        steps: int = 1000,
        n: int = 128,
        t: int = 512,
        batch_size: int = 64, # 重点：batch_size不能太大
        eps: float = 0.1
) -> None:
    env = gym.make(env_name)
    eval_env = gym.make(env_name, render_mode="human")  # 不渲染评估环境
    stat_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    dqn = DQN(stat_size, hidden_size, act_size).to(device)
    target_dqn = DQN(stat_size, hidden_size, act_size).to(device)
    optim = torch.optim.Adam(dqn.parameters(), lr)

    qlearn = QLearn(dqn, target_dqn, device, optim, alpha, env, eval_env)
    buffer = ReplayBuffer()
    buffer.clear()

    for epo in range(epos):
        pbar = tqdm(range(steps), desc=f"Epo{epo}", unit="step")
        qlearn.collect(buffer, n, t, eps)
        for i in pbar:
            batch = buffer.sample(batch_size)  # 随着训练采样新的训练数据，on-policy方法提升效果
            qlearn.learn(batch)

            if i == steps - 1:
                ret = evaluate(dqn, eval_env, num_episodes=2)
                pbar.set_postfix({"ret": ret})


if __name__ == "__main__":
    trainer()
