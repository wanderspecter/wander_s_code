from collections import deque
from collections.abc import Sequence
import numpy as np

import gym
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
from tqdm import tqdm


class DoubleQFunc(nn.Module):
    def __init__(
            self,
            stat_size: int,
            hidden_size: int,
            act_size: int
    ):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(stat_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.net2 = nn.Sequential(
            nn.Linear(stat_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x: Tensor, y: Tensor) -> Sequence[Tensor]:
        z = torch.cat((x, y), dim=-1)
        return self.net1(z), self.net2(z)


class PolicyNet(nn.Module):
    def __init__(
            self,
            stat_size: int,
            hidden_size: int,
            act_size: int
    ):
        super().__init__()
        self.net_pre = nn.Sequential(
            nn.Linear(stat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.net_mean = nn.Linear(hidden_size, act_size)
        self.net_std = nn.Linear(hidden_size, act_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, stat: Tensor) -> Sequence[Tensor]:
        # 动作输出的最后一层不能使用relu函数，会消除负项
        x = self.net_pre(stat)
        mean = self.net_mean(x)
        std = torch.exp(torch.clamp(self.net_std(x), -20, 2))  # 保证std严格大于0, 并限制std的大小
        return mean, std

    def get_action(self, stat: Tensor) -> tuple:
        mean, std = self.forward(stat)
        normal = torch.distributions.Normal(mean, std)
        rsam = normal.rsample()
        action = F.tanh(rsam)  # 使用重参数化技巧
        # J行列式处理tanh的空间缩放，加入小偏差1e-6使不为0
        logp = normal.log_prob(rsam) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()
        self.length = 0

    def push(
            self,
            stat,
            action,
            logp,
            rew,
            nxt_stat,
            done
    ):
        self.buffer.append((
            stat, action, logp, rew, nxt_stat, done
        ))
        self.length += 1

    def get(self, batch_size):
        index = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        batch = [self.buffer[i] for i in index]
        stat, action, logp, rew, nxt_stat, done = zip(*batch)
        return (
            torch.stack(stat),
            torch.stack(action),
            torch.stack(logp),
            torch.stack(rew),
            torch.stack(nxt_stat),
            torch.stack(done)
        )


class SAC:
    def __init__(
            self,
            optim_q: optim.Optimizer,
            optim_p: optim.Optimizer,
            env,
            alpha: float,
            gamma: float,
            policy,
            qf1,
            qf2,
            tau,
            dim
    ):
        self.optim_q = optim_q
        self.optim_p = optim_p
        self.env = env
        self.alpha = tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.policy = policy
        self.qf1: DoubleQFunc = qf1
        self.qf2: DoubleQFunc = qf2
        self.tau: float = tau

        self.target_ent = tensor(-dim, dtype=torch.float32)
        self.log_alpha = tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True)
        self.optim_alpha = optim.Adam([self.log_alpha], lr=1e-4)

    def learn(self, batch):
        stat, action, logp, rew, nxt_stat, done = batch
        rew = rew.unsqueeze(-1)
        done = done.unsqueeze(-1)
        # vloss、qloss、ploss共享一部分梯度计算图，使用detach()，或者计算时torch.no_grad()
        # logp也包含计算图，因为使用了重参数化
        # 因为用到了rew，所以采用历史数据的action
        q1, q2 = self.qf1(stat, action)
        with torch.no_grad():
            nxt_action, nxt_logp = self.policy.get_action(nxt_stat)
            q1_target, q2_target = self.qf2(nxt_stat, nxt_action)

            q_target = torch.min(q1_target, q2_target)

            # 1 - done: 标量与张量运算，标量自动广播
            q_target = (1 - done) * self.gamma * (q_target - self.alpha * nxt_logp) + rew

        qloss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.optim_q.zero_grad()
        qloss.backward()
        self.optim_q.step()

        # 训练策略，不使用rew，用最新动作训练
        new_action, new_logp = self.policy.get_action(stat)
        with torch.no_grad():
            q3, q4 = self.qf1(stat, new_action)
            q_min = torch.min(q3, q4)
        ploss = (self.alpha * new_logp - q_min).mean()  # 使用mean求平均使其变为标量

        self.optim_p.zero_grad()
        ploss.backward()
        self.optim_p.step()

        # 6. 软更新目标网络（保持不变）
        with torch.no_grad():
            for target_param, param in zip(self.qf2.parameters(), self.qf1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        aloss = (-self.log_alpha * (self.target_ent + new_logp).detach()).mean()
        self.optim_alpha.zero_grad()
        aloss.backward()
        self.optim_alpha.step()
        self.alpha = torch.exp(self.log_alpha).detach()

    def collect(
            self,
            buffer,
            n,
            t
    ):
        for _ in range(n):
            stat, _ = self.env.reset()
            for _ in range(t):
                # 在收集数据时关闭梯度计算，防止出现问题
                with torch.no_grad():
                    action, logp = self.policy.get_action(Tensor(stat))
                    nxt_stat, rew, terminated, truncated, _ = self.env.step(action.detach().numpy())
                    done = terminated or truncated
                    buffer.push(tensor(stat, dtype=torch.float32),
                                action, logp, tensor(rew/10.0, dtype=torch.float32),
                                tensor(nxt_stat, dtype=torch.float32), tensor(done, dtype=torch.int))

                    if done:
                        break
                    stat = nxt_stat


def evaluate(
        env_name: str,
        epo: int,
        policy: PolicyNet
) -> float:
    env = gym.make(env_name)
    ret = 0.0
    for _ in range(epo):
        stat, _ = env.reset()
        done = 0
        while not done:
            mean, std = policy.forward(Tensor(stat))
            nxt_stat, rew, terminated, truncated, _ = env.step(mean.detach().numpy())
            done = terminated or truncated
            ret += rew
            stat = nxt_stat
    return ret / epo


def trainer(
        env_name: str = "Pendulum-v1",
        hidden_size: int = 128,
        epo: int = 10,
        steps: int = 1000,
        tau: float = 0.001,
        batch_size: int = 256
):
    env = gym.make(env_name)
    stat_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    qf1 = DoubleQFunc(stat_size, hidden_size, act_size)
    qf2 = DoubleQFunc(stat_size, hidden_size, act_size)
    policy = PolicyNet(stat_size, hidden_size, act_size)

    optim_q = optim.Adam(qf1.parameters(), lr=1e-4)
    optim_p = optim.Adam(policy.parameters(), lr=1e-4)

    sac = SAC(optim_q, optim_p, env, 0.2, 0.99, policy, qf1, qf2, tau, act_size)
    buffer = ReplayBuffer()

    sac.collect(buffer, 1000, 200)

    for epo in range(epo):
        pbar = tqdm(range(steps), desc=f"train{epo}")
        for i in pbar:
            batch = buffer.get(batch_size)
            sac.learn(batch)
            ret = evaluate(env_name, 10, policy)
            pbar.set_postfix({"ret": ret})


if __name__ == "__main__":
    trainer()
