from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm

from torch.distributions import Categorical
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")


class Policy(nn.Module):
    def __init__(
            self,
            obs_size: int,
            hiden_size: int,
            act_size: int
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hiden_size),
            nn.ReLU(),
            nn.Linear(hiden_size, hiden_size),
            nn.ReLU(),
            nn.Linear(hiden_size, act_size),
            nn.Softmax(dim=-1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)


class ValueFunc(nn.Module):
    def __init__(
            self,
            obs_size: int,
            hidden_size: int
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Collecter:
    def __init__(
            self,
            env,
            actor: Policy,
            t_size: int,
            n_size: int
    ) -> None:
        self.env = env
        self.actor = actor
        self.t_size = t_size
        self.n_size = n_size
        self.batch_list = None

    def collect(self) -> Sequence:
        self.batch_list = []
        for _ in range(self.n_size):
            obs, _ = self.env.reset()
            batch = []
            for _ in range(self.t_size):
                with torch.no_grad():
                    prob = self.actor.forward(torch.Tensor(obs))

                    cate = Categorical(prob)
                    action = cate.sample()
                    logp = cate.log_prob(action)

                    nxt_obs, rew, terminated, truncated, _ = self.env.step(action.item())
                    done = terminated or truncated

                    batch.append((obs, prob, action, rew, nxt_obs, logp, done))

                    obs = nxt_obs

                    if done:
                        self.env.reset()
                        break
            self.batch_list.append(batch)
        return self.batch_list


class PPO:
    def __init__(
            self,
            eps: float,
            actor: Policy,
            critic: ValueFunc,
            gamma: float,
            actor_lr: float = 1e-3,
            critic_lr: float = 1e-2
    ) -> None:
        super().__init__()
        self.eps = eps
        self.actor = actor
        self.critic = critic
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), critic_lr)
        self.gamma = gamma

    def learn(
            self,
            batch_list,
            lamda
    ):
        for batch in batch_list:
            batch_with_adv = self.GAE(batch, lamda)
            for tur in batch_with_adv:
                stat, prob, act, rew, nxtstat, logp_old, done, adv, ret = tur

                stat, nxtstat = torch.Tensor(stat), torch.Tensor(nxtstat)

                prob_new = self.actor(stat)
                loss_ent = Categorical(prob_new).entropy()

                loss_value = nn.functional.mse_loss(self.critic(stat), rew + self.gamma * self.critic(nxtstat))

                ratio = torch.exp(Categorical(prob_new).log_prob(act) - logp_old)
                clip_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                loss_clip = -torch.min(ratio * adv, clip_ratio * adv)

                # 两个loss共享计算图：1. 使用独立计算的两个计算图 2. 使用detach()切断梯度传播
                loss_all = loss_clip - 0.01 * loss_ent + 0.5 * loss_value.detach()

                self.actor_optim.zero_grad()
                loss_all.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                loss_value.backward()
                self.critic_optim.step()

    def GAE(self, batch, lamda) -> Sequence:
        n = len(batch)
        batch_with_adv = []

        # 提取所有状态和下一状态
        states = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([item[4] for item in batch]), dtype=torch.float32)
        # 计算所有状态和下一状态的值
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

        rewards = torch.tensor([item[3] for item in batch], dtype=torch.float32)
        dones = torch.tensor([item[6] for item in batch], dtype=torch.float32)

        # 初始化优势和GAE返回值
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_advantage = 0
        for i in reversed(range(n)):
            next_value = next_values[i]
            if dones[i]:
                next_value = 0
                last_advantage = 0
            delta = rewards[i] + self.gamma * next_value - values[i]
            advantages[i] = self.gamma * lamda * last_advantage + delta
            returns[i] = advantages[i] + values[i]
            last_advantage = advantages[i]

        # 将计算的优势和回报添加到批次数据中
        for i in range(n):
            stat, prob, act, rew, nxtstat, logp_old, done = batch[i]
            batch_with_adv.append((
                stat, prob, act, rew, nxtstat, logp_old, done,
                advantages[i].item(), returns[i].item()
            ))

        return batch_with_adv


def trainer(
        gym_name: str,
        epoch: int,
        steps_per_epoch: int,
        eps: float,
        hidden_size: int,
        t_size: int = 1024,
        n_size: int = 10,
        lamda: float = 0.98
) -> None:
    env = gym.make(gym_name)
    eval_env = gym.make(gym_name, render_mode="human")  # 不渲染评估环境
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    actor = Policy(obs_size, hidden_size, act_size)
    critic = ValueFunc(obs_size, hidden_size)
    ppo = PPO(eps, actor, critic, 0.99, actor_lr=1e-3, critic_lr=1e-2)
    collecter = Collecter(env, actor, t_size, n_size)

    for epo in range(epoch):
        pbar = tqdm(range(steps_per_epoch), desc=f"Epo{epo}", unit="steps")
        for i in pbar:
            batch_list = collecter.collect()
            ppo.learn(batch_list, lamda)
            if i == steps_per_epoch - 1:
                ret = evaluate(actor, eval_env, num_episodes=100)
                pbar.set_postfix({"ret": ret})

    env.close()
    eval_env.close()


def evaluate(agent: Policy, env, num_episodes=100) -> float:
    total_steps = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        steps = 0
        while True:
            with torch.no_grad():  # 评估时不计算梯度
                # 确定性选择动作
                prob = agent(torch.Tensor(obs))
                action = torch.argmax(prob).numpy()

            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        total_steps += steps
    return total_steps / num_episodes  # 返回平均步数


if __name__ == "__main__":
    # 影响性能的关键因素：超参数的选择是关键
    # actor_lr: 1e-3
    # critic_lr: 1e-2
    # epoch: 50
    # eps: 0.2
    # gamma: 0.99
    # lamda: 0.98
    trainer("CartPole-v1", 10, 50, 0.2, 128, 1024, 5, 0.98)
    # 训练结果：[12.55, 89.85, 483.17, 500.0, 491.65, 500.0, 362.84, 481.42, 500.0, 490.2]
