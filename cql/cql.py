import glob
import math
import pickle
import random
from collections import deque, namedtuple
from collections.abc import Sequence
import numpy as np
import psutil
import gymnasium as gym
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

writer = SummaryWriter("log/cql_sac")  # TensorBoard日志记录器


class DoubleQFunc(nn.Module):
    def __init__(
            self,
            stat_size: int,
            hidden_size: int,
            act_size: int,
            device: torch.device
    ):
        super().__init__()
        self.device = device
        self.net1 = nn.Sequential(
            nn.Linear(stat_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)  # 模型迁移到设备

        self.net2 = nn.Sequential(
            nn.Linear(stat_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)  # 模型迁移到设备

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x: Tensor, y: Tensor) -> Sequence[Tensor]:
        # 确保输入张量在正确设备上
        x = x.to(self.device)
        y = y.to(self.device)
        z = torch.cat((x, y), dim=-1)
        return self.net1(z), self.net2(z)


class PolicyNet(nn.Module):
    def __init__(
            self,
            stat_size: int,
            hidden_size: int,
            act_size: int,
            device: torch.device
    ):
        super().__init__()
        self.device = device
        self.net_pre = nn.Sequential(
            nn.Linear(stat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(self.device)  # 模型迁移到设备

        self.net_mean = nn.Linear(hidden_size, act_size).to(self.device)
        self.net_std = nn.Linear(hidden_size, act_size).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, stat: Tensor) -> Sequence[Tensor]:
        # 确保输入张量在正确设备上
        stat = stat.to(self.device)
        x = self.net_pre(stat)
        mean = self.net_mean(x)
        std = torch.exp(torch.clamp(self.net_std(x), -20, 2))  # 保证std严格大于0
        return mean, std

    def get_action(self, stat: Tensor) -> tuple:
        # 确保输入张量在正确设备上
        stat = stat.to(self.device)
        mean, std = self.forward(stat)
        normal = torch.distributions.Normal(mean, std)
        rsam = normal.rsample()
        action = F.tanh(rsam)  # 使用重参数化技巧
        # J行列式处理tanh的空间缩放
        logp = normal.log_prob(rsam) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp

    def get_eval_action(self, stat: Tensor) -> Tensor:
        stat = stat.to(self.device)
        mean, _ = self.forward(stat)
        return torch.tanh(mean).detach()


class ReplayBuffer:
    def __init__(self, device: torch.device):
        self.buffer = deque()
        self.length = 0
        self.device = device

    def push(
            self,
            stat,
            action,
            logp,
            rew,
            nxt_stat,
            done
    ):
        # 将数据迁移到设备并保存
        self.buffer.append((
            stat.to(self.device),
            action.to(self.device),
            logp.to(self.device),
            rew.to(self.device),
            nxt_stat.to(self.device),
            done.to(self.device)
        ))
        self.length += 1

    def get(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("缓冲区小于批次大小")

        index = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        batch = [self.buffer[i] for i in index]
        stat, action, logp, rew, nxt_stat, done = zip(*batch)
        return namedtuple(
            'ReplayBufferSample',
            ['stat', 'action', 'logp', 'rew', 'nxt_stat', 'done']
        )(
            torch.stack(stat),
            torch.stack(action),
            torch.stack(logp),
            torch.stack(rew),
            torch.stack(nxt_stat),
            torch.stack(done)
        )

    def save_to_file(self, file_path: str, i, size):
        """将缓冲区内容保存到文件（将张量转换为NumPy数组）"""
        # 转换缓冲区中的张量为NumPy数组（先转移到CPU）
        numpy_buffer = []
        for k in range(size):
            if i + k >= len(self.buffer):
                break
            item = self.buffer[i + k]
            stat, action, logp, rew, nxt_stat, done = item
            numpy_buffer.append((
                stat.cpu().numpy(),
                action.cpu().numpy(),
                logp.cpu().numpy(),
                rew.cpu().numpy(),
                nxt_stat.cpu().numpy(),
                done.cpu().numpy()
            ))
        # 使用pickle保存（支持列表和NumPy数组）
        with open(file_path, 'wb') as f:
            pickle.dump(numpy_buffer, f)

    def load_from_file(self, file_path: str):
        """从文件加载缓冲区内容（将NumPy数组转换为张量）"""
        with open(file_path, 'rb') as f:
            numpy_buffer = pickle.load(f)

        # 转换NumPy数组为张量并添加到缓冲区
        for item in numpy_buffer:
            stat_np, action_np, logp_np, rew_np, nxt_stat_np, done_np = item
            self.buffer.append((
                torch.tensor(stat_np, dtype=torch.float32, device=self.device),
                torch.tensor(action_np, dtype=torch.float32, device=self.device),
                torch.tensor(logp_np, dtype=torch.float32, device=self.device),
                torch.tensor(rew_np, dtype=torch.float32, device=self.device),
                torch.tensor(nxt_stat_np, dtype=torch.float32, device=self.device),
                torch.tensor(done_np, dtype=torch.int, device=self.device)
            ))
        self.length = len(self.buffer)


class CQLSAC:
    def __init__(
            self,
            optim_q1: optim.Optimizer,
            optim_q2: optim.Optimizer,
            optim_p: optim.Optimizer,
            env,
            alpha: float,
            gamma: float,
            policy: PolicyNet,
            qf1: DoubleQFunc,
            qf2: DoubleQFunc,
            tau: float,
            dim: int,
            policy_frequency: int,
            target_frequency: int,
            q_frquency: int,
            device: torch.device,
            repeat: int = 10,
            beta: float = 0.5,
            temp: float = 0.2
    ):
        self.device = device
        self.optim_q1: optim.Optimizer = optim_q1
        self.optim_q2: optim.Optimizer = optim_q2
        self.optim_p = optim_p
        self.env = env
        # 将alpha和目标熵迁移到设备
        self.alpha = tensor(alpha, dtype=torch.float32, device=self.device).detach()
        self.gamma = gamma
        self.policy = policy
        self.qf1: DoubleQFunc = qf1
        self.qf2: DoubleQFunc = qf2
        self.tau: float = tau

        self.target_ent = tensor(-dim, dtype=torch.float32, device=self.device)
        self.log_alpha = torch.log(self.alpha).clone().requires_grad_(True).to(self.device)
        self.optim_alpha = optim.Adam([self.log_alpha], lr=1e-4)
        self.global_steps = 0
        self.policy_frequency = policy_frequency
        self.target_frequency = target_frequency
        self.q_frequency = q_frquency
        self.repeat = repeat
        self.beta = beta
        self.temp = temp

    def learn(self, batch):
        stat, action, logp, rew, nxt_stat, done = batch
        rew = rew.unsqueeze(-1)
        done = done.unsqueeze(-1)

        self.global_steps = (self.global_steps + 1) % 1000000  # 全局步数

        if self.global_steps % self.q_frequency == 0:
            q1, q2 = self.qf1(stat, action)
            with torch.no_grad():
                nxt_action, nxt_logp = self.policy.get_action(nxt_stat)
                q1_target, q2_target = self.qf2(nxt_stat, nxt_action)

                q_target = torch.min(q1_target, q2_target)
                q_target = (1 - done) * self.gamma * (q_target - self.alpha * nxt_logp) + rew

            batch_size = stat.shape[0]
            # 生成随机动作并迁移到设备
            random_action = torch.rand(
                [self.repeat * batch_size, action.shape[-1]],
                dtype=torch.float,
                device=self.device
            ).uniform_(-1, 1)

            # 扩展状态维度
            temp_stat = stat.unsqueeze(1).repeat(1, self.repeat, 1).view(-1, stat.shape[1])
            temp_nxt_stat = nxt_stat.unsqueeze(1).repeat(1, self.repeat, 1).view(-1, nxt_stat.shape[1])
            temp_action, log_temp = self.policy.get_action(temp_stat)
            temp_nxt_action, log_temp_nxt = self.policy.get_action(temp_nxt_stat)

            # 计算各类Q值并重塑
            random_q1 = self.qf1(temp_stat, random_action)[0].reshape(batch_size, self.repeat, 1)
            random_q2 = self.qf1(temp_stat, random_action)[1].reshape(batch_size, self.repeat, 1)

            temp_q1 = self.qf1(temp_stat, temp_action)[0].reshape(batch_size, self.repeat, 1)
            temp_q2 = self.qf1(temp_stat, temp_action)[1].reshape(batch_size, self.repeat, 1)

            temp_nxt_q1 = self.qf1(temp_nxt_stat, temp_nxt_action)[0].reshape(batch_size, self.repeat, 1)
            temp_nxt_q2 = self.qf1(temp_nxt_stat, temp_nxt_action)[1].reshape(batch_size, self.repeat, 1)

            cur_q1 = self.qf1(stat, action)[0]
            cur_q2 = self.qf1(stat, action)[1]

            # 计算CQL损失（重要性采样校正）
            cat_q1 = torch.cat([
                random_q1 - math.log(0.5 ** action.shape[1]),
                temp_q1 - log_temp.reshape(batch_size, self.repeat, 1),
                temp_nxt_q1 - log_temp_nxt.reshape(batch_size, self.repeat, 1)
            ], dim=1)
            cat_q2 = torch.cat([
                random_q2 - math.log(0.5 ** action.shape[1]),
                temp_q2 - log_temp.reshape(batch_size, self.repeat, 1),
                temp_nxt_q2 - log_temp_nxt.reshape(batch_size, self.repeat, 1)
            ], dim=1)

            cql_q1_loss = (torch.logsumexp(cat_q1 / self.temp, dim=1) * self.beta * self.temp).mean() - cur_q1.mean()
            cql_q2_loss = (torch.logsumexp(cat_q2 / self.temp, dim=1) * self.beta * self.temp).mean() - cur_q2.mean()

            qloss_1 = F.mse_loss(q1, q_target) + self.beta * cql_q1_loss
            qloss_2 = F.mse_loss(q2, q_target) + self.beta * cql_q2_loss

            self.optim_q1.zero_grad()
            qloss_1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.qf1.net1.parameters(), 1)
            check_net(self.qf1, "q1", self.global_steps, "net1")
            self.optim_q1.step()

            self.optim_q2.zero_grad()
            qloss_2.backward()
            torch.nn.utils.clip_grad_norm_(self.qf1.net2.parameters(), 1)
            check_net(self.qf1, "q2", self.global_steps, "net2")
            self.optim_q2.step()

        if self.global_steps % self.policy_frequency == 0:
            # 训练策略网络
            new_action, new_logp = self.policy.get_action(stat)
            with torch.no_grad():
                q3, q4 = self.qf1(stat, new_action)
                q_min = torch.min(q3, q4)
            ploss = (self.alpha * new_logp - q_min).mean()

            self.optim_p.zero_grad()
            ploss.backward()
            check_net(self.policy, "policy", self.global_steps)
            self.optim_p.step()

            # 更新温度参数alpha
            aloss = (-self.log_alpha.exp() * (self.target_ent.detach() + new_logp.detach())).mean()
            self.optim_alpha.zero_grad()
            aloss.backward()
            writer.add_histogram("policy/alpha_grad", self.log_alpha.grad.clone().cpu().numpy(), self.global_steps)

            self.optim_alpha.step()
            self.alpha = torch.exp(self.log_alpha).detach()

        if self.global_steps % self.target_frequency == 0:
            # 软更新目标网络
            with torch.no_grad():
                for target_param, param in zip(self.qf2.parameters(), self.qf1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def collect(
            self,
            buffer: ReplayBuffer,
            n: int,
            t: int
    ):
        for _ in tqdm(range(n), desc="collect"):
            stat, _ = self.env.reset(seed=random.randint(0, 100))
            for _ in range(t):
                with torch.no_grad():
                    # 状态转为张量并迁移到设备
                    stat_tensor = Tensor(stat).to(self.device)
                    action, logp = self.policy.get_action(stat_tensor)
                    nxt_stat, rew, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy())
                    done = terminated or truncated

                    buffer.push(
                        tensor(stat, dtype=torch.float32),
                        action,
                        logp,
                        tensor(rew, dtype=torch.float32),
                        tensor(nxt_stat, dtype=torch.float32),
                        tensor(done, dtype=torch.int)
                    )

                    if done:
                        break
                    stat = nxt_stat


def evaluate(
        env: gym.Env,
        epo: int,
        policy: PolicyNet,
        device: torch.device
) -> float:
    ret = 0.0
    for _ in range(epo):
        stat, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                stat_tensor = Tensor(stat).to(device)
                action = policy.get_eval_action(stat_tensor)
            nxt_stat, rew, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            ret += rew
            stat = nxt_stat
    return ret / epo


def check_net(net: nn.Module, net_name, step: int, fliter=""):
    for name, param in net.named_parameters():
        if name.startswith(fliter):
            # writer.add_histogram(net_name + f"/{name}_grad", param.grad.clone().cpu().numpy(), step)
            writer.add_histogram(net_name + f"/{name}_data", param.data.clone().cpu().numpy(), step)


def is_resource_enough():
    """检查显存和内存是否充足"""
    # 检查系统内存
    sys_mem = psutil.virtual_memory()
    if sys_mem.available < 2.0:
        return False

    # 检查GPU显存（如果有GPU）
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        available_gpu = gpu_mem - (allocated + (reserved - allocated))  # 实际可用显存

        if available_gpu < 1.0:
            return False

    return True


def trainer(
        env_name: str = "Pendulum-v1",
        hidden_size: int = 64,
        epos: int = 100000,
        steps: int = 200,
        tau: float = 5e-3,
        batch_size: int = 128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    stat_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    qf1 = DoubleQFunc(stat_size, hidden_size, act_size, device)
    qf2 = DoubleQFunc(stat_size, hidden_size, act_size, device)
    qf2.load_state_dict(qf1.state_dict())  # 初始化目标网络
    policy = PolicyNet(stat_size, hidden_size, act_size, device)

    # load_stat
    if glob.glob("model/best_qf1.pth"):
        qf1.load_state_dict(torch.load("model/best_qf1.pth", map_location=device))
    if glob.glob("model/best_qf2.pth"):
        qf2.load_state_dict(torch.load("model/best_qf2.pth", map_location=device))
    if glob.glob("model/best_policy.pth"):
        policy.load_state_dict(torch.load("model/best_policy.pth", map_location=device))
        print("load policy")

    optim_q1 = optim.Adam(qf1.net1.parameters(), lr=3e-4)
    optim_q2 = optim.Adam(qf1.net2.parameters(), lr=3e-4)
    optim_p = optim.Adam(policy.parameters(), lr=3e-4)

    cqlsac = CQLSAC(
        optim_q1, optim_q2, optim_p, env, 0.2, 0.99,
        policy, qf1, qf2, tau, act_size, 2, 1, 1,
        device=device, repeat=10, beta=2, temp=0.2
    )

    buffer = ReplayBuffer(device)
    cqlsac.collect(buffer, 200, 200)

    best_ret = -float('inf')
    for epo in range(epos):
        pbar = tqdm(range(steps), desc=f"train{epo}")
        for i in pbar:
            batch = buffer.get(batch_size)
            cqlsac.learn(batch)
            if i == steps - 1:
                ret = evaluate(eval_env, 10, policy, device)
                pbar.set_postfix({"ret": ret})
                if ret > best_ret:
                    best_ret = ret
                    torch.save(policy.state_dict(), "model/best_policy.pth")
                    torch.save(qf1.state_dict(), "model/best_qf1.pth")
                    torch.save(qf2.state_dict(), "model/best_qf2.pth")
                if epo % 4 == 0:
                    torch.save(policy.state_dict(), "model/cur_policy.pth")
                    torch.save(qf1.state_dict(), "model/cur_qf1.pth")
                    torch.save(qf2.state_dict(), "model/cur_qf2.pth")
        if is_resource_enough() and epo <= 85:
            batch_file = f"replay/batch-{epo}.pkl"
            if glob.glob(batch_file):
                buffer.load_from_file(batch_file)
            else:
                buffer.save_to_file(batch_file, epo * 200, 200)
                cqlsac.collect(buffer, 200, 200)


if __name__ == "__main__":

    trainer()
