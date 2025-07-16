import torch
import numpy as np
from collections import deque, namedtuple
from torch import Tensor


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque()
        self.length = 0

    def add(
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
        return namedtuple(
            'ReplayBufferSample',
            ['stat', 'action', 'rew', 'nxt_stat', 'done']
        )(
            stat=torch.Tensor(np.array(stat)).squeeze(1),
            action=torch.LongTensor(np.array(action)).squeeze(1),
            rew=torch.Tensor(np.array(rew)),
            nxt_stat=torch.Tensor(np.array(nxt_stat)).squeeze(1),
            done=torch.Tensor(np.array(done))
        )
