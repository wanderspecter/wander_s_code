from time import sleep

import tyro
from dataclasses import dataclass
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Any, Optional
import pygame
from pygame import gfxdraw


class GomokuEnv(gym.Env):
    def __init__(self, board_width=15, board_height=15, n_in_row=5):
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.action_space = spaces.Discrete(board_width * board_height)
        self.observation_space = spaces.Box(0, 1, shape=(4, board_width, board_height))

        # Pygame 初始化
        self.window_size = 800
        self.cell_size = self.window_size // max(board_width, board_height)
        self.window_size = self.cell_size * max(board_width, board_height)
        self.radius = int(self.cell_size * 0.4)
        self.padding = self.cell_size // 2

        self.screen = None
        self.clock = None
        self.isopen = True

        self.reset()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_width, self.board_height), dtype=int)
        self.current_player = 1
        self.last_move = None
        return self._get_obs()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        x, y = divmod(action, self.board_width)

        if self.board[x, y] != 0:
            return self._get_obs(), -1, True, {"error": "Invalid move"}

        self.board[x, y] = self.current_player
        self.last_move = (x, y)

        done = self._check_winner(x, y)

        if done:
            reward = 1
        elif np.all(self.board != 0):
            reward = 0
            done = True
        else:
            reward = 0

        self.current_player = 3 - self.current_player

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = np.zeros((4, self.board_width, self.board_height), dtype=np.float32)
        obs[0] = (self.board == self.current_player).astype(np.float32)
        obs[1] = (self.board == (3 - self.current_player)).astype(np.float32)
        if self.last_move:
            obs[2, self.last_move[0], self.last_move[1]] = 1.0
        obs[3] = np.full((self.board_width, self.board_height),
                         (self.current_player - 1) * 0.5,
                         dtype=np.float32)
        return obs

    def _check_winner(self, x, y):
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            tx, ty = x + dx, y + dy
            while 0 <= tx < self.board_width and 0 <= ty < self.board_height and self.board[tx, ty] == player:
                count += 1
                tx += dx
                ty += dy
            tx, ty = x - dx, y - dy
            while 0 <= tx < self.board_width and 0 <= ty < self.board_height and self.board[tx, ty] == player:
                count += 1
                tx -= dx
                ty -= dy
            if count >= self.n_in_row:
                return True
        return False

    def render(self, mode='human'):
        if self.screen is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('Gomoku')
            self.clock = pygame.time.Clock()

        if self.screen is None:
            return

        # 绘制背景
        self.screen.fill((220, 179, 92))  # 木质棋盘颜色

        # 绘制网格线
        for i in range(self.board_width):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.padding, self.padding + i * self.cell_size),
                (self.window_size - self.padding, self.padding + i * self.cell_size),
                2
            )

        for j in range(self.board_height):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.padding + j * self.cell_size, self.padding),
                (self.padding + j * self.cell_size, self.window_size - self.padding),
                2
            )

        # 绘制棋子
        for i in range(self.board_width):
            for j in range(self.board_height):
                if self.board[i, j] == 1:  # 玩家1（黑子）
                    pygame.gfxdraw.filled_circle(
                        self.screen,
                        self.padding + j * self.cell_size,
                        self.padding + i * self.cell_size,
                        self.radius,
                        (0, 0, 0)
                    )
                    pygame.gfxdraw.aacircle(
                        self.screen,
                        self.padding + j * self.cell_size,
                        self.padding + i * self.cell_size,
                        self.radius,
                        (0, 0, 0)
                    )
                elif self.board[i, j] == 2:  # 玩家2（白子）
                    pygame.gfxdraw.filled_circle(
                        self.screen,
                        self.padding + j * self.cell_size,
                        self.padding + i * self.cell_size,
                        self.radius,
                        (255, 255, 255)
                    )
                    pygame.gfxdraw.aacircle(
                        self.screen,
                        self.padding + j * self.cell_size,
                        self.padding + i * self.cell_size,
                        self.radius,
                        (0, 0, 0)
                    )

        # 标记最后一步
        if self.last_move:
            x, y = self.last_move
            pygame.draw.circle(
                self.screen,
                (255, 0, 0) if self.board[x, y] == 1 else (0, 0, 255),
                (self.padding + y * self.cell_size, self.padding + x * self.cell_size),
                self.radius // 3
            )

        # 显示当前玩家
        font = pygame.font.Font(None, 36)
        text = font.render(f"Current Player: {'Black (1)' if self.current_player == 1 else 'White (2)'}",
                           True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        if mode == 'human':
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(60)
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class GomokuEnv(gym.Env):
    def __init__(self, start_player=0):
        self.start_player = start_player

        self.action_space = Discrete((board_width * board_height))
        self.observation_space = Box(0, 1, shape=(4, board_width, board_height))
        self.reward = 0
        self.info = {}
        self.players = [1, 2]  # player1 and player2

    def step(self, action):
        self.states[action] = self.current_player
        if action in self.availables:
            self.availables.remove(action)

        self.last_move = action

        done, winner = self.game_end()
        reward = 0
        if done:
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1

        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

        # update state
        obs = self.current_state()

        return obs, reward, done, self.info

    def reset(self):
        if board_width < n_in_row or board_height < n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(n_in_row))
        self.current_player = self.players[self.start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(board_width * board_height))
        self.states = {}
        self.last_move = -1

        return self.current_state()

    def render(self, mode='human', start_player=0):
        width = board_width
        height = board_height

        p1, p2 = self.players

        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = self.states.get(loc, -1)
                if p == p1:
                    print('B'.center(8), end='')
                elif p == p2:
                    print('W'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def has_a_winner(self):
        states = self.states
        moved = list(set(range(board_width * board_height)) - set(self.availables))
        if len(moved) < n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // board_width
            w = m % board_width
            player = states[m]

            if (w in range(board_width - n_in_row + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n_in_row))) == 1):
                return True, player

            if (h in range(board_height - n_in_row + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n_in_row * board_width, board_width))) == 1):
                return True, player

            if (w in range(board_width - n_in_row + 1) and h in range(board_height - n_in_row + 1) and
                    len(set(
                        states.get(i, -1) for i in range(m, m + n_in_row * (board_width + 1), board_width + 1))) == 1):
                return True, player

            if (w in range(n_in_row - 1, board_width) and h in range(board_height - n_in_row + 1) and
                    len(set(
                        states.get(i, -1) for i in range(m, m + n_in_row * (board_width - 1), board_width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            # print("winner is player{}".format(winner))
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((4, board_width, board_height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // board_width,
                            move_curr % board_height] = 1.0
            square_state[1][move_oppo // board_width,
                            move_oppo % board_height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // board_width,
                            self.last_move % board_height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def start_play(self, player1, player2, start_player=0):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.reset()
        p1, p2 = self.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        while True:
            player_in_turn = players[self.current_player]
            move = player_in_turn.get_action(self)
            self.step(move)
            end, winner = self.game_end()
            if end:
                return winner

    def start_self_play(self, player):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.reset()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self, return_prob=1)
            # store the data
            states.append(self.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.current_player)
            # perform a move
            self.step(move)
            end, winner = self.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                return winner, zip(states, mcts_probs, winners_z)

    def location_to_move(self, location):
        if (len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * board_width + w
        if (move not in range(board_width * board_width)):
            return -1
        return move

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // board_width
        w = move % board_width
        return [h, w]



@dataclass
class Args:
    pass



def trainer():
    tyro.cli(Args, description="Monte Carlo Tree Search (MCTS) Trainer")



if __name__ == "__main__":
    trainer()
    env = GomokuEnv()
    obs = env.reset()
    env.render()  # 显示Pygame窗口

    # 执行一些动作
    for _ in range(5000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        sleep(1)  # 控制渲染速度

        if done:
            break

    env.close()
