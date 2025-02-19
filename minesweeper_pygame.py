import pygame
import numpy as np
import random
from time import sleep

class Minesweeper:
    def __init__(self, grid_width=10, grid_height=10, num_mines=15, tile_size=30, display_mode=False):
        self.GRID_WIDTH = grid_width
        self.GRID_HEIGHT = grid_height
        self.NUM_MINES = num_mines
        self.TILE_SIZE = tile_size
        self.display_mode = display_mode

        self.MINE = -1  # 地雷
        self.UNREVEALED = 9  # 未揭露

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (192, 192, 192)
        self.RED = (255, 0, 0)

        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.revealed = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), False, dtype=bool)
        self.game_over = False
        self.win = False
        self.last_reveal_position = None  # 记录最后一次揭露的位置
        self.last_reveal_status = None  # 记录最后一次揭露的状态
        self.first_click = True

        if self.display_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE))
            pygame.display.set_caption("Minesweeper")
            self.font = pygame.font.SysFont(None, 24)

    def reset_game(self):
        """重置游戏到初始状态"""
        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.revealed = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), False, dtype=bool)
        self.game_over = False
        self.win = False
        self.last_reveal_position = None
        self.last_reveal_status = None
        self.first_click = True
        if self.display_mode:
            self.screen.fill(self.GRAY)

    def place_mines(self, exclude_x, exclude_y):
        mines = []
        while len(mines) < self.NUM_MINES:
            x = random.randint(0, self.GRID_HEIGHT - 1)
            y = random.randint(0, self.GRID_WIDTH - 1)
            if (x, y) != (exclude_x, exclude_y) and (x, y) not in mines:
                mines.append((x, y))
                self.board[x, y] = self.MINE  # 地雷标记为MINE

        # 设置数字
        for x in range(self.GRID_HEIGHT):
            for y in range(self.GRID_WIDTH):
                if self.board[x, y] == self.MINE:
                    continue
                count = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_HEIGHT and 0 <= ny < self.GRID_WIDTH and self.board[nx, ny] == self.MINE:
                            count += 1
                self.board[x, y] = count

    def draw_board(self):
        if not self.display_mode:
            return
        for x in range(self.GRID_HEIGHT):
            for y in range(self.GRID_WIDTH):
                rect = pygame.Rect(y * self.TILE_SIZE, x * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.revealed[x, y]:
                    pygame.draw.rect(self.screen, self.WHITE, rect)  # 绘制已揭露的砖块背景
                    pygame.draw.rect(self.screen, self.GRAY, rect, 2)  # 绘制砖块边框
                    if self.board[x, y] > 0:
                        text = self.font.render(str(self.board[x, y]), True, self.BLACK)  # 显示数字
                        self.screen.blit(text, (y * self.TILE_SIZE + 10, x * self.TILE_SIZE + 5))
                    elif self.board[x, y] == self.MINE:
                        pygame.draw.circle(self.screen, self.RED, rect.center, self.TILE_SIZE // 4)  # 显示地雷
                else:
                    pygame.draw.rect(self.screen, self.GRAY, rect)  # 绘制未揭露的砖块背景
                    pygame.draw.rect(self.screen, self.BLACK, rect, 2)  # 绘制砖块边框

    def reveal_tile(self, x, y):

        self.last_reveal_position = (x, y)  # 记录最后一次揭露的位置

        if self.revealed[x, y]:
            self.last_reveal_status = "revealed"
            return

        if self.first_click:
            self.place_mines(x, y)
            self.first_click = False
            self.last_reveal_status = "first"
        else:
            self.last_reveal_status = "unrevealed"
        self.revealed[x, y] = True
        if self.board[x, y] == self.MINE:  # 踩到地雷
            self.game_over = True
            return

        if self.board[x, y] == 0:  # 空白格子，递归揭露周围格子
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_HEIGHT and 0 <= ny < self.GRID_WIDTH and not self.revealed[nx, ny]:
                        self.reveal_tile(nx, ny)

        # 检查是否赢了
        if np.sum(self.revealed) == self.GRID_WIDTH * self.GRID_HEIGHT - self.NUM_MINES:
            self.win = True
            self.game_over = True

    def get_game_board(self):
        return np.where(self.revealed, self.board, self.UNREVEALED)  # 未揭露的区域用UNREVEALED表示

    def get_game_status(self):
        if self.game_over:
            return "win" if self.win else "lose"
        return "ongoing"

    def get_last_revealed(self):
        return self.last_reveal_status

    def automated_play(self):
        running = True
        while running:
            if self.display_mode:
                self.screen.fill(self.GRAY)  # 清空屏幕，填充背景颜色
                self.draw_board()  # 调用绘制棋盘函数
                pygame.display.flip()  # 更新显示内容

            if not self.game_over:
                unrevealed_positions = [(x, y) for x in range(self.GRID_HEIGHT) for y in range(self.GRID_WIDTH) if not self.revealed[x, y]]
                if unrevealed_positions:
                    x, y = random.choice(unrevealed_positions)  # 随机选择一个未揭露的位置
                    self.reveal_tile(x, y)  # 揭露该位置
                    if not self.display_mode:
                        print(f"Revealed tile at ({x}, {y})")
                        print("Current Board:\n", self.get_game_board())
                        print("Game Status:", self.get_game_status())
                else:
                    running = False

            if self.get_game_status() != "ongoing":
                running = False  # 结束游戏循环
                if not self.display_mode:
                    print("Final Status:", self.get_game_status())
                    print("Final Board:\n", self.get_game_board())

            if self.display_mode:
                sleep(0.5)  # 控制自动执行速度，每次操作间隔0.5秒

        if self.display_mode:
            pygame.quit()  # 退出pygame

if __name__ == "__main__":
    game = Minesweeper(display_mode=True)
    game.automated_play()
