import time
from minesweeper_pygame import Minesweeper
from tqdm import tqdm
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

# Create Minesweeper Environment
class MinesweeperEnv:
    def __init__(self, board_size=10, num_mines=10, display_mode=False):
        self.board_size = board_size
        self.num_mines = num_mines
        self.display_mode = display_mode
        self.ms = Minesweeper(grid_width=board_size, grid_height=board_size, num_mines=num_mines, display_mode=display_mode)

    def reset(self):
        self.ms.reset_game()
        if self.display_mode:
            self.ms.draw_board()
        return self.get_state()

    def step(self, action_x, action_y):
        self.ms.reveal_tile(action_x, action_y)
        state = self.get_state()
        status = self.ms.get_game_status()
        last_revealed = self.ms.get_last_revealed()

        if self.display_mode:
            self.ms.draw_board()
            time.sleep(0.1)

        if status == 'win':
            reward = 1
            done = True
        elif status == 'lose':
            reward = -1
            done = True
        else:
            if last_revealed == 'revealed':  # Revealed a mine
                reward = -1
                done = True
            elif last_revealed == 'guess':  # Guessed a mine incorrectly
                reward = -0.5
                done = False
            elif last_revealed == 'unrevealed':  # Revealed a safe tile
                reward = 0.1
                done = False
            elif last_revealed == 'first':  # First move
                reward = 0
                done = False
            else:
                raise ValueError("Invalid last_revealed value")

        return state, reward, done

    def get_state(self):
        # Get the game board and normalize it
        board = self.ms.get_game_board()
        # Normalize the board values to [0, 1]
        normalized_board = np.where(
            board == -1, 0.0,  # Map mines (-1) to 0.0
            np.where(
                board == 9, 0.5,  # Map unrevealed (9) to 0.5
                (board + 1) / 10  # Map numbers (0-8) to [0.1, 0.9]
            )
        )
        return torch.tensor(normalized_board, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    def get_board_size(self):
        return self.board_size

# QNet
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.mlp_layer = nn.Sequential(
            nn.Linear(128 * input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.mlp_layer(x)
        return x

# ExperienceBuffer
class ExperienceBuffer:
    def __init__(self, buffer_size=100000, device='cuda'):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
        self.device = device

    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.stack([e[1] for e in batch])
        rewards = torch.stack([e[2] for e in batch])
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.stack([e[4] for e in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# DQNAgent
class DQNAgent:
    def __init__(self, board_size, buffer_size=100000, min_train_size=2000, batch_size=64, gamma=0.99, lr=0.0001, device='cuda', eval_mode=False):
        if eval_mode:
            self.q_net = QNet(board_size**2, board_size**2).to(device)
            self.q_net.eval()
            self.device = device
            self.eval_mode = True
            return
        self.eval_mode = False
        self.min_train_size = min_train_size
        self.input_dim = board_size**2
        self.output_dim = board_size**2
        self.q_net = QNet(self.input_dim, self.output_dim).to(device)
        self.target_net = QNet(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ExperienceBuffer(buffer_size, device=device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

    def act(self, state, exploration_rate):
        if self.eval_mode:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                state = state.unsqueeze(0)
                q_values = self.q_net(state)
                return torch.argmax(q_values).item()

        if random.random() < exploration_rate:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                state = state.unsqueeze(0)  # Add batch dimension
                q_values = self.q_net(state)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.buffer) < self.min_train_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

# Training and Validation
def validate_agent(env, agent, num_episodes=10, p=0.01):
    total_rewards = 0
    win_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, p)  # Low exploration rate for validation
            action_x = action // env.get_board_size()
            action_y = action % env.get_board_size()
            next_state, reward, done = env.step(action_x, action_y)
            total_rewards += reward
            state = next_state
        if reward == 1:
            win_count += 1
    return total_rewards, win_count * 100 / num_episodes

def train_agent(env, agent, num_episodes=100000, test_num=1000, save_num=1000, exploration_rate=0.95, min_exploration_rate=0.05, exploration_decay=0.9999):
    test_results = []
    train_results = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        done = False
        while not done:
            action = agent.act(state, exploration_rate)
            action_x = action // env.get_board_size()
            action_y = action % env.get_board_size()
            next_state, reward, done = env.step(action_x, action_y)
            total_reward += reward
            agent.add_experience(state, action, reward, next_state, done)
            loss = agent.train()
            if loss:
                total_loss += loss
            state = next_state
        if (episode + 1) % save_num == 0:
            agent.update_target()
        if (episode + 1) % test_num == 0:
            test_reward, test_win_r = validate_agent(env, agent, num_episodes=100)
            test_results.append((test_reward, test_win_r))
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Total Loss: {total_loss:.2f}, Test Win: {test_win_r:.2f}")
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
        train_results.append((total_reward, total_loss))
    return train_results, test_results

# Main
size = 4
env = MinesweeperEnv(board_size=size, num_mines=5, display_mode=False)
agent = DQNAgent(board_size=size, min_train_size=10000, buffer_size=100000, batch_size=64, lr=0.000004, device='cuda')
agent.load('minesweeper_dqn(4).pth')
train_results, test_results = train_agent(env, agent, test_num=4000, save_num=200, num_episodes=80000)
#
# # Plot results
plt.figure(figsize=(15, 12))
_, axes = plt.subplots(3, 1)
axes[0].plot([r[0] for r in train_results], label='Train Reward')
axes[0].set_title('Rewards')
axes[1].plot([r[1] for r in train_results], label='Train Loss')
axes[1].set_title('Loss')
axes[2].plot([r[1] for r in test_results], label='Win Rate')
axes[2].set_title('Win Rate')
plt.savefig('train_results.png')
agent.save('minesweeper_dqn(4).pth')

test_reward, test_win_r = validate_agent(env, agent, num_episodes=10000, p=-1)
print(f"test_reward:{test_reward}, test_win_r:{test_win_r}")