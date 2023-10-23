import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plotter import plot

MAXIMUM_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, gamma = 0.9):
        self.n_games = 0
        self.epsilon = 0 #parameter to control randomness
        self.gamma = gamma #discount rate #0.9 for model0, 0.95 for model 1
        self.memory = deque(maxlen = MAXIMUM_MEMORY) #calls popleft is memory is full

        self.model = Linear_QNet(15, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        #find all immediate points around the snake head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        #booleans - True if the snake is facing in that direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        food_left = game.food.x < game.head.x  # food left
        food_right = game.food.x > game.head.x # food right
        food_up = game.food.y < game.head.y  # food up
        food_down = game.food.y > game.head.y  # food down

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            (food_left) and not (food_up or food_down), #left
            (food_right) and not (food_up or food_down),#right
            (food_up) and not (food_left or food_right),#up
            (food_down) and not (food_left or food_right),#down
            food_left and food_up,
            food_right and food_up,
            food_left and food_down,
            food_right and food_down
            ]
        #return array of 1s and 0s depending on whether each state is true or false
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft in MAXIMUM_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, random_moves = True):
        if random_moves == True:
        #random moves: tradeoff exploration/exploitation
            self.epsilon = 100 - self.n_games
        else:
            self.epsilon = 0
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(gamma = 0.95) #model0 gamma = 0.9 ## model1 gammma = 0.95
    game = SnakeGameAI(w = 20*8, h = 20*8, speed = 600, reward_food= 10, reward_death= -10, reward_step= -0.1) #model0 default w and h, 30, -10, 0 ## model1 20*8,20*8, 10, -10, -0.1
    model_number = 1
    while True:
        #get old state
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)
        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (replay memory), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                file_name = 'model' + str(model_number) + '.pt'
                agent.model.save(file_name)
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
if __name__ == '__main__':
    train()