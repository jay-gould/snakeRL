import torch
from agent import Agent
from snake_game import SnakeGameAI
from plotter import plot 

def test(model_name):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    agent.model.load_state_dict(torch.load('./model/'+model_name))
    game = SnakeGameAI(speed = 200)
    while agent.n_games < 100:
        #get old state
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old, random_moves=False)
        #perform move and get new state
        _, done, score = game.play_step(final_move)
        _ = agent.get_state(game)

        if done:
            # train long memory (replay memory), plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, train = False)
    print(f'Final Mean score: {mean_score}')
    variance_list = [pow(score - mean_score,2) for score in plot_scores]
    variance = sum(variance_list)/agent.n_games
    print(f'Final Variance: {variance}')
    
if __name__ == '__main__':
    test('model1.pt')