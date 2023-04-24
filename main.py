import gym
import retro
import retrowrapper
import sys
import numpy as np
from PIL import Image
import cv2
import neat
import pickle
import pyautogui
import matplotlib.pyplot as plt


class PlotReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.fitnesses = []
        self.bests = []
        self.worsts = []
        self.avg = []

    def post_evaluate(self, config, population, species, best_genome):
        global all_fit
        generation = len(self.fitnesses) + 1
        fitnesses = [x.fitness for x in population.values()]
        self.fitnesses.append(fitnesses)

        best_fitness = max(fitnesses)
        self.bests.append(best_fitness)

        worst_fitness = min(fitnesses)
        self.worsts.append(worst_fitness)

        avg_fitness = sum(fitnesses) / len(fitnesses)
        self.avg.append(avg_fitness)

        plt.xticks(ticks=(range(0, generation)))

        plt.plot(self.bests, label='Best')
        plt.plot(self.worsts, label='Worst')
        plt.plot(self.avg, label='Average')
        plt.title('Fitnesses over generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

        # Reset all_fit for each new generation
        all_fit = []


def eval_genomes(genomes, config):
    global all_fit
    img_array = []
    for genome_id, genome in genomes:
        ob = env.reset()
        action = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape  # size of the image created
        # print(inx, iny, inc)

        inx = int(inx / 8)
        iny = int(iny / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        max_xpos = 0

        coins = 0
        powerups = 0
        yoshiCoins = 0

        done = False
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:
            # env.render()
            frame += 1

            scaled_img = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)

            computer_img = cv2.resize(scaled_img, (iny, inx))
            computer_img = cv2.cvtColor(computer_img, cv2.COLOR_BGR2GRAY)

            scaled_img_copy = scaled_img.copy()

            x_offset = y_offset = 10
            computer_img_resized = computer_img

            scaled_img_copy[y_offset:y_offset + computer_img_resized.shape[0],
            x_offset:x_offset + computer_img_resized.shape[1], :] = cv2.cvtColor(computer_img_resized,
                                                                                 cv2.COLOR_GRAY2RGB)

            cv2.imshow('main', scaled_img_copy)
            cv2.waitKey(1)

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            # Reshape the 2D grayscale image to a flattened 1D array
            img_array = ob.reshape(-1)

            nn_output = net.activate(img_array)

            # print(nn_output)

            ob, rew, done, info = env.step(nn_output)

            # Clear the contents of img_array
            img_array = np.array([], dtype=np.uint8)

            xpos = info['x']

            if xpos > max_xpos:
                fitness_current += 5  # itll change to rew
                max_xpos = xpos

            if powerups > info['powerups']:
                fitness_current += 200
                fitness_current = info['powerups']

            if yoshiCoins > info['yoshiCoins']:
                fitness_current += 200
                yoshiCoins = info['yoshiCoins']

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            fitness_current -= 0.4  # each frame lose

            # if the player doesn't move right in 250 frames, throw
            if done or counter == 250:
                done = True
                print(genome_id, round(fitness_current))

            all_fit.append(fitness_current)
            genome.fitness = fitness_current


# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#  A  B  X  Y  L  R  ↑  ↓  ←  →  Select Start

# best_fitnesses = []
# avg_fitnesses = []
# worst_fitnesses = []
# log_fitness = []
all_fit = []

# game_path = retro.data.get_romfile_path(game='SuperMarioWorld-Snes')
game_path = r'C:\Users\Alon\PycharmProjects\NeuMario\venv\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes'
env = retro.make(game=game_path, state='YoshiIsland1.state', record=False)
obs = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, 'config-feedfoward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()  # calls post_evaluate
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))  # every 10 generation - save
plot_reporter = PlotReporter()
p.add_reporter(plot_reporter)

winner = p.run(eval_genomes)

with open("winner.pkl", "wb") as output:
    pickle.dump(winner, output, 1)

with open("winner.pkl", "rb") as input_file:
    loaded_winner = pickle.load(input_file)

print("Loaded winner's fitness:", loaded_winner.fitness)
