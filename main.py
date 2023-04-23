import gym
import retro
import retrowrapper
import sys
import numpy as np
from PIL import Image
import cv2
import neat
import pickle


def eval_genomes(genomes, config):
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

        done = False

        while not done:
            env.render()

            frame += 1

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
                fitness_current += 1  # itll change to rew
                max_xpos = xpos

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # if the player doesn't move right in 250 frames, throw
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#  A  B  X  Y  L  R  ↑  ↓  ←  →  Select Start

# game_path = retro.data.get_romfile_path(game='SuperMarioWorld-Snes')
game_path = r'C:\Users\Alon\PycharmProjects\NeuMario\venv\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes'
env = retro.make(game=game_path, state='YoshiIsland1.state', record=False)
obs = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, 'config-feedfoward')

p = neat.Population(config)

winner = p.run(eval_genomes)
