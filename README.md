# RL_superMarioWorld
This is a Python script that uses a neural network to play the classic video game Super Mario World, using the NEAT algorithm for reinforcement learning.
It imports necessary modules such as gym, retro and neat.
The script defines a class called PlotReporter that contains a method for reporting the fitness of the neural network over time as it is trained.
Another function called eval_genomes evaluates the fitness of a given set of genomes by running the game and evaluating their performance.
The script initializes the game environment, sets up the neural network with NEAT, and then uses the eval_genomes function to train the network.
The training process involves running the game and collecting data on the network's performance, which is then used to adjust its weights and biases.

Right now, the script only plots the network's fitness over time using matplotlib. It is still in progress and I hope I'll keep going.

Heuristic fitness progress:


![image](https://user-images.githubusercontent.com/97172662/234115568-a26a2bc3-f351-49e0-ae46-d4a826d54ff1.png)
