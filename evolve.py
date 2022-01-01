import os
import multiprocessing
import random
import pickle
from typing import List
from itertools import count

import gym
import neat
from neat import nn
import numpy as np

RUNS_PER_NET = 5
MAX_GENERATION = 100

env = gym.make("CartPole-v1")


def plot(statistics: neat.StatisticsReporter, filename: str):
    import matplotlib.pyplot as plt

    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    most_fit_genomes = statistics.most_fit_genomes
    generation = range(len(most_fit_genomes))
    best_fitness = [genome.fitness for genome in most_fit_genomes]

    plt.plot(generation, avg_fitness, "b-", label="average")
    plt.plot(generation, avg_fitness + stdev_fitness, "g-", label="+1 stdev")
    plt.plot(generation, best_fitness, "r-", label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig(filename)
    plt.close()


def eval_genome(genome: neat.DefaultGenome, config: neat.Config):
    network = nn.FeedForwardNetwork.create(genome=genome, config=config)

    fitnesses = []
    for _ in range(RUNS_PER_NET):
        observation = env.reset()
        for t in count(1):
            output = network.activate(observation)
            action = np.argmax(output)
            observation, reward, done, info = env.step(action)
            if done:
                genome.fitness = t
                fitnesses.append(t)
                break

    return min(fitnesses)


def evolve():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    config = neat.Config(
        genome_type=neat.DefaultGenome,
        reproduction_type=neat.DefaultReproduction,
        species_set_type=neat.DefaultSpeciesSet,
        stagnation_type=neat.DefaultStagnation,
        filename=config_path,
    )

    while True:
        population = neat.Population(config=config)
        population.add_reporter(stats := neat.StatisticsReporter())
        population.add_reporter(neat.StdOutReporter(show_species_detail=True))

        parallel_evaluator = neat.ParallelEvaluator(
            num_workers=multiprocessing.cpu_count(), eval_function=eval_genome
        )
        best_genome = population.run(fitness_function=parallel_evaluator.evaluate)
        plot(statistics=stats, filename="feedforward_stats.svg")

        with open("best-feedforward.pickle", "wb") as f:
            pickle.dump(best_genome, f)

        if np.mean(stats.get_fitness_mean()) > 300:
            break

if __name__ == "__main__":
    evolve()
