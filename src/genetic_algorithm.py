##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## Genetic algorithm for neural network evolution
##

import random
from src.neuron_network import Neuron_Network, Activation
from src.network_init import init_network

SIZE_GENERATION = 8
NETWORKS_SELECTION = 2
NB_PARENTS_TO_SELECT = 3

def crossover(networks: list[Neuron_Network]):

    if (len(networks) != NB_PARENTS_TO_SELECT):
        raise RuntimeError("Error: Invalid number of Parent networks")

    next_gen: list[Neuron_Network] = []

    for index in range(len(networks)):
        a = index
        b = index + 1 if index < len(networks) - 1 else 0
        chosen_strct = random.choice([a, b])
        nb_layers: int = len(networks[chosen_strct].layers)
        layers_neurons: list[int] = [len(networks[chosen_strct].layers[i].neurons)\
            for i in range(len((networks[chosen_strct].layers)) - 1)]
        layers_activation: list[Activation] = [layer.act for layer in networks[chosen_strct].layers]
        learning_rate: float = (networks[a].learning_rate + networks[b].learning_rate) / 2
        child: Neuron_Network = init_network(nb_layers, layers_neurons, layers_activation, learning_rate)
        next_gen.append(child)
    return next_gen

def select_next_gen(networks: list[Neuron_Network]):
    if (len(networks) < NB_PARENTS_TO_SELECT):
        raise RuntimeError("Error: Invalid number of Neural networks")
    selected_parents: list[Neuron_Network] = []

    for _ in range(NB_PARENTS_TO_SELECT):
        candidates: list[Neuron_Network] = []
        for _ in range(NETWORKS_SELECTION):
            candidates.append(random.choice(networks))
        winner: Neuron_Network = max(candidates, key=lambda net: net.fitness)
        selected_parents.append(winner)

    return crossover(selected_parents)

def select_best_parents(networks: list[Neuron_Network]):
    if (len(networks) <= SIZE_GENERATION):
        return networks
    networks.sort(key=lambda x: x.fitness, reverse=True)
    return networks[:SIZE_GENERATION]
