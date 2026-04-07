##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## Network initialization and generation
##

import random
from src.neuron_network import Neuron_Network, Neuron_Layer, Neuron, Input, Activation
from src.calcul import leakyReLu, leakyReLu_derivative

MIN_NEURONS = 256
MAX_NEURONS = 416
MIN_LAYER = 3
MAX_LAYER = 4
MIN_LR = 0.01
MAX_LR = 0.1
MIN_WEIGHT = -0.05
MAX_WEIGHT = 0.05
MIN_BIAS = -0.05
MAX_BIAS = 0.05

CHESS_INPUTS = 832
HANDLED_CASES = 5

ACTIVATION: list[Activation] = [
    Activation("LeakyReLu", leakyReLu, leakyReLu_derivative),
]

LEAKYRELU = 0

def init_network(nb_layers: int = None,
    layers_nb_neurons: list[int] = None,
    layers_activation: list[Activation] = None,
    learning_rate: float = None):

    if nb_layers is None:
        nb_layers = random.randint(MIN_LAYER, MAX_LAYER)
    layers_sizes: list[int] = []
    if layers_nb_neurons is None:
        min_neurons: int = MIN_NEURONS
        max_neurons: int = MAX_NEURONS
        for l in range(nb_layers - 1):
            layers_sizes.append(random.randint(int(min_neurons), int(max_neurons)))
            max_neurons = layers_sizes[l]
            min_neurons /= 2
    else:
        layers_sizes = layers_nb_neurons.copy()
    layers_sizes.append(HANDLED_CASES)
    layers_act: list[int] = [random.choice([LEAKYRELU]) for _ in range(nb_layers)]
    layers_act.append(None)
    prev_size: int = CHESS_INPUTS
    layers: list[Neuron_Layer] = []

    for i in range(nb_layers):
        neurons = [
            Neuron(
                inputs=[Input(weight=random.uniform(MIN_WEIGHT, MAX_WEIGHT), value=0) for _ in range(prev_size)],
                bias=random.uniform(MIN_BIAS, MAX_BIAS)
            )
            for _ in range(layers_sizes[i])
        ]
        activate: Activation = None
        if (layers_activation is None):
            activate = None if layers_act[i] == None else ACTIVATION[layers_act[i]]
        else:
            activate = layers_activation[i]
        layer = Neuron_Layer(neurons, activate)
        layers.append(layer)
        prev_size = layers_sizes[i]
    network: Neuron_Network = Neuron_Network(layers, lr=learning_rate\
        if learning_rate is not None else random.uniform(MIN_LR, MAX_LR),\
        awaited_output=[0 for _ in range(HANDLED_CASES)])
    return network
