##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## Training module for neural networks
##

from logging import ERROR
import random
import numpy as np
import math
from typing import Callable, Optional
from functools import lru_cache
from src.fen_utils import get_awaited_output, load_inputs
from src.network_init import ACTIVATION, LEAKYRELU
from src.neuron_network import Activation, Neuron_Layer, Neuron_Network, Neuron
from src.calcul import leakyReLu_derivative, softMax, leakyReLu

BATCH = 512
CHESS_INPUTS = 832
INPUT_LAYER = 0

@lru_cache(maxsize=500000)
def parse_line(line: str):
    return (np.array(load_inputs(line), dtype=np.float64), get_awaited_output(line))

class Training_Session:
    def __init__(self, total_loss: float, network: Neuron_Network,
                 accumulated_weight_deltas, accumulated_bias_deltas):
        self.total_loss: float = total_loss
        self.network: Neuron_Network = network
        self.accumulated_weight_deltas: list[list[list[float]]] = accumulated_weight_deltas
        self.accumulated_bias_deltas: list[list[float]] = accumulated_bias_deltas

def ensure_np_cache(network: Neuron_Network):
    if not hasattr(network, "np_cache"):
        weights = []
        biases = []
        for layer in network.layers:
            w = np.array([[inp.weight for inp in n.inputs] for n in layer.neurons], dtype=np.float64)
            b = np.array([n.bias for n in layer.neurons], dtype=np.float64)
            weights.append(w)
            biases.append(b)
        
        network.np_cache = {
            "weights_cache": weights,
            "biases_cache": biases,
            "current": [None] * (len(network.layers) + 1),
            "raw": [None] * (len(network.layers))
        }
    return network.np_cache

def init_weight_deltas(network: Neuron_Network):
    cache = ensure_np_cache(network)
    return [np.zeros_like(weight) for weight in cache["weights_cache"]]

def init_bias_deltas(network: Neuron_Network):
    cache = ensure_np_cache(network)
    return [np.zeros_like(bias) for bias in cache["biases_cache"]]
def activation(first_layer: bool, layer: Neuron_Layer, raw_res: np.ndarray):
    if (first_layer):
        exps = np.exp(raw_res - np.max(raw_res))
        return exps / np.sum(exps)
    elif (layer.act.name == ACTIVATION[LEAKYRELU].name):
        return np.where(raw_res > 0, raw_res, raw_res * 0.01)
    raise RuntimeError("Error: Invalid activation function")

def frontpropagation(network: Neuron_Network, input_values: np.ndarray):
    cache = ensure_np_cache(network)
    cache["current"][0] = input_values

    for l, layer in enumerate(network.layers):
        weights = cache["weights_cache"][l]
        biases = cache["biases_cache"][l]
        raw_res = np.dot(weights, cache["current"][l]) + biases
        cache["raw"][l] = raw_res
        activated = activation(l == len(network.layers) - 1, layer, raw_res)
        cache["current"][l + 1] = activated
    return network

def cross_entropy(target, output):
    eps = 1e-9
    return -sum(t * math.log(o + eps) for t, o in zip(target, output))

def backpropagation(network: Neuron_Network,
    accumulated_weight_deltas, accumulated_bias_deltas):

    cache = ensure_np_cache(network)
    y_target = np.array(network.awaited_output)
    delta = cache["current"][-1] - y_target

    for i in range(len(network.layers) - 1, -1, -1):
        gradient_w = np.outer(delta, cache["current"][i])
        accumulated_weight_deltas[i] += gradient_w
        accumulated_bias_deltas[i] += delta
        if i > 0:
            error_prop = np.dot(cache["weights_cache"][i].T, delta)

            if (network.layers[i-1].act.name == ACTIVATION[LEAKYRELU].name):
                derivative = np.where(cache["raw"][i-1] > 0, 1.0, 0.01)
            else:
                d_func = network.layers[i-1].act.derivate
                derivative = np.array([d_func(z) for z in cache["raw"][i-1]])
                
            delta = error_prop * derivative
    return accumulated_weight_deltas, accumulated_bias_deltas

def apply_accumulated_deltas(network: Neuron_Network,
    accumulated_weight_deltas, accumulated_bias_deltas,
    nb_tests: int):
    cache = ensure_np_cache(network)
    scale = network.learning_rate / nb_tests
    for i in range(len(cache["weights_cache"])):
        cache["biases_cache"][i] -= accumulated_bias_deltas[i] * scale
        cache["weights_cache"][i] -= accumulated_weight_deltas[i] * scale
    return network

def synchronize_network_objects(network: Neuron_Network):
    cache = ensure_np_cache(network)
    weights, bias_list = cache["weights_cache"], cache["biases_cache"]
    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.bias = float(bias_list[layer_index][neuron_index])
            row = weights[layer_index][neuron_index]
            for input_index, inp in enumerate(neuron.inputs):
                inp.weight = float(row[input_index])
    return network

def train_neural_network(content: list[str], training: Training_Session):
    nb_train: int = len(content)
    serie: int = max(1, int(nb_train * 0.1))
    ensure_np_cache(training.network)

    for line in range(nb_train):
        input_values, awaited_output = parse_line(content[line])
        if (len(input_values) != CHESS_INPUTS):
            print("Error: Wrong number of inputs for neural network")
            return ERROR
        training.network.awaited_output = awaited_output
        training.network = frontpropagation(training.network, input_values)

        outputs = training.network.np_cache["current"][-1]
        training.total_loss += cross_entropy(awaited_output, outputs)
        training.accumulated_weight_deltas, training.accumulated_bias_deltas =\
            backpropagation(training.network, training.accumulated_weight_deltas, training.accumulated_bias_deltas)

        if ((line + 1) % BATCH == 0):
            training.network = apply_accumulated_deltas(training.network,\
                training.accumulated_weight_deltas, training.accumulated_bias_deltas, BATCH)
            training.accumulated_weight_deltas = init_weight_deltas(training.network)
            training.accumulated_bias_deltas = init_bias_deltas(training.network)
        if ((line + 1) % serie == 0):
            avg_loss: float = training.total_loss / (line + 1)
            print(f" - Testing... {int(((line + 1) * 100) / nb_train)}% | Loss: {avg_loss:.4f}")

    remaining: int = nb_train % BATCH
    if (remaining != 0):
        training.network = apply_accumulated_deltas(training.network,\
            training.accumulated_weight_deltas, training.accumulated_bias_deltas, remaining)

    training.network = synchronize_network_objects(training.network)
    return training

def experiment_accuracy(content: list[str], network: Neuron_Network):
    validation_size = len(content)
    ensure_np_cache(network)
    network.fitness = 0
    for line in range(validation_size):
        input_values, awaited = parse_line(content[line])
        network = frontpropagation(network, input_values)
        outputs: list[float] = network.np_cache["current"][-1]
        expected = np.argmax(awaited)
        predicted = np.argmax(outputs)
        if (predicted == expected):
            network.fitness += 1
    network.fitness = (network.fitness * 100) / validation_size
    print(f"________________________\n")
    print(f"Accuracy: {network.fitness:.2f}% ({int(network.fitness * validation_size / 100)}/{validation_size} correct)")
    print(f"________________________")
    return network.fitness

def predict(network: Neuron_Network, input_values: list[float]):
    network = frontpropagation(network, input_values)
    return network.np_cache["current"][-1].tolist()

def train_on_data(network: Neuron_Network, lines: list[str], nb_epochs: int = 10):
    ensure_np_cache(network)
    random.shuffle(lines)
    split_index = int(len(lines) * 0.9)
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch + 1}/{nb_epochs}")
        acc_wgt_dt = init_weight_deltas(network)
        acc_bias_dt = init_bias_deltas(network)
        training = Training_Session(0, network, acc_wgt_dt, acc_bias_dt)
        training = train_neural_network(train_lines, training)
        print(f"_ _ _\n\nTraining finished")
        network.fitness = experiment_accuracy(val_lines, network)
        print(f"\nEpoch finished\n_ _ _\n")
        training.total_loss = 0
    
    return network
