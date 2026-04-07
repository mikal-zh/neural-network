##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## neuron_network
##
from typing import Callable

class Activation:
    def __init__(self, name, activate, derivate):
        self.name : str = name
        self.activate : Callable = activate
        self.derivate : Callable = derivate

class Input:
    def __init__(self, weight, value):
        self.weight : float = weight
        self.value : float = value

    def to_dict(self):
        return {
            'weight': self.weight,
        }

class Neuron:
    def __init__(self, inputs, bias, output = 0, pre_output = 0):
        self.inputs : list[Input] = inputs
        self.bias : float = bias
        self.output : float = output
        self.pre_output : float = pre_output
    
    
    def to_dict(self):
        return {
            'inputs': [input.to_dict() for input in self.inputs],
            'bias': self.bias,
        }

class Neuron_Layer:
    def __init__(self, neurons, act):
        self.neurons: list[Neuron] = neurons
        self.act: Activation = act

    def to_dict(self):
        return {
            'neurons': [neuron.to_dict() for neuron in self.neurons],
            'activate': self.act.name
        }

class Neuron_Network:
    def __init__(self, layers, lr, awaited_output):
        self.layers: list[Neuron_Layer] = layers
        self.learning_rate: float = lr
        self.awaited_output : list[float] = awaited_output
        self.fitness : float = 0

    def to_dict(self):
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'learning_rate': self.learning_rate,
        }