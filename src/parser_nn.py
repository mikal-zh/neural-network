##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## parser_nn
##
import json
import sys
from src.neuron_network import Neuron_Network, Neuron_Layer, Neuron, Input, Activation
from src.calcul import sigmoid, sigmoid_derivative, ReLu, ReLu_derivative, leakyReLu, leakyReLu_derivative
from src.training import predict as training_predict

class Parser_nn:
    def __init__(self, loadfile: str = None):
        self.network = None
        if loadfile:
            self.load_from_file(loadfile)

    def load_from_file(self, loadfile: str):
        try:
            with open(loadfile, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{loadfile}' not found")
            sys.exit(84)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{loadfile}'")
            sys.exit(84)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(84)

        if not isinstance(data, dict):
            print("Error: Invalid neural network file format")
            sys.exit(84)

        learning_rate = data.get('learning_rate')
        if learning_rate is None:
            print("Error: Missing 'learning_rate' in file")
            sys.exit(84)

        layers_data = data.get('layers')
        if layers_data is None:
            print("Error: Missing 'layers' in file")
            sys.exit(84)

        layers = []
        for layer_data in layers_data:
            neurons_data = layer_data.get('neurons', [])
            activation_name = layer_data.get('activate', 'Sigmoid')

            if activation_name == 'Sigmoid':
                activation = Activation('Sigmoid', sigmoid, sigmoid_derivative)
            elif activation_name == 'ReLu':
                activation = Activation('ReLu', ReLu, ReLu_derivative)
            elif activation_name == 'LeakyReLu':
                activation = Activation('LeakyReLu', leakyReLu, leakyReLu_derivative)
            else:
                activation = Activation('ReLu', ReLu, ReLu_derivative)

            neurons = []
            for neuron_data in neurons_data:
                inputs_data = neuron_data.get('inputs', [])
                bias = neuron_data.get('bias', 0)

                inputs = []
                for inp_data in inputs_data:
                    weight = inp_data.get('weight', 0)
                    inputs.append(Input(weight, 0))
                neurons.append(Neuron(inputs, bias))
            layers.append(Neuron_Layer(neurons, activation))
        self.network = Neuron_Network(layers, learning_rate, [])

    def to_dict(self):
        if self.network is None:
            return {
                'layers': [],
                'learning_rate': 0.0
            }
        return self.network.to_dict()

    def save_to_file(self, filepath: str):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            print(f"Error: Failed to save file '{filepath}': {e}")
            return 84
        return 0
