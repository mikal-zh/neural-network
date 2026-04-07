from src.calcul import ReLu, leakyReLu, sigmoid, softMax
from src.network_init import CHESS_INPUTS, HANDLED_CASES, init_network


def test_init_network_has_output_layer_of_expected_size():
	network = init_network(nb_layers=3)
	assert len(network.layers[-1].neurons) == HANDLED_CASES


def test_init_network_first_layer_matches_input_size():
	network = init_network(nb_layers=3)
	first_neuron = network.layers[0].neurons[0]
	assert len(first_neuron.inputs) == CHESS_INPUTS


def test_activation_functions_basic_behavior():
	assert sigmoid(0) == 0.5
	assert ReLu(-3) == 0
	assert ReLu(3) == 3
	assert leakyReLu(-2) == -0.02
	assert leakyReLu(2) == 2


def test_softmax_is_probability_distribution():
	outputs = softMax([1.0, 2.0, 3.0])
	assert len(outputs) == 3
	assert abs(sum(outputs) - 1.0) < 1e-9
	assert outputs[2] > outputs[1] > outputs[0]
