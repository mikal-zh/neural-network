import pytest

from src.fen_utils import CHESS_INPUTS, get_awaited_output, load_inputs


def test_load_inputs_empty_board_size():
	line = "8/8/8/8/8/8/8/8 w - - 0 1 Nothing"
	inputs = load_inputs(line)
	assert len(inputs) == CHESS_INPUTS


def test_load_inputs_with_piece_changes_vector():
	empty_line = "8/8/8/8/8/8/8/8 w - - 0 1 Nothing"
	piece_line = "8/8/8/8/8/8/8/7K w - - 0 1 Nothing"

	empty_inputs = load_inputs(empty_line)
	piece_inputs = load_inputs(piece_line)

	assert piece_inputs != empty_inputs


def test_get_awaited_output_known_label():
	output = get_awaited_output("8/8/8/8/8/8/8/8 w - - 0 1 Checkmate White")
	assert output == [0, 1, 0, 0, 0]


def test_get_awaited_output_raises_on_unknown_label():
	with pytest.raises(KeyError):
		get_awaited_output("8/8/8/8/8/8/8/8 w - - 0 1 Unknown")
