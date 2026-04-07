import pytest
from src.parser_nn import Parser_nn


def _write_nn_file(tmp_path, payload):
    nn_path = tmp_path / "network.nn"
    nn_path.write_text(payload)
    return nn_path


def test_parser_exits_when_file_missing():
    with pytest.raises(SystemExit) as exc:
        Parser_nn("/does/not/exist.nn")
    assert exc.value.code == 84


def test_parser_exits_on_invalid_json(tmp_path):
    path = _write_nn_file(tmp_path, "{not-valid-json")
    with pytest.raises(SystemExit) as exc:
        Parser_nn(str(path))
    assert exc.value.code == 84


def test_parser_exits_when_learning_rate_missing(tmp_path):
    path = _write_nn_file(tmp_path, '{"layers": []}')
    with pytest.raises(SystemExit) as exc:
        Parser_nn(str(path))
    assert exc.value.code == 84


def test_parser_exits_when_layers_missing(tmp_path):
    path = _write_nn_file(tmp_path, '{"learning_rate": 0.01}')
    with pytest.raises(SystemExit) as exc:
        Parser_nn(str(path))
    assert exc.value.code == 84


def test_parser_loads_minimal_valid_network(tmp_path):
    valid_json = (
        '{'
        '"learning_rate": 0.01,'
        '"layers": ['
        '{'
        '"activate": "LeakyReLu",'
        '"neurons": ['
        '{"inputs": [{"weight": 0.1}, {"weight": -0.2}], "bias": 0.0}'
        ']'
        '}'
        ']'
        '}'
    )
    path = _write_nn_file(tmp_path, valid_json)
    parser = Parser_nn(str(path))

    assert parser.network is not None
    assert parser.network.learning_rate == 0.01
    assert len(parser.network.layers) == 1
    assert len(parser.network.layers[0].neurons) == 1

