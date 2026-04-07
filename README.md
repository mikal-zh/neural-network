# Chess Neural Network (Custom Python Framework)

Lightweight end-to-end neural network project for chess position classification.
The project includes model generation, training, inference, and benchmark tracking.

## Why this project is interesting

- Built from scratch in Python (no high-level ML framework for core NN logic).
- Full training loop with forward pass, backpropagation, batching, and model serialization.
- Domain-specific input pipeline for FEN chess positions.
- CLI tools for generation, training, and prediction.

## Problem tackled

Classify a chess position into one of 5 labels:

- `Nothing`
- `Check White`
- `Check Black`
- `Checkmate White`
- `Checkmate Black`

## Tech stack

- Python 3.11+
- NumPy
- Pytest

## Project structure

```text
.
|- my_torch_generator        # Generate/train candidate models from config
|- my_torch_analyzer         # Train existing model or run prediction
|- src/
|  |- training.py            # Forward pass, backprop, training loop
|  |- network_init.py        # Model initialization
|  |- neuron_network.py      # Core data structures
|  |- fen_utils.py           # FEN parsing and label encoding
|  |- parser_nn.py           # .nn JSON load/save
|  |- tests/                 # Automated tests
|- BENCHMARKS.md             # Historical performance results
```

## Quick start

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run tests.

```bash
pytest -q
```

3. Generate models.

```bash
./my_torch_generator <config.conf> <count>
```

4. Train an existing model.

```bash
./my_torch_analyzer --train <model.nn> <dataset.conf> --save <output_prefix>
```

5. Predict labels.

```bash
./my_torch_analyzer --predict <model.nn> <dataset.conf>
```

## Model format (`.nn`)

Models are saved as JSON with:

- `learning_rate`
- `layers`
    - `neurons`
        - `inputs` (`weight` per connection)
        - `bias`
    - `activate`

This keeps models easy to inspect and debug.

## Results snapshot

- Best reported accuracy in this repo history: around `68.38%` (see `BENCHMARKS.md`).
- Includes multiple experiments across architectures and learning rates.

## Known limitations

- Heavy assets are stored directly in the repo (`train`, `.nn` files).
- Limited active unit test coverage (improved for core utility paths).
- No CI pipeline yet.

## Suggested next steps

- Add GitHub Actions for tests and linting.
- Move datasets/models to release artifacts or external storage.
- Add richer metrics (per-class confusion matrix, precision/recall export).

## Author note

This repository is intended as an educational deep-dive into neural network internals,
with a practical chess-oriented use case.

```bash
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2 ...]
```

#### Parameters

- `config_file_i`: Not actually used for generation (placeholder for compatibility)
- `nb_i`: Number of networks to generate

#### Example

```bash
./my_torch_generator basic_network.conf 3
```

Creates:
- `basic_network_1.nn`
- `basic_network_2.nn`
- `basic_network_3.nn`

#### Generated Network Architecture

Networks are generated based on configuration:
- **Input layer**: 832 neurons (8×8 board × 13 piece types)
- **Hidden layers**: Defined in config file or randomly generated
- **Output layer**: 5 neurons (automatically added, one per class)
- **Activation**: LeakyReLu for all layers
- **Learning rate**: Specified in config or random between 0.01 and 0.1
- **Weights**: Random between -0.05 and 0.05
- **Biases**: Random between -0.05 and 0.05

**Note**: The output layer with 5 neurons is always added automatically - never include it in your configuration file!

### 2. my_torch_analyzer

Train or make predictions with neural networks.

#### Usage

```bash
# Training
./my_torch_analyzer --train <network.nn> <dataset.conf> --save <output_name>

# Prediction
./my_torch_analyzer --predict <network.nn> <dataset.conf>
```

#### Examples

```bash
# Train a network on a dataset
./my_torch_analyzer --train my_network.nn dataset.conf --save trained_model

# Make predictions
./my_torch_analyzer --predict my_network.nn test_dataset.conf
```

---

## Usage Examples

### Complete Workflow

```bash
# 1. Generate and train with optimized config (RECOMMENDED)
./my_torch_generator ./config_optimized.conf 1

# 2. Quick test with small dataset
./my_torch_generator ./config_test.conf 1

# 3. Genetic algorithm approach
./my_torch_generator ./config_random.conf 1

# 4. Make predictions
./my_torch_analyzer --predict my_torch_network.nn test_positions.conf
```

### Recommendations for 60%+ Accuracy

#### 1. Learning Rate
Try values between 0.001 and 0.01:
- **Too high**: Causes divergence
- **Too low**: Slow learning
- **Recommended**: 0.005 (as in `config_optimized.conf`)

#### 2. Architecture
- Start with 256-384 neurons in first layer
- Reduce progressively (divide by 2)
- 4-5 hidden layers is optimal

#### 3. Dataset
- Use `dataset_complet_shuffled.conf` for best results
- Ensure data is properly shuffled
- Check distribution is balanced

#### 4. Training Epochs
- `NB_EPOCH = 15` should be sufficient
- Auto-stops if < 25% at epoch 5
- Auto-stops if < 40% at epoch 12

### Troubleshooting Low Performance

- [ ] Verify dataset is balanced (use `dataset/check_distribution.py`)
- [ ] Try different learning rates (0.001, 0.005, 0.01)
- [ ] Reduce BATCH size in `src/training.py` (currently 1024)
- [ ] Check weight initialization (MIN_WEIGHT, MAX_WEIGHT in `network_init.py`)

---

## Architecture

### Input Encoding

Chess positions are encoded as **832-dimensional vectors**:
- 8 rows × 8 columns × 13 piece types = 832 features
- One-hot encoding: exactly one `1` per square, rest are `0`

**13 piece types:**
1. Empty square
2. White Pawn, Knight, Bishop, Rook, Queen, King
3. Black Pawn, Knight, Bishop, Rook, Queen, King

### Network Architecture

```
Input Layer (832)
    ↓
Hidden Layer 1 (256-416 neurons, LeakyReLu)
    ↓
Hidden Layer 2 (128-416 neurons, LeakyReLu)
    ↓
Hidden Layer 3 (64-416 neurons, LeakyReLu) [optional]
    ↓
Output Layer (5 neurons, LeakyReLu)
```

### Training Process

1. **Forward propagation**: Input flows through layers
2. **Loss calculation**: Compare output with expected class
3. **Backpropagation**: Calculate gradients
4. **Weight update**: Adjust weights using learning rate

---

## Benchmarks

### Training Performance

| Network | Epochs | Training Accuracy | File Size | Training Time |
|---------|--------|-------------------|-----------|---------------|
| `my_torch_network_0_2_68.38.nn` | 2 | 68.38% | 45 MB | ~30 min |
| `my_torch_network_0_3_65.30.nn` | 3 | 65.30% | 52 MB | ~45 min |
| `my_torch_network_67.32.nn` | 5 | 67.32% | 38 MB | ~1 hour |

### Dataset Statistics

- **Total positions**: 500,000+
- **Classes**: 5 (balanced)
- **Positions per class**: ~100,000
- **Training set**: 80%
- **Validation set**: 20%

### Model Performance Comparison

| Model | Check Detection | Checkmate Detection | Overall Accuracy |
|-------|-----------------|---------------------|------------------|
| Baseline Random | 20% | 20% | 20% |
| My Torch v1 | 65% | 70% | 67.32% |
| My Torch v2 (optimized) | 68% | 72% | 68.38% |

---

## Design Choices Justification

### Why JSON for .nn files?

 **Pros:**
- Human-readable for debugging
- Easy to inspect weights and architecture
- Universal format (any language can parse)
- Simple to extend with new fields
- No binary compatibility issues

 **Cons:**
- Larger file size than binary formats
- Slower to parse than binary

**Verdict**: Readability and flexibility outweigh size concerns for this educational project.

### Why LeakyReLu activation?

- **Prevents dying neurons**: Unlike ReLu, allows small negative gradients
- **Faster than Sigmoid**: No exponential calculations
- **Better gradient flow**: Helps with deep networks
- **Industry standard**: Widely used in modern neural networks

### Why separate .nn and .conf files?

- **Modularity**: Same network can be used with different datasets
- **Flexibility**: Easy to swap datasets or networks
- **Clarity**: Clear separation between model and data
- **Reusability**: Train one network on multiple datasets

---

## Getting Started

1. **Generate a network**:
   ```bash
   ./my_torch_generator my_network.conf 1
   ```

2. **Train it**:
   ```bash
   ./my_torch_analyzer --train my_network_1.nn dataset.conf --save trained
   ```

3. **Test it**:
   ```bash
   ./my_torch_analyzer --predict trained.nn test_data.conf
   ```

---

## License

This project is part of EPITECH's Neural Network project (G-CNA-500).

---

## Additional Resources

- **[BENCHMARKS.md](BENCHMARKS.md)** - Detailed performance analysis

---

**Made by Idriss DUPOISOT, Mikal ZHENG and Alexandre ODRISOLO**
