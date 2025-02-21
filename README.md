# GNSSjamLoc: Federated Learning for Jammer Localization

## Overview

GNSSjamLoc is a localization tool designed for detecting intentional interference sources (jammers) that disrupt GNSS signal reception. This project employs **Federated Learning (FL)** combined with a **Physics-Augmented Model** to enable **privacy-preserving, collaborative jammer localization**.

### Key Features:
- **Federated Learning**: Distributes model training across multiple nodes without sharing raw data.
- **Physics-Augmented AI Model**: Combines **Neural Networks (NN)** and **Pathloss Models (PL)**.
- **Monte Carlo Simulations**: Multiple runs ensure robust statistical evaluation.
- **Scalable Experimentation**: Supports different configurations to test various conditions.

---

## Folder Structure

```plaintext
jammerLocalization/
│── datasets/                 # Contains datasets used in experiments
│── results/                  # Stores experiment results and logs
│── src/                      # Source code for model training and evaluation
│   │── data_loader.py        # Loads and processes data for training
│   │── fedavg.py             # Implements the Federated Averaging (FedAvg) algorithm
│   │── model.py              # Contains model architectures (NN, PL, APBM)
│   │── plots.py              # Generates visualization plots for results
│   ├── data_generation/      # Scripts for data simulation
│   │   ├── jammedAgentsSimulation.m
│   │   ├── src/
│── main.py                   # Main experiment execution script
│── README.md                 # Documentation file
│── .gitignore                # Specifies files to exclude from version control
```
---

# Running Experiments

## 1. Configuration Parameters

Experiments are controlled by a **configuration dictionary** (`config`) in `main.py`. Here’s an overview of the key parameters:

```python
config = {
    'test_ratio': 0.2,  # 20% of data is used for testing
    'data_preprocessing': 2,  # 0 = none, 1 = replace -inf, 2 = remove outliers
    'noise': True,  # Add noise to measurements
    'meas_noise_var': 1,  # Measurement noise variance
    'num_nodes': 10,  # Number of federated learning clients
    'num_obs': 1000,  # Dataset size
    'batch_size': 8,  # Training batch size
    'num_rounds_nn': 40,  # NN training rounds
    'num_rounds_pl': 40,  # PL training rounds
    'num_rounds_apbm': 40,  # APBM training rounds
}
```

## 2. Choosing an Experiment Mode

There are two modes for running experiments:

1. **Single Experiment Mode**: Runs one experiment with a fixed configuration.
2. **Multiple Experiment Mode**: Loops through multiple parameter settings for systematic testing.

Modify this setting in `main.py`:

```python
execution_type = 'one_experiment'  # Run a single experiment
# execution_type = 'all_experiments'  # Run multiple experiments
```

## 3. Selecting Experiment Parameters

### Single Experiment Mode

Modify the following parameters:

```python
scenarios = ['urban_raytrace']  # Choose dataset type (urban, suburban, pathloss)
experiments = ['show_figures']  # Select experiment type (noise, number of nodes, etc.)
numNodes = 5  # Number of nodes
meas_noise_var = 1  # Noise variance level
show_figures = True  # Enable visualizations
```

### Multiple Experiment Mode

Modify the following to run experiments across multiple configurations:

```python
experiments = ['meas_noise_var']  # Run tests for different noise levels
numNodes = np.array([1, 5, 10, 25, 50])  # Different client configurations
posEstVar = np.array([0, 36])  # Different position estimation errors
num_obs = np.array([250, 500, 750, 1000])  # Varying dataset sizes
meas_noise_var = np.array([10, 10/np.sqrt(10), 1, 0.1])  # Different noise levels
```

## 4. Federated Learning Workflow

### Step 1: Data Loading
- GNSS signal measurements are loaded from `datasets/`.
- Data is preprocessed and split into training and test sets.

### Step 2: Model Training

The system trains three models:
1. **Neural Network (NN)**: Learns an initial approximation of jammer location.
2. **Pathloss Model (PL)**: Uses physical equations to refine localization.
3. **Augmented Physics-Based Model (APBM)**: Combines NN and PL for accuracy.

### Step 3: Federated Learning (FL)
- Each node trains its local model independently.
- A central server aggregates updates using **Federated Averaging (FedAvg)**.

### Step 4: Evaluation & Metrics
- Localization errors are computed for **NN, PL, and APBM**.
- Results are stored in `results/` and optionally visualized.

### Step 5: Monte Carlo Simulations
- The experiment repeats **N_mc = 10** times to ensure reliable results.

---

## 5. Results & Output

### Where Are the Results Stored?
- All results are saved in the `results/` directory.
- Each experiment creates a unique folder with logs and figures.

Example structure:

```plaintext
results/
│── Execution_1_suburban/
│── Execution_2_urban/
│── Execution_3_pathloss/
│── Execution_4_plots_urban/
```

### Log File Output

Each Monte Carlo run logs key metrics:

```plaintext
Monte Carlo Run 1/10 with Seed: 42
Final Test Loss (NN): 0.2345
Final Test Loss (PL): 0.1234
Final Test Loss (APBM): 0.0987
Jammer Localization Error (PL): 2.15m
Jammer Localization Error (APBM): 1.02m
```

## 6. How to Modify the Experiment?

1. **Change the dataset**: Modify `scenarios` in `main.py`.
2. **Adjust parameters**: Update values in the `config` dictionary.
3. **Enable/disable visualizations**: Set `show_figures = True/False`.
4. **Change the number of Monte Carlo runs**: Modify `N_mc = X`.

---

## 7. Example: Running an Experiment

### Single Experiment

Run a single test case:

```bash
python main.py
```

### Multiple Experiments

Run all parameter variations:

Modify in `main.py`:

```python
execution_type = 'all_experiments'
```

## 8. Dependencies & Installation

This project requires **Python 3.8+** and the following packages:

```bash
pip install torch numpy matplotlib scipy scikit-learn
```

## 9. Citing This Work

If you use this software, please cite:

Mariona Jaramillo-Civill, Peng Wu, Andrea Nardin, Tales Imbiriba, Pau Closas. Jammer Source Localization with Federated
Learning. IEEE/ION Position, Location and Navigation Symposium (PLANS), Salt Lake City, UT, USA, April 2025.

---

## 10. License

This project is licensed under the **GNU General Public License v3**. See `LICENSE` for details.