# TSP Solver

This repository contains the implementation of algorithms and techniques for solving the Traveling Salesman Problem (TSP). The project includes classical and quantum-inspired approaches for optimization.

## Project Structure

- **main.py**: The main entry point for the project, orchestrating the execution of various components.

- **ClassicSA.py**: Implements a classical Simulated Annealing algorithm for solving TSP. Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function.

- **Lqa.py**: Implementations of a Local Quantum Annealing approach. Adjustments and modification of code by https://doi.org/10.1103/PhysRevApplied.18.03401

- **QML.py**: Implements Quantum Machine Learning techniques.
  
## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Required libraries: NumPy, Torch, Matplotlib

You can install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. Add your data to the appropriate directory. TSP problems used for this code can be found in https://www.kaggle.com/datasets/stephanhocke/15k-tsps-with-optimal-tours, https://www.math.uwaterloo.ca/tsp/world/countries.html, and http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/index.html
3. Run the main script to solve a QUBO instance:
    ```bash
    python main.py
    ```
4. Modify parameters in `SimCIM.py` to adjust optimization settings.

## Contributions

Feel free to submit issues or pull requests for bug fixes, new features, or improvements.

