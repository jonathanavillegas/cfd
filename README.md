# CFD Heat Transfer Solver

A Python package for solving heat conduction problems using finite difference methods.

## Features

- **1D Steady-State Solver**: Solves Poisson equation for 1D heat conduction
- **2D Steady-State Solver**: Solves 2D Poisson equation with various boundary conditions
- **2D Transient Solver**: Time-dependent 2D heat conduction
- **Heat Shield Solvers**: 2D and 3D cylindrical coordinate solvers for heat shield simulations
- **Visualization**: 2D and 3D plotting capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jonathanavillegas/cfd.git
cd cfd
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### 1D Steady-State
```python
from cfd.solvers.poisson_1d import solve_1d_steady

x, T = solve_1d_steady(num_nodes=100, length=10, k=5, BCR_type="neumann")
```

### 2D Steady-State
```python
from cfd.solvers.poisson_2d import solve_2d_steady

x, y, T = solve_2d_steady(numXNodes=50, numYNodes=50, xL=10, yL=10, k=0.001)
```

### 2D Transient
```python
from cfd.solvers.transient_2d import solve_2d_transient

x, y, transient = solve_2d_transient(
    numXNodes=50, numYNodes=50, xL=10, yL=10,
    time=100, dt=0.05, k=700, rho=1000, cp=10, T0=0
)
```

### Heat Shield 3D
```python
from cfd.solvers.heat_shield import solve_heat_shield_3d

r, theta, z, transient = solve_heat_shield_3d(
    Nr=50, Ntheta=50, Nz=50, R=80, thickness=0.05,
    time=100, dt=0.1, k=0.04, rho=270, cp=1100
)
```

## Scripts

Run pre-configured simulations:

```bash
python scripts/run_1d_steady.py
python scripts/run_2d_steady.py
python scripts/run_2d_transient.py
python scripts/run_heat_shield_2d.py
python scripts/run_heat_shield_3d.py
```

## Project Structure

```
cfd/
├── src/
│   └── cfd/
│       ├── solvers/      # Solver implementations
│       ├── utils/        # Utility functions
│       └── visualization/ # Plotting functions
├── scripts/              # Executable scripts
├── config/               # Configuration files
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Boundary Conditions

The package supports:
- **Dirichlet**: Fixed temperature
- **Neumann**: Fixed heat flux
- **Robin**: Mixed boundary condition

## Disclosure

AI was used to assist with code structure and with writing function descriptions

