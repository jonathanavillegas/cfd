# Finite Difference Heat Transfer Solver

A Python package for solving heat conduction problems using finite difference methods.

## Features

- **1D Steady-State Solver**: Solves 1D heat conduction with Dirichlet and Neumann boundary conditions
- **2D Steady-State Solver**: Solves 2D heat conduction with Dirichlet and Neumann boundary conditions
- **2D Transient Solver**: Solves time-dependent 2D heat conduction with Dirichlet and Neumann boundary conditions
- **Heat Shield Solvers**: 2D and 3D cylindrical coordinate solvers for heat shield simulations
- **Visualization**: 2D animated plots, cylindrical plots, and r-z cross-sections

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

Run pre-configured simulations:

```bash
python scripts/run_1d_steady.py
python scripts/run_2d_steady.py
python scripts/run_2d_transient.py
python scripts/run_heat_shield_2d.py
python scripts/run_heat_shield_3d.py
```

Modify parameters in the script files to customize simulations.

## Visualization

Available plotting functions:

- `plot_2d_animated()`: 2D animated plots for Cartesian coordinates
- `plot_cylindrical_2d_animated()`: 2D animated plots for cylindrical coordinates (r-θ plane)
- `plot_rz_cross_sections()`: r-z cross-sections for 3D heat shield visualization

Example:
```python
from cfd.visualization.plot_3d import plot_rz_cross_sections

plot_rz_cross_sections(transient, r, theta, z, time=100, dt=1, theta_idx=0)
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

Two boundary condition types are supported:

- **Dirichlet**: Fixed temperature at boundary. Specify temperature value directly.
  - Example: `SW=100` sets 100°C at west boundary

- **Neumann**: Fixed heat flux at boundary. Positive flux = heat into domain, negative flux = heat out of domain.
  - Example: `SW=5` sets 5 W/m² into domain from west
  - Example: `SW=-5` sets 5 W/m² out of domain from west

**Notes:**
- Both 2D solvers use coordinates for source location (`source_x`, `source_y`) with bounds checking
- Source terms are scaled automatically to maintain consistent results across different grid resolutions

## Disclosure

AI was used to assist with code structure, with writing function descriptions, and with commenting code for readability. All finite difference methods and model logic were developed and implemented by me

