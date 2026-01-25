# Finite Difference Heat Transfer Solver

A Python package for solving heat conduction problems using finite difference methods with proper grid convergence and boundary condition handling.

## Features

- **1D Steady-State Solver**: Solves Poisson equation for 1D heat conduction with Dirichlet/Neumann boundary conditions
- **2D Steady-State Solver**: Solves 2D Poisson equation with grid-convergent source terms and flexible boundary conditions
- **2D Transient Solver**: Time-dependent 2D heat conduction with proper Neumann boundary condition support
- **Heat Shield Solvers**: 2D and 3D cylindrical coordinate solvers for heat shield simulations
- **Visualization**: Focused visualization tools including r-z cross-sections with adaptive time intervals

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

x, T = solve_1d_steady(
    num_nodes=100, length=10, k=5,
    BCL_type="dirichlet", BCR_type="neumann",
    sl=0, sr=10,  # Left temp, right flux
    source_pos=5, source_strength=1000
)
```

### 2D Steady-State
```python
from cfd.solvers.poisson_2d import solve_2d_steady

x, y, T = solve_2d_steady(
    numXNodes=50, numYNodes=50, xL=10, yL=10, k=0.001,
    BCW_type="neumann", BCE_type="neumann",
    BCN_type="dirichlet", BCS_type="dirichlet",
    source_i=25, source_j=25, source_strength=1000,
    SW=5, SE=-5, SN=100, SS=100  # Flux for Neumann, temp for Dirichlet
)
```

### 2D Transient
```python
from cfd.solvers.transient_2d import solve_2d_transient

x, y, transient = solve_2d_transient(
    numXNodes=50, numYNodes=50, xL=10, yL=10,
    time=100, dt=0.05, k=700, rho=1000, cp=10, T0=0,
    BCW_type="neumann", BCE_type="neumann",
    BCN_type="dirichlet", BCS_type="dirichlet",
    SW=5, SE=-5, SN=100, SS=100,  # Flux for Neumann, temp for Dirichlet
    source_x=5.0, source_y=5.0, source_strength=10  # Coordinates, not indices
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

## Visualization

The package includes focused visualization tools:

- **2D Animated Plots**: `plot_2d_animated()` for Cartesian coordinates
- **Cylindrical 2D Animated**: `plot_cylindrical_2d_animated()` for r-θ plane
- **r-z Cross-Sections**: `plot_rz_cross_sections()` for 3D heat shield visualization
  - Shows temperature evolution in radial-depth plane
  - Adaptive time intervals based on simulation duration
  - Automatically adjusts number of snapshots (5-15) based on total time

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

The package supports Dirichlet and Neumann boundary conditions:

- **Dirichlet**: Fixed temperature at boundary
  - Specify temperature value directly
  - Example: `SW=100` means 100°C at west boundary

- **Neumann**: Fixed heat flux at boundary
  - Specify flux value (positive = heat flowing INTO domain, negative = heat flowing OUT)
  - Example: `SW=5` means 5 W/m² flowing into domain from west
  - Example: `SW=-5` means 5 W/m² flowing out of domain from west

**Important Notes:**
- 2D steady-state solver: Source location uses indices (`source_i`, `source_j`)
- 2D transient solver: Source location uses coordinates (`source_x`, `source_y`) with automatic bounds checking
- All solvers now properly handle boundary condition types and values
- Grid convergence: Source terms are properly scaled for consistent results across different grid resolutions

## Disclosure

AI was used to assist with code structure and with writing function descriptions

