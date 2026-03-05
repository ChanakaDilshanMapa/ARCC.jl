# AUCC.jl

**Arbitrary Unitary Coupled Cluster Doubles (AUCC) Theory Implementation in Julia**


## Overview

AUCC.jl is a high-performance Julia package implementing the Arbitrary Unitary Coupled Cluster Doubles (AUCC) method for quantum chemistry calculations. This package provides a complete framework for solving coupled-cluster equations directly in the atomic orbital (AO) basis, enabling basis-set invariant formulations and efficient computational approaches.

### Key Features

- **AO-Based Formulation**: Direct implementation of coupled-cluster theory in the atomic orbital basis
- **Basis Invariance**: Support for arbitrary unitary transformations while maintaining equivalence to molecular orbital (MO) results
- **Multiple Solvers**: Fixed-point iteration, Newton-Krylov, Anderson acceleration, and gradient descent with Barzilai-Borwein step size
- **Tensor Operations**: Efficient tensor contractions using `TensorOperations.jl` and `Einsum.jl`
- **Comprehensive Testing**: Extensive test suite with reference data from PySCF

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Theoretical Background](#theoretical-background)
- [Package Architecture](#package-architecture)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [Solver Methods](#solver-methods)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Citation](#citation)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ChanakaDilshanMapa/AUCC.jl")
```

Or in the Julia REPL package mode (`]`):
```
add https://github.com/ChanakaDilshanMapa/AUCC.jl
```

## Quick Start

```julia
using AUCC
using LinearAlgebra
using NPZ

# Load molecular data (overlap, kinetic, potential matrices, and ERIs)
S = npzread("overlap_matrix.npy")
T = npzread("kinetic_energy_matrix.npy")
A = npzread("nuclear_potential_matrix.npy")
eris = npzread("ERI.npy")
nocc = npzread("nocc.npy")
n_b = size(S, 1)

# Orthogonalize basis
new_S, new_T, new_A, new_eris = orthogonalize(S, T, A, eris)

# Set up calculation
Cscf = # ... your SCF coefficients
f = # ... your Fock matrix
peris = make_physaoeris(new_eris)

# Run AUCC calculation with identity transformation (MO basis)
T_I = Matrix{Float64}(I, n_b, n_b)
Tbar = T_bar(T_I, new_S)
slice = make_slices(Cscf, T_I, Tbar, nocc, n_b)
proj = make_projectors(slice)
int = make_integrals(proj, peris)
# ... continue with solver
```

## Theoretical Background

### Coupled-Cluster Doubles (CCD) Theory

The Coupled-Cluster Doubles method approximates the many-electron wavefunction as:

$$|\Psi\rangle = e^{\hat{T}_2}|\Phi_0\rangle$$

where $\hat{T}_2$ is the doubles excitation operator:

$$\hat{T}_2 = \frac{1}{4}\sum_{ijab} t_{ij}^{ab} \hat{a}^\dagger_a \hat{a}^\dagger_b \hat{a}_j \hat{a}_i$$

### Atomic Orbital Formulation

AUCC.jl implements CCD theory directly in the atomic orbital basis. The key innovation is the ability to work with arbitrary unitary transformations $T$ while maintaining equivalence to the canonical MO solution.

The AO amplitude tensor $\theta_{\mu\nu\lambda\sigma}$ is related to the MO amplitudes $t_{ij}^{ab}$ via:

$$\theta_{\mu\nu\lambda\sigma} = \sum_{ijab} \bar{T}_{\mu i} \bar{T}_{\nu j} \bar{T}_{\lambda a} \bar{T}_{\sigma b} t_{ij}^{ab}$$

where $\bar{T}_{\mu i} = \sum_\alpha S_{\mu\alpha} T_{\alpha i}$ represents the metric-transformed orbital coefficients.

### AUCC Equations

The AUCC residual equations take the form:

$$Z_{\mu\nu\lambda\sigma} = D_{\mu\nu\lambda\sigma} + \Delta F_{\mu\nu\lambda\sigma} + R_{\mu\nu\lambda\sigma}$$

where:
- $D$: Direct integral contributions
- $\Delta F$: Off-diagonal Fock operator corrections
- $R$: Residual terms including Coulomb and exchange interactions

## Package Architecture

AUCC.jl is organized into modular components:

```
AUCC.jl/
├── src/
│   ├── AUCC.jl                    # Main module
│   ├── Structs.jl                 # Data structure definitions
│   ├── Constructors.jl            # Constructor functions
│   ├── Slices.jl                  # Orbital space partitioning
│   ├── Projectors.jl              # Projection operators (P, Q, P̄, Q̄)
│   ├── Orthogonal.jl              # Basis orthogonalization
│   ├── Integrals..jl              # Electron repulsion integrals
│   ├── Fock.jl                    # Fock operator construction
│   ├── Theta.jl                   # AO-MO amplitude transformations
│   ├── Energy.jl                  # Correlation energy evaluation
│   ├── AUCCDeqns.jl              # AUCC residual equations
│   ├── FixedPointFormulation.jl   # Fixed-point solver components
│   ├── Optimizers.jl              # Various solver algorithms
│   └── Factory.jl                 # High-level workflow factories
└── test/
    ├── test_*.jl                  # Unit tests
    └── pyscf_data/                # Reference data
```

## Core Components

### 1. Data Structures (`Structs.jl`)

#### `Slices`
Partitions the orbital space into occupied and virtual subspaces:
```julia
struct Slices
    Cocc::AbstractMatrix      # SCF occupied orbitals
    Cvir::AbstractMatrix      # SCF virtual orbitals
    Tocc::AbstractMatrix      # Transformed occupied orbitals
    Tvir::AbstractMatrix      # Transformed virtual orbitals
    Tbarocc::AbstractMatrix   # Metric-transformed occupied
    Tbarvir::AbstractMatrix   # Metric-transformed virtual
end
```

#### `Projectors`
Defines projection operators for occupied and virtual spaces:
```julia
struct Projectors
    P::AbstractMatrix         # Occupied space projector
    Q::AbstractMatrix         # Virtual space projector
    Pbar::AbstractMatrix      # Metric-transformed occupied projector
    Qbar::AbstractMatrix      # Metric-transformed virtual projector
end
```

#### `Integrals`
Stores pre-computed electron repulsion integrals:
```julia
struct Integrals
    piint::AbstractArray      # Projected integrals π
    a::AbstractArray          # a-type integrals
    b::AbstractArray          # b-type integrals
    c::AbstractArray          # c-type integrals
    cperm::AbstractArray      # Permuted c-type integrals
    d::AbstractArray          # d-type integrals (inhomogeneity)
end
```

#### `FockOperators`
Fock operators in occupied and virtual subspaces:
```julia
struct FockOperators
    Fo::AbstractArray         # Occupied Fock operator
    Fv::AbstractArray         # Virtual Fock operator
end
```

### 2. Orbital Transformations (`Theta.jl`)

Transform between MO and AO amplitudes:

```julia
# MO → AO transformation
θ_fun = theta(slice)
θ_AO = θ_fun(t_MO)

# AO → MO transformation
t_fun = theta2mo_amp(slice)
t_MO = t_fun(θ_AO)
```

### 3. Integral Evaluation (`Integrals..jl`)

Compute various classes of integrals:

- **π-integrals**: Projected two-electron integrals
  ```julia
  piint = piintegrals(proj, peris)
  ```

- **a, b, c, d integrals**: Different projections of ERIs
  ```julia
  a = a_integral(proj, peris)  # Occupied-occupied block
  b = b_integral(proj, peris)  # Virtual-virtual block
  c = c_integral(proj, peris)  # Mixed occupied-virtual
  d = d_integral(proj, peris)  # Inhomogeneous term
  ```

- **Coulomb (j) and Exchange (k) integrals**: Amplitude-dependent terms
  ```julia
  j_fun = j_integral(int, slice)
  k_fun = k_integral(int, slice)
  j = j_fun(t_MO)
  k = k_fun(t_MO)
  ```

### 4. AUCC Equations (`AUCCDeqns.jl`)

The core residual equations:

```julia
# Build AUCC equations
eqn_fun = au_ccd_eqns(int, j, k, G_o, G_v)
Z = eqn_fun(θ)
```

The residual includes:
- Two-electron integral contractions
- Fock operator contributions
- Amplitude-dependent Coulomb and exchange terms

### 5. Fock Operators (`Fock.jl`)

Construct occupied and virtual Fock operators:

```julia
# Project Fock matrix
F_o = fock_o(proj, fock_ao)  # Occupied block
F_v = fock_v(proj, fock_ao)  # Virtual block

# Separate diagonal and off-diagonal parts
Fo_diag = fock_o_diag(fop)
Fo_off = fock_o_off(fop)
```

### 6. Fixed-Point Formulation (`FixedPointFormulation.jl`)

Implements fixed-point iteration:

$$\theta^{(n+1)} = \frac{D + \Delta F + R(\theta^{(n)})}{\text{denom}}$$

Key functions:
- `bigR`: Computes the residual function
- `deltaF`: Off-diagonal Fock contributions
- `ao_denominator`: Orbital energy denominators
- `ao_amps`: Fixed-point amplitude update
- `ao_fixed_point_iteration`: Iterative solver

### 7. Advanced Solvers (`Optimizers.jl`)

#### Newton-Krylov Solver
Jacobian-free Newton method with GMRES inner iterations:
```julia
θ_final = newton_krylov(θ_init, residual_fun; 
                        tol=1e-8, max_outer=300, m=50)
```

#### Anderson Acceleration (NLsolve)
Using the `NLsolve.jl` library:
```julia
θ_final = nl_solve(θ_init, residual_fun; 
                   maxiter=100, ftol=1e-12)
```

#### Gradient Descent with Barzilai-Borwein
Adaptive step-size gradient descent:
```julia
θ_final = gradient_descent_bb(θ_init, residual_fun; 
                               tol=1e-8, max_iter=500)
```

## Usage Examples

### Example 1: Standard AUCC Calculation

```julia
using AUCC, LinearAlgebra, NPZ

# Load data
S = npzread("overlap_matrix.npy")
T = npzread("kinetic_energy_matrix.npy")
A = npzread("nuclear_potential_matrix.npy")
eris = npzread("ERI.npy")
nocc = npzread("nocc.npy")
n_b = size(S, 1)

# Orthogonalize
new_S, new_T, new_A, new_eris = orthogonalize(S, T, A, eris)

# Prepare calculation (assume Cscf, f, t2 from SCF/CCD)
peris = make_physaoeris(new_eris)
T_transform = Matrix{Float64}(I, n_b, n_b)

# Build AUCC system
Tbar = T_bar(T_transform, new_S)
slice = make_slices(Cscf, T_transform, Tbar, nocc, n_b)
proj = make_projectors(slice)
int = make_integrals(proj, peris)
coulombint = make_coulomb_integrals(int, slice)
fop = make_fock_operators(proj, f)
fd = make_fock_diags_and_offs(fop)

# Solve with fixed-point iteration
elt = make_fixed_point_elements(int, coulombint, slice, fd, n_b)
amp_fun = ao_amps(int, elt)
t_fun = theta2mo_amp(slice)
initial_guess = zeros(n_b, n_b, n_b, n_b)

θ_final, diffs = ao_fixed_point_iteration(amp_fun, initial_guess, t_fun, 300, 1e-10)

# Compute correlation energy
E_corr = corr_ene(int, θ_final)
```

### Example 2: Using Factory Functions

Factory functions provide streamlined workflows:

```julia
# AUCC equations factory
run_auccd = auccd_factory(new_S, t2, nocc, n_b, Cscf, f, peris)
Z_MO = run_auccd(Matrix{Float64}(I, n_b, n_b))
@assert norm(Z_MO) < 1e-8  # Should converge

# Fixed-point solver factory
run_fp = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, 
                             initial_guess, 300, 1e-10)
θ_final, θ_benchmark, diffs = run_fp(T_transform)

# Newton-Krylov solver factory
run_nk = nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, 
                    initial_guess, 300, 1e-8, 50)
θ_final, θ_benchmark = run_nk(T_transform)
```

### Example 3: Analyzing Fixed-Point Stability

```julia
# Analyze spectral radius of Jacobian
analyzer = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, 1e-6)
conclusion = analyzer(T_transform)
# Returns: "attractor", "repulsor", or "inconclusive"
```

### Example 4: Different Basis Transformations

```julia
# Identity (MO basis)
T_I = Matrix{Float64}(I, n_b, n_b)
Z_MO = run_auccd(T_I)

# Small perturbation
T_perturbed = Matrix(qr(Matrix(I, n_b, n_b) + 1e-2*ones(n_b, n_b)).Q)
Z_perturbed = run_auccd(T_perturbed)

# Full AO basis
T_AO = Cscf
Z_AO = run_auccd(T_AO)

# All should give equivalent converged residuals
@assert norm(Z_MO) < 1e-8
@assert norm(Z_perturbed) < 1e-8
@assert norm(Z_AO) < 1e-8
```

## Solver Methods

### Fixed-Point Iteration

The simplest approach, suitable for well-conditioned systems:

**Pros:**
- Simple implementation
- Low memory footprint
- Guaranteed convergence for spectral radius < 1

**Cons:**
- Slow for poorly conditioned systems
- May not converge if spectral radius ≥ 1

**Usage:**
```julia
θ, diffs = ao_fixed_point_iteration(amp_fun, initial_guess, t_fun, 
                                     max_iter, tol, verbose)
```

### Newton-Krylov Method

Jacobian-free Newton with GMRES for linear solves:

**Pros:**
- Quadratic convergence near solution
- No explicit Jacobian required
- Handles stiff systems well

**Cons:**
- Higher computational cost per iteration
- Requires good initial guess

**Usage:**
```julia
θ_final = newton_krylov(θ_init, residual_fun; 
                        tol=1e-8, max_outer=300, m=50)
```

### Anderson Acceleration (NLsolve)

Uses `NLsolve.jl` with Anderson mixing:

**Pros:**
- Often faster than fixed-point
- Stable for many problems
- Well-tested implementation

**Cons:**
- Requires additional dependency
- May require tuning for some systems

**Usage:**
```julia
θ_final = nl_solve(θ_init, residual_fun; 
                   maxiter=100, ftol=1e-12)
```

### Gradient Descent with Barzilai-Borwein

Adaptive step-size gradient descent:

**Pros:**
- Simple and robust
- Adaptive step size
- Good for exploration

**Cons:**
- Slower convergence than Newton methods
- Only first-order convergence

**Usage:**
```julia
θ_final = gradient_descent_bb(θ_init, residual_fun; 
                               tol=1e-8, max_iter=500)
```

## API Documentation

### Main Module Functions

#### Orthogonalization
```julia
orthogonalize(S, T, A, eris) → (new_S, new_T, new_A, new_eris)
```
Orthogonalizes the atomic orbital basis using $S^{-1/2}$ transformation.

#### Constructors
```julia
make_slices(C, T, Tbar, nocc, nbasis) → Slices
make_projectors(slice) → Projectors
make_physaoeris(aoeris) → PhysAOEris
make_integrals(proj, peris) → Integrals
make_coulomb_integrals(int, slice) → CoulombIntegrals
make_fock_operators(proj, fock_ao) → FockOperators
make_fock_diags_and_offs(fop) → FockDiagOffs
make_fixed_point_elements(int, coulombint, slice, fd, nbasis) → FixedPointElements
```

#### Core Calculations
```julia
corr_ene(int, θ) → Float64
```
Computes the CCD correlation energy from AO amplitudes.

```julia
au_ccd_eqns(int, j, k, G_o, G_v) → Function
```
Returns residual function for AUCC equations.

```julia
theta(slice) → Function
```
Creates MO → AO amplitude transformation function.

```julia
theta2mo_amp(slice) → Function
```
Creates AO → MO amplitude transformation function.

### Solver Functions

```julia
ao_fixed_point_iteration(amp_fun, initial_guess, t_fun, max_iter, tol, verbose)
```

```julia
newton_krylov(θ, residual_fun; tol, max_outer, m, verbose)
```

```julia
nl_solve(θ_init, residual_fun; maxiter, ftol, verbose)
```

```julia
gradient_descent_bb(θ, residual_fun; tol, max_iter, verbose)
```

### Analysis Functions

```julia
compute_jacobian(amp_fun, t_fun, θ_benchmark, perturbation)
```
Computes numerical Jacobian via finite differences.

```julia
analyze_fixed_point(amp_fun, t_fun, θ_benchmark, perturbation) → Float64
```
Returns spectral radius of Jacobian (< 1 means stable fixed point).

## Testing

The package includes comprehensive tests with reference data from PySCF:

```julia
using Pkg
Pkg.test("AUCC")
```

### Test Categories

1. **`test_auccd_eqns.jl`**: Validates AUCC residual equations
2. **`test_fixed_point.jl`**: Tests fixed-point solver
3. **`test_nk_solver.jl`**: Tests Newton-Krylov implementation
4. **`test_nlsolve.jl`**: Tests Anderson acceleration
5. **`test_gd_bb.jl`**: Tests gradient descent solver
6. **`test_orthogonal.jl`**: Validates orthogonalization
7. **`test_projectors.jl`**: Tests projection operators
8. **`test_theta.jl`**: Tests amplitude transformations
9. **`test_corr_ene.jl`**: Validates correlation energy
10. **`test_ao_amps_ortho.jl`**: Tests AO amplitude calculations

### Test Molecules

Reference data is provided for:
- BeH₂, BH₃, C₂, CH₂, CO, F₂, H₂, H₂O, LiH, N₂, NH₃, O₂

## Dependencies

- **Julia** ≥ 1.0
- **LinearAlgebra.jl**: Linear algebra operations
- **Einsum.jl**: Einstein summation notation
- **TensorOperations.jl**: Efficient tensor contractions
- **NPZ.jl**: NumPy file I/O
- **NLsolve.jl**: Nonlinear equation solvers
- **ForwardDiff.jl**: Automatic differentiation
- **Plots.jl**: Visualization
- **Test.jl**: Unit testing
- **GTO.jl**: Gaussian-type orbitals
- **WTP.jl**: Helper utilities
- **CCD.jl**: Reference CCD implementation

## Performance Considerations

1. **Tensor Contractions**: Use `@tensor` macro for optimal performance
2. **Memory**: Pre-allocate arrays when possible
3. **Convergence**: Newton-Krylov typically requires fewer iterations but higher cost per iteration
4. **Basis Size**: Computational cost scales as $\mathcal{O}(N^6)$ for $N$ basis functions

## Citation

If you use AUCC.jl in your research, please cite:

```bibtex
@software{aucc_jl,
  author = {Chanaka Dilshan Mapa},
  title = {AUCC.jl: Atomic Orbital Unitary Coupled Cluster Doubles in Julia},
  year = {2025},
  url = {https://github.com/ChanakaDilshanMapa/AUCC.jl}
}
```

## Acknowledgments

- PySCF for providing reference quantum chemistry calculations
- Julia community for excellent scientific computing ecosystem
- Contributors and testers of this package

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub:
https://github.com/ChanakaDilshanMapa/AUCC.jl/issues

---

**Author**: Chanaka Dilshan Mapa  
**Email**: namalisha@gmail.com  
**Repository**: https://github.com/ChanakaDilshanMapa/AUCC.jl
