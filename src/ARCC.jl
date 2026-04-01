module ARCC

using LinearAlgebra, Printf
using Einsum  
using NPZ
using TensorOperations
using Glob
using Plots
using Base.Iterators: product
using NLsolve
using ForwardDiff
using LinearMaps
using Arpack

# Struct files
include("Structs.jl")
include("Constructors.jl")

# Files corresponding "ARCC"
include("Slices.jl")
include("Projectors.jl")
include("Orthogonal.jl")
include("Integrals..jl")
include("Fock.jl")
include("Theta.jl")
include("Energy.jl")
include("ARCCDeqns.jl")
include("FixedPointFormulation.jl")
include("SpectralAnalyzer.jl")
include("UsingNLsolve.jl")
include("FPIteration.jl")
include("NKSolvers.jl")
include("Preconditoner.jl")
include("PreconditionedNKSolvers.jl")

# This contains helper functions to make test files less verbose
include("Factory.jl")






end 