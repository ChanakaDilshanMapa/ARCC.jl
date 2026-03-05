export Slices, Projectors

struct Slices
    Cocc::AbstractMatrix
    Cvir::AbstractMatrix
    Tocc::AbstractMatrix
    Tvir::AbstractMatrix
    Tbarocc::AbstractMatrix
    Tbarvir::AbstractMatrix
end

struct Projectors
    P::AbstractMatrix
    Q::AbstractMatrix
    Pbar::AbstractMatrix
    Qbar::AbstractMatrix
end

export PhysAOEris

struct PhysAOEris
    physaoeris::AbstractArray
end

export Integrals, CoulombIntegrals

struct Integrals
    piint::AbstractArray
    a::AbstractArray
    b::AbstractArray
    c::AbstractArray
    cperm::AbstractArray
    d::AbstractArray
end

struct CoulombIntegrals
    j::Function
    k::Function
    jθ::Function
    kθ::Function
end

export FockOperators, FockDiagOffs, FockIntegrals

struct FockOperators
    Fo::AbstractArray
    Fv::AbstractArray
end

struct FockDiagOffs
    Fodiag::AbstractArray
    Fvdiag::AbstractArray
    Fooff::AbstractArray
    Fvoff::AbstractArray
end

struct FockIntegrals 
    Go::Function
    Gv::Function
end

export FixedPointElements

struct FixedPointElements
    R::Function
    ΔF::Function
    denom::AbstractArray
end