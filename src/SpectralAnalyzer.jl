export compute_jacobian, analyze_fixed_point

function compute_jacobian(amp_fun, t_fun, θ_benchmark, purt)
    θ_vec = vec(θ_benchmark)
    n = length(θ_vec)

    F = θv -> begin
        θ = reshape(θv, size(θ_benchmark))
        t = t_fun(θ)
        vec(amp_fun(t))
    end

    F_θ = F(θ_vec)
    m = length(F_θ)
    J = zeros(m, n)

    for i in 1:n
        e = zeros(n)
        e[i] = purt
        J[:, i] = (F(θ_vec + e) - F_θ) / purt
    end

    return J
end


function estimate_spectral_radius(amp_fun, t_fun, θ_benchmark;
                                  purt=1e-6, maxiter=25, tol=1e-8,
                                  v0::Union{Nothing,AbstractVector}=nothing)
    θ_vec = vec(θ_benchmark)

    F = θv -> begin
        θ = reshape(θv, size(θ_benchmark))
        t = t_fun(θ)
        vec(amp_fun(t))
    end

    Fθ = F(θ_vec)
    if v0 === nothing        
        v = fill(1.0, length(θ_vec))
        v ./= norm(v)
    else
        v = copy(v0)
        v ./= norm(v)
    end
    λ_old = 0.0

    for iter in 1:maxiter
        Jv = (F(θ_vec + purt*v) - Fθ) / purt
        λ = dot(v, Jv)
        v = Jv / norm(Jv)
        abs(λ - λ_old) < tol && break
        λ_old = λ
    end

    return abs(λ_old)
end