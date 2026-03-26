export bigR
export deltaF
export ao_denominator
export ao_amps

function bigR(int::Integrals, coulombint::CoulombIntegrals, slice::Slices)
    a = int.a
    b = int.b
    πint = int.piint

    j_fun = coulombint.j
    k_fun = coulombint.k

    θ_fun =  theta(slice)

    return t -> begin
        θ = θ_fun(t)  
        j = j_fun(t)
        k = k_fun(t)
 
        @tensor R[μ,ν,λ,σ] := a[μ,ν,α,β] * θ[α,β,λ,σ]

        @tensor R[μ,ν,λ,σ] += θ[μ,ν,α,β] * b[α,β,λ,σ]

        tmp = similar(R)
        @tensor tmp[μ,ν,γ,δ] := θ[μ,ν,α,β] * πint[α,β,γ,δ]
        @tensor R[μ,ν,λ,σ] += tmp[μ,ν,γ,δ] * θ[γ,δ,λ,σ]

        temp = similar(R)
        @tensor temp[α,λ] := (2 * θ[γ,δ,λ,β] - θ[γ,δ,β,λ]) * πint[α,β,γ,δ]
        @tensor tmp[μ,ν,λ,σ] := temp[α,λ] * θ[μ,ν,α,σ]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        @tensor temp[μ,α] := (2 * θ[μ,β,γ,δ] - θ[μ,β,δ,γ]) * πint[γ,δ,α,β]
        @tensor tmp[μ,ν,λ,σ] := temp[μ,α] * θ[α,ν,λ,σ]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        @tensor tmp[μ,ν,λ,σ] := j[μ,α,λ,γ] * (2 * θ[γ,ν,α,σ] - θ[γ,ν,σ,α])
        R += tmp + permutedims(tmp, (2,1,4,3))
    
        @tensor tmp[μ,ν,λ,σ] := k[μ,α,γ,λ] * θ[γ,ν,α,σ]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        @tensor tmp[μ,ν,λ,σ] := k[μ,α,γ,σ] * θ[γ,ν,λ,α]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        return R
    end
end

function deltaF(fd::FockDiagOffs, slice::Slices)
    Fo_off = fd.Fooff
    Fv_off = fd.Fvoff   

    θ_fun =  theta(slice)
    return t -> begin
        θ = θ_fun(t)
        @tensor ΔF[μ,ν,λ,σ] := Fv_off[α,λ] * θ[μ,ν,α,σ] + Fv_off[α,σ] * θ[μ,ν,λ,α] - Fo_off[μ,α] * θ[α,ν,λ,σ] - Fo_off[ν,α] * θ[μ,α,λ,σ]        
    end    
    return ΔF
end

function ao_denominator(fd::FockDiagOffs, nbasis)
    fo_diag = fd.Fodiag
    fv_diag = fd.Fvdiag

    denom = reshape(fo_diag, nbasis, 1, 1, 1) .+ 
            reshape(fo_diag, 1, nbasis, 1, 1) .- 
            reshape(fv_diag, 1, 1, nbasis, 1) .- 
            reshape(fv_diag, 1, 1, 1, nbasis)
    
    return denom
end



function ao_amps(int::Integrals, elt::FixedPointElements, slice::Slices, shift)
    d = int.d  
    R_fun = elt.R       
    delF_fun = elt.ΔF           
    denom = elt.denom   
    
    θ_fun =  theta(slice)

    safe_denom = denom .+ shift * sign.(denom)
    safe_denom[abs.(safe_denom) .< shift] .= shift
    
    # regularizer = 1e10 .* (ones(size(safe_denom)))
    return t -> begin
        ΔF = delF_fun(t)
        R = R_fun(t)
        θ = θ_fun(t)

        # amps = (d .+ ΔF .+ R) ./ safe_denom

        amps =  ((d .+ ΔF .+ R) .+ ((safe_denom .- denom) .* θ)) ./ safe_denom
        # amps =  (((100 .* d) .+ (100 .* ΔF) .+ (100 .* R)) .+ (100 .* ((safe_denom .- denom) .* θ))) ./ (100 .* safe_denom)
        
        # amps =  (((d .* 10^3) .+ ΔF .+ R) .+ ((safe_denom .- denom) .* θ) .+ regularizer .* θ) ./ (safe_denom + regularizer)

        return amps
    end
end

export ao_fixed_point_iteration

function ao_fixed_point_iteration(amp_fun, initial_guess, t_fun, max_iter, tol, verbose=false)    
    θ = copy(initial_guess)
    converged = false
    diffs = Float64[]
    
    for iter in 1:max_iter 
        t = t_fun(θ)

        θ_new = amp_fun(t)    
 
        if !all(isfinite.(θ_new))
            if verbose
                println("Diverged (Nan detected) after $iter iterations")
            end
            break
        end

        diff = norm(θ_new - θ)
        push!(diffs, diff)

        if diff > 100
            if verbose
                println("Diverged (Δ > 100) after $iter iterations")
            end
            break
        end

        if verbose
            println("Iteration $iter: Δ = ", diff)
        end        

        θ = θ_new          
        
        if diff < tol
            converged = true
            if verbose
                println("Convergence achieved after $iter iterations")
            end
            break
        end
    end
    
    if !converged
        verbose && println("Warning: Failed to converge after $max_iter iterations")
    end
    
    return θ, diffs
end

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


