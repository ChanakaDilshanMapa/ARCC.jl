"""
The following two Newton-Krylov solvers can be used in any guage.
One with logs and other one without logs
"""

export nk_solver_with_logs

function nk_solver_with_logs(
    θ::Array{Float64,4},
    residual_fun::Function;
    tol=1e-8,
    max_outer=300,
    m=20,
    verbose=true,
    η=0.1
)

    θ_shape = size(θ)
    n = prod(θ_shape)

    num_residual_evals = Ref(0)

    function vec_residual(θ_vec::Vector{Float64})
        θ_tensor = reshape(θ_vec, θ_shape)
        num_residual_evals[] += 1
        return vec(residual_fun(θ_tensor))
    end

    function Jv(θ_vec::Vector{Float64}, r::Vector{Float64}, v::Vector{Float64})
        δ = sqrt(eps(Float64)) * (1 + norm(θ_vec)) / max(norm(v), 1e-12)
        return (vec_residual(θ_vec .+ δ .* v) .- r) ./ δ
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)
    normr = norm(r)

    verbose && println("NK: initial ||r|| = $normr")

    newton_residuals_pre  = Float64[]
    newton_residuals_post = Float64[]
    gmres_residuals       = Vector{Vector{Float64}}()

    push!(newton_residuals_pre, normr)

    k = 0
    total_inner = 0

    while normr > tol && k < max_outer

        β = normr
        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= r / β

        verbose && println("Newton iter $k: ||r|| = $normr")

        gmres_current = Float64[]
        jmax = 0

        for j = 1:m

            w = Jv(θ_vec, r, V[:,j])

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] .* V[:,i]
            end

            H[j+1,j] = norm(w)

            if H[j+1,j] < 1e-14
                jmax = j
                break
            end

            V[:,j+1] .= w ./ H[j+1,j]
            jmax = j

            Hj = H[1:j+1, 1:j]
            e1 = zeros(j+1)
            e1[1] = β

            s = Hj \ e1
            resid_inner = norm(e1 - Hj * s)

            push!(gmres_current, resid_inner)
            verbose && println("  GMRES $j: projected residual = $resid_inner")

            if resid_inner <= η * β
                verbose && println("  GMRES stopping criterion satisfied.")
                break
            end
        end

        push!(gmres_residuals, gmres_current)
        total_inner += jmax

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1)
        e1[1] = β

        s = Hj \ e1
        Δθ = -V[:,1:jmax] * s

        θ_vec .+= Δθ
        r = vec_residual(θ_vec)
        normr = norm(r)

        push!(newton_residuals_post, normr)

        if normr > tol
            push!(newton_residuals_pre, normr)
        end

        k += 1
    end

    if normr <= tol
        verbose && println("Converged in $k Newton and $total_inner GMRES iterations.")
    else
        verbose && println("Did NOT converge in $k Newton and $total_inner GMRES iterations.")
    end

    verbose && println("Final ||r|| = $normr")
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)

    return θ_final,
           newton_residuals_pre,
           newton_residuals_post,
           gmres_residuals,
           num_residual_evals[]
end

export nk_solver

function nk_solver(
    θ::Array{Float64, 4},
    residual_fun::Function;
    tol=1e-8,
    max_outer=300,
    m=5,
    verbose=true,
    η=0.1
)

    θ_shape = size(θ)
    n = prod(θ_shape)

    num_residual_evals = Ref(0) 

    function vec_residual(θ_vec::Vector{Float64})
        θ_tensor = reshape(θ_vec, θ_shape)
        num_residual_evals[] += 1
        return vec(residual_fun(θ_tensor))
    end
 
    function Jv(θ_vec::Vector{Float64}, r::Vector{Float64}, v::Vector{Float64})
        δ = sqrt(eps(Float64)) * (1 + norm(θ_vec)) / max(norm(v), 1e-12)  
        return (vec_residual(θ_vec .+ δ .* v) .- r) ./ δ
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)
    normr = norm(r)
    verbose && println("NK: initial ||r|| = $normr")

    k = 0
    total_inner = 0
    while normr > tol && k < max_outer
        β = normr
        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= r / β
        verbose && println("Newton iter $k: ||r|| = $normr")

        inner = 0
        for j = 1:m
            w = Jv(θ_vec, r, V[:,j])
            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end
            H[j+1,j] = norm(w)
            if H[j+1,j] < tol
                break
            end
            V[:,j+1] .= w / H[j+1,j]
            Hj = H[1:j+1, 1:j]
            e1 = zeros(j+1); e1[1] = β
            resid_inner = norm(Hj * (Hj \ e1) - e1)
            verbose && println("  GMRES $j: inner = $resid_inner")
            inner += 1
            
            # Eisenstat-Walker stopping criterion: stop GMRES when
            # ||J^{-1}r|| <= η * ||r||, estimated by projected residual
            if resid_inner <= η * β
                verbose && println("  GMRES stopping criterion satisfied: $resid_inner <= $(η * β)")
                break
            end

        end
        total_inner += inner

        jmax = findlast(!iszero, sum(abs.(H), dims=1))[2]
        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β
        s = Hj \ (β * (e1 ./ β))
        Δθ = -V[:,1:jmax] * s
        θ_vec .+= Δθ
        r = vec_residual(θ_vec)
        normr = norm(r)
        k += 1
    end

    if normr <= tol
        verbose && println("Converged in $k Newton and $total_inner GMRES iterations. Final ||r|| = $normr")
    else
        verbose && println("Did NOT converge in $k Newton and $total_inner GMRES iterations. Final ||r|| = $normr")
    end

    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)
    return θ_final, num_residual_evals[]
end