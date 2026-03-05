export nl_solve

function nl_solve(θ_init::Array{Float64,4}, residual_fun::Function;
                                 maxiter=100, ftol=1e-12, verbose=false)
    θ_shape = size(θ_init)
    function f!(F, θ_flat)
        θ = reshape(θ_flat, θ_shape)
        F .= vec(residual_fun(θ))
    end
    solution = nlsolve(f!, vec(copy(θ_init));
                       method=:anderson,
                       ftol=ftol,
                       xtol=ftol,
                       iterations=maxiter,
                       show_trace=verbose,
                       store_trace=verbose)
    θ_final = reshape(solution.zero, θ_shape)
    return θ_final
end

export newton_krylov


function newton_krylov(
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

# Extended version that records convergence history (Newton + GMRES)
export newton_krylov_with_logs

# function newton_krylov_with_logs(
#     θ::Array{Float64,4},
#     residual_fun::Function;
#     tol=1e-8,
#     max_outer=300,
#     m=5,
#     verbose=true,
#     η=0.1
# )
#     θ_shape = size(θ)
#     n = prod(θ_shape)

#     function vec_residual(θ_vec::Vector{Float64})
#         θ_tensor = reshape(θ_vec, θ_shape)
#         # num_residual_evals[] += 1
#         return vec(residual_fun(θ_tensor))
#     end

#     num_residual_evals = Ref(0)
#     function Jv(θ_vec::Vector{Float64}, r::Vector{Float64}, v::Vector{Float64})
#         δ = sqrt(eps(Float64)) * (1 + norm(θ_vec)) / max(norm(v), 1e-12)
#         # δ = 1e-6
#         num_residual_evals[] += 1
#         return (vec_residual(θ_vec .+ δ .* v) .- r) ./ δ
#     end

#     θ_vec = vec(copy(θ))
#     r = vec_residual(θ_vec)
#     normr = norm(r)
#     verbose && println("NK: initial ||r|| = $normr")

#     newton_residuals_pre = Float64[]  # residual before each Newton update
#     newton_residuals_post = Float64[] # residual after each Newton update
#     push!(newton_residuals_pre, normr)
#     gmres_residuals = Vector{Vector{Float64}}()  # per-Newton GMRES projected residual sequence

#     k = 0
#     total_inner = 0
#     while normr > tol && k < max_outer
#         β = normr
#         V = zeros(n, m+1)
#         H = zeros(m+1, m)
#         V[:,1] .= r / β
#         verbose && println("Newton iter $k: ||r|| = $normr")

#         inner = 0
#         gmres_current = Float64[]
#         for j = 1:m
#             w = Jv(θ_vec, r, V[:,j])
#             for i = 1:j
#                 H[i,j] = dot(V[:,i], w)
#                 w .-= H[i,j] * V[:,i]
#             end
#             H[j+1,j] = norm(w)
#             # if H[j+1,j] < 1e-14
#             #     break
#             # end
#             V[:,j+1] .= w / H[j+1,j]
#             Hj = H[1:j+1, 1:j]
#             e1 = zeros(j+1); e1[1] = β
#             resid_inner = norm(Hj * (Hj \ e1) - e1)
#             verbose && println("  GMRES $j: inner = $resid_inner")
#             push!(gmres_current, resid_inner)
#             inner += 1
            
#             if resid_inner <= η * β
#                 verbose && println("  GMRES stopping criterion satisfied: $resid_inner <= $(η * β)")
#                 break
#             end
#         end
#         push!(gmres_residuals, gmres_current)
#         total_inner += inner

#         jmax = findlast(!iszero, sum(abs.(H), dims=1))[2]
#         # jmax = m
#         Hj = H[1:jmax+1, 1:jmax]
#         e1 = zeros(jmax+1); e1[1] = β        
#         s = Hj \ (β * (e1 ./ β))
#         Δθ = -V[:,1:jmax] * s
#         θ_vec .+= Δθ
#         r = vec_residual(θ_vec)
#         normr = norm(r)
#         k += 1
#         push!(newton_residuals_post, normr)
#         if normr > tol
#             push!(newton_residuals_pre, normr)
#         end
#     end

#     if normr <= tol
#         verbose && println("Converged in $k Newton and $total_inner GMRES iterations. Final ||r|| = $normr")
#     else
#         verbose && println("Did NOT converge in $k Newton and $total_inner GMRES iterations. Final ||r|| = $normr")
#     end

#     verbose && println("Total residual evaluations: $(num_residual_evals[])")

#     θ_final = reshape(θ_vec, θ_shape)
#     return θ_final, newton_residuals_pre, newton_residuals_post, gmres_residuals, num_residual_evals[]
# end

function newton_krylov_with_logs(
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
        # δ = 1e-5
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

            # if H[j+1,j] < 1e-14
            #     jmax = j
            #     break
            # end

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
export newton_krylov_with_logs_preconditioned

function newton_krylov_with_logs_preconditioned(
    θ::Array{Float64,4},
    residual_fun::Function,
    precond_fun::Function;
    tol=1e-8,
    max_outer=300,
    m=5,
    verbose=true,
    η=0.1
)
    θ_shape = size(θ)
    n = prod(θ_shape)

    function vec_residual(θ_vec::Vector{Float64})
        θ_tensor = reshape(θ_vec, θ_shape)
        return vec(residual_fun(θ_tensor))
    end

    num_residual_evals = Ref(0)
    function Jv(θ_vec::Vector{Float64}, r::Vector{Float64}, v::Vector{Float64})
        δ = sqrt(eps(Float64)) * (1 + norm(θ_vec)) / max(norm(v), 1e-12)
        num_residual_evals[] += 1
        return (vec_residual(θ_vec .+ δ .* v) .- r) ./ δ
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)
    r̃ = precond_fun(r)
    normr = norm(r̃)
    verbose && println("NK (PC): initial ||r̃|| = $normr")

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
            w = precond_fun(w) 
            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end
            H[j+1,j] = norm(w)
            if H[j+1,j] < 1e-14
                break
            end
            V[:,j+1] .= w / H[j+1,j]
            Hj = H[1:j+1, 1:j]
            e1 = zeros(j+1); e1[1] = β
            resid_inner = norm(Hj * (Hj \ e1) - e1)
            verbose && println("  GMRES $j: inner = $resid_inner")
            inner += 1

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
        Δθ = -(V[:,1:jmax] * s)
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

export gradient_descent_bb

function gradient_descent_bb(θ::Array{Float64, 4}, residual_fun::Function;
                              tol=1e-8, max_iter=500, verbose=true)
    θ_vec = vec(copy(θ))
    r_vec = vec(residual_fun(reshape(θ_vec, size(θ))))
    norm_r = norm(r_vec)
    println("GD: initial ||r|| = $norm_r")

    α = 1.0  # start with a moderate step
    θ_prev, r_prev = copy(θ_vec), copy(r_vec)

    for k in 1:max_iter
        θ_vec .-= α .* r_vec
        r_new = vec(residual_fun(reshape(θ_vec, size(θ))))
        norm_r_new = norm(r_new)

        if verbose && (k % 10 == 0 || k == 1)
            println("iter $k: ||r|| = $norm_r_new (α = $α)")
        end

        # Check convergence
        if norm_r_new < tol
            println("Converged in $k iterations with ||r|| = $norm_r_new")
            return reshape(θ_vec, size(θ))
        end

        # Barzilai–Borwein step size
        s = θ_vec .- θ_prev
        y = r_new .- r_prev
        α = abs(dot(s, y)) / dot(y, y)  # adaptive step size estimate
        α = clamp(α, 1e-5, 10.0)        # safety bounds

        # Update
        θ_prev .= θ_vec
        r_prev .= r_new
        norm_r = norm_r_new
        r_vec .= r_new
    end

    println("Did not converge after $max_iter iterations, ||r|| = $norm_r")
    θ_final = reshape(θ_vec, θ_shape)

    return θ_final
end


