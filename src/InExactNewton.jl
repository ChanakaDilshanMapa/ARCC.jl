export inexact_newton 

function inexact_newton(
    θ::Array{Float64, 4},
    residual_fun::Function;
    fop,
    tol=1e-8,
    max_outer=300,
    m=5,
    verbose=true,
    η=0.5
)

    θ_shape = size(θ)
    println("θ_shape: ", θ_shape)
    n = prod(θ_shape)

    num_residual_evals = Ref(0)

    function vec_residual(θ_vec::Vector{Float64})
        θ_tensor = reshape(θ_vec, θ_shape)
        num_residual_evals[] += 1
        return vec(residual_fun(θ_tensor))
    end

    function Jv(v::Vector{Float64})
        Jv_tensor = P_inv_action(fop, θ_shape, v)
        return vec(Jv_tensor)
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)
    normr = norm(r)
    verbose && println("NK: initial ||r|| = $normr")

    newton_residuals_pre  = Float64[]
    newton_residuals_post = Float64[]

    push!(newton_residuals_pre, normr)

    k = 0

    while normr > tol && k < max_outer
        β = normr
        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= r / β
        verbose && println("Newton iter $k: ||r|| = $normr")

        jmax = 0

        for j = 1:m
            w = Jv(V[:,j])   

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end

            H[j+1,j] = norm(w)
            if H[j+1,j] < 1e-14
                jmax = j
                break
            end

            V[:,j+1] .= w / H[j+1,j]
            jmax = j

            Hj = H[1:j+1, 1:j]
            e1 = zeros(j+1); e1[1] = β
            resid_inner = norm(e1 - Hj * (Hj \ e1))

            verbose && println("  GMRES $j: inner = $resid_inner")

            if resid_inner <= η * β
                verbose && println("  GMRES stopping criterion satisfied: $resid_inner <= $(η * β)")
                break
            end
        end

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β

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
        verbose && println("Converged in $k Newton iterations. Final ||r|| = $normr")
    else
        verbose && println("Did NOT converge in $k Newton iterations. Final ||r|| = $normr")
    end

    verbose && println("Final ||r|| = $normr")
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)
    return θ_final,
           newton_residuals_pre,
           newton_residuals_post,
           num_residual_evals[]
end