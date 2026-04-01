"""
The following two Newton-Krylov solvers are using a preconditioner, which can be used in any guage.
One with logs and other one without logs
"""

export preconditioned_nk_solver
export preconditioned_nk_solver_with_diis_logs

function preconditioned_nk_solver_with_logs(
    θ::Array{Float64,4},
    residual_fun::Function,
    fop;
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

    function apply_Pinv(rhs)
        β = norm(rhs)
        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= rhs / β

        for j = 1:m
            w = P_inv_action(fop, θ_shape, V[:,j])

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end

            H[j+1,j] = norm(w)
            if H[j+1,j] < tol
                break
            end

            V[:,j+1] .= w / H[j+1,j]
        end

        col_norms = vec(sum(abs.(H), dims=1))
        jmax = findlast(x -> x > 0, col_norms)
        jmax === nothing && error("GMRES failed: all columns zero")

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β

        s = Hj \ e1
        return -V[:,1:jmax] * s
    end

    p_inv_r = apply_Pinv(r)
    normr = norm(p_inv_r)

    verbose && println("NK: initial ||P⁻¹r|| = $normr")

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

        V[:,1] .= p_inv_r / β

        verbose && println("Newton iter $k: ||P⁻¹r|| = $normr")

        gmres_current = Float64[]
        jmax = 0

        for j = 1:m
            function preconditioned_Jv(v)

                Jv_val = Jv(θ_vec, r, v)

                normJ = norm(Jv_val)
                βJ = normJ

                VJ = zeros(n, m+1)
                HJ = zeros(m+1, m)

                VJ[:,1] .= Jv_val / βJ

                for jj = 1:m
                    wJ = P_inv_action(fop, θ_shape, VJ[:,jj])

                    for ii = 1:jj
                        HJ[ii,jj] = dot(VJ[:,ii], wJ)
                        wJ .-= HJ[ii,jj] * VJ[:,ii]
                    end

                    HJ[jj+1,jj] = norm(wJ)

                    if HJ[jj+1,jj] < tol
                        break
                    end

                    VJ[:,jj+1] .= wJ / HJ[jj+1,jj]
                end

                col_normsJ = vec(sum(abs.(HJ), dims=1))
                jmaxJ = findlast(x -> x > 0, col_normsJ)
                jmaxJ === nothing && error("Inner GMRES failed")

                HjJ = HJ[1:jmaxJ+1, 1:jmaxJ]

                e1J = zeros(jmaxJ+1)
                e1J[1] = βJ

                sJ = HjJ \ e1J

                return -VJ[:,1:jmaxJ] * sJ
            end

  
            w = preconditioned_Jv(V[:,j])

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end

            H[j+1,j] = norm(w)

            if H[j+1,j] < tol
                break
            end

            V[:,j+1] .= w / H[j+1,j]
            jmax = j

            Hj = H[1:j+1, 1:j]
            e1 = zeros(j+1); e1[1] = β

            s = Hj \ e1
            resid_inner = norm(e1 - Hj * s)

            push!(gmres_current, resid_inner)
            verbose && println("  GMRES $j: projected residual = $resid_inner")

            if resid_inner <= η * β
                verbose && println("  GMRES stopping criterion satisfied")
                break
            end
        end

        push!(gmres_residuals, gmres_current)
        total_inner += jmax

        col_norms = vec(sum(abs.(H), dims=1))
        jmax = findlast(x -> x > 0, col_norms)
        jmax === nothing && error("Outer GMRES failed")

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β

        s = Hj \ e1
        Δθ = -V[:,1:jmax] * s

        θ_vec .+= Δθ

        r = vec_residual(θ_vec)

        p_inv_r = apply_Pinv(r)
        normr = norm(p_inv_r)

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

    verbose && println("Final ||P⁻¹r|| = $normr")
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)
    return θ_final,
           newton_residuals_pre,
           newton_residuals_post,
           gmres_residuals,
           num_residual_evals[]
end

function preconditioned_nk_solver_with_diis_logs(
    θ::Array{Float64,4},
    residual_fun::Function,
    fop;
    tol=1e-8,
    max_outer=300,
    m=5,
    m_diis=3,
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

    function apply_Pinv(rhs::Vector{Float64})
        β = norm(rhs)
        if β < 1e-14
            return zeros(length(rhs))
        end

        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= rhs / β

        for j = 1:m
            w = P_inv_action(fop, θ_shape, V[:,j])

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end

            H[j+1,j] = norm(w)
            if H[j+1,j] < tol
                break
            end

            V[:,j+1] .= w / H[j+1,j]
        end

        col_norms = vec(sum(abs.(H), dims=1))
        jmax = findlast(x -> x > 0, col_norms)
        jmax === nothing && error("GMRES failed: all columns zero")

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1)
        e1[1] = β

        s = Hj \ e1
        return -V[:,1:jmax] * s
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)
    p_inv_r = apply_Pinv(r)
    normr = norm(p_inv_r)

    verbose && println("NK: initial ||P⁻¹r|| = $normr")

    newton_residuals_pre  = Float64[]
    newton_residuals_post = Float64[]
    gmres_residuals       = Vector{Vector{Float64}}()

    push!(newton_residuals_pre, normr)

    θ_list = Vector{Vector{Float64}}()
    r_list = Vector{Vector{Float64}}()

    k = 0
    total_inner = 0

    while k < max_outer && normr > tol
        for inner_iter in 1:m_diis
            if k >= max_outer || normr <= tol
                break
            end

            β = normr
            if β < 1e-14
                break
            end

            V = zeros(n, m+1)
            H = zeros(m+1, m)

            V[:,1] .= p_inv_r / β

            verbose && println("Newton iter $k: ||P⁻¹r|| = $normr")

            gmres_current = Float64[]
            jmax = 0

            for j = 1:m
                function preconditioned_Jv(v)
                    Jv_val = Jv(θ_vec, r, v)
                    normJ = norm(Jv_val)
                    βJ = normJ

                    if βJ < 1e-14
                        return zeros(length(v))
                    end

                    VJ = zeros(n, m+1)
                    HJ = zeros(m+1, m)

                    VJ[:,1] .= Jv_val / βJ

                    for jj = 1:m
                        wJ = P_inv_action(fop, θ_shape, VJ[:,jj])

                        for ii = 1:jj
                            HJ[ii,jj] = dot(VJ[:,ii], wJ)
                            wJ .-= HJ[ii,jj] * VJ[:,ii]
                        end

                        HJ[jj+1,jj] = norm(wJ)

                        if HJ[jj+1,jj] < tol
                            break
                        end

                        VJ[:,jj+1] .= wJ / HJ[jj+1,jj]
                    end

                    col_normsJ = vec(sum(abs.(HJ), dims=1))
                    jmaxJ = findlast(x -> x > 0, col_normsJ)
                    jmaxJ === nothing && error("Inner GMRES failed")

                    HjJ = HJ[1:jmaxJ+1, 1:jmaxJ]
                    e1J = zeros(jmaxJ+1)
                    e1J[1] = βJ

                    sJ = HjJ \ e1J
                    return -VJ[:,1:jmaxJ] * sJ
                end

                w = preconditioned_Jv(V[:,j])

                for i = 1:j
                    H[i,j] = dot(V[:,i], w)
                    w .-= H[i,j] * V[:,i]
                end

                H[j+1,j] = norm(w)

                if H[j+1,j] < tol
                    jmax = j
                    break
                end

                V[:,j+1] .= w / H[j+1,j]
                jmax = j

                Hj = H[1:j+1, 1:j]
                e1 = zeros(j+1)
                e1[1] = β

                s = Hj \ e1
                resid_inner = norm(e1 - Hj * s)

                push!(gmres_current, resid_inner)
                verbose && println("  GMRES $j: projected residual = $resid_inner")

                if resid_inner <= η * β
                    verbose && println("  GMRES stopping criterion satisfied")
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
            p_inv_r = apply_Pinv(r)
            normr = norm(p_inv_r)

            push!(newton_residuals_post, normr)
            push!(θ_list, copy(θ_vec))
            push!(r_list, copy(r))

            k += 1
            verbose && println("Iteration $k: Δ = $normr")

            if normr <= tol
                verbose && println("Converged in $k Newton and $total_inner GMRES iterations.")
                verbose && println("Final ||P⁻¹r|| = $normr")
                verbose && println("Total residual evaluations: $(num_residual_evals[])")

                θ_final = reshape(θ_vec, θ_shape)
                return θ_final,
                    newton_residuals_pre,
                    newton_residuals_post,
                    gmres_residuals,
                    num_residual_evals[]
            end

            push!(newton_residuals_pre, normr)
        end

        if k >= max_outer || normr <= tol
            break
        end

        k_diis = min(length(r_list), m_diis)

        if k_diis >= 2
            r_sub = r_list[end-k_diis+1:end]
            θ_sub = θ_list[end-k_diis+1:end]

            B = zeros(Float64, k_diis+1, k_diis+1)

            for i in 1:k_diis
                for j in 1:k_diis
                    B[i,j] = dot(r_sub[i], r_sub[j])
                end
                B[i,k_diis+1] = -1.0
                B[k_diis+1,i] = -1.0
            end

            rhs = zeros(Float64, k_diis+1)
            rhs[k_diis+1] = -1.0

            coeffs = B \ rhs
            c = coeffs[1:k_diis]

            θ_diis = zero(θ_vec)
            for i in 1:k_diis
                θ_diis .+= c[i] .* θ_sub[i]
            end

            r_diis = vec_residual(θ_diis)
            p_inv_r_diis = apply_Pinv(r_diis)
            normr_diis = norm(p_inv_r_diis)

            verbose && println("DIIS update: Δ = $normr_diis")

            if all(isfinite.(θ_diis)) && all(isfinite.(r_diis)) && all(isfinite.(p_inv_r_diis))
                θ_vec .= θ_diis
                r .= r_diis
                p_inv_r .= p_inv_r_diis
                normr = normr_diis

                if normr <= tol
                    verbose && println("Converged in $k Newton and $total_inner GMRES iterations.")
                    verbose && println("Final ||P⁻¹r|| = $normr")
                    verbose && println("Total residual evaluations: $(num_residual_evals[])")

                    θ_final = reshape(θ_vec, θ_shape)
                    return θ_final,
                        newton_residuals_pre,
                        newton_residuals_post,
                        gmres_residuals,
                        num_residual_evals[]
                end

                push!(newton_residuals_pre, normr)
            end
        end
    end

    if normr <= tol
        verbose && println("Converged in $k Newton and $total_inner GMRES iterations.")
    else
        verbose && println("Did NOT converge in $k Newton and $total_inner GMRES iterations.")
    end

    verbose && println("Final ||P⁻¹r|| = $normr")
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)

    return θ_final,
        newton_residuals_pre,
        newton_residuals_post,
        gmres_residuals,
        num_residual_evals[]
end

export preconditioned_nk_solver

function preconditioned_nk_solver(
    θ::Array{Float64,4},
    residual_fun::Function,
    fop;
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

    function apply_Pinv(rhs)
        β = norm(rhs)
        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= rhs / β

        for j = 1:m
            w = P_inv_action(fop, θ_shape, V[:,j])

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end

            H[j+1,j] = norm(w)
            if H[j+1,j] < tol
                break
            end

            V[:,j+1] .= w / H[j+1,j]
        end

        col_norms = vec(sum(abs.(H), dims=1))
        jmax = findlast(x -> x > 0, col_norms)
        jmax === nothing && error("GMRES failed: all columns zero")

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β

        s = Hj \ e1
        return -V[:,1:jmax] * s
    end

    p_inv_r = apply_Pinv(r)
    normr = norm(p_inv_r)

    verbose && println("NK: initial ||P⁻¹r|| = $normr")

    k = 0
    total_inner = 0


    while normr > tol && k < max_outer

        β = normr
        V = zeros(n, m+1)
        H = zeros(m+1, m)

        V[:,1] .= p_inv_r / β

        verbose && println("Newton iter $k: ||P⁻¹r|| = $normr")

        inner = 0

        for j = 1:m


            function preconditioned_Jv(v)

                Jv_val = Jv(θ_vec, r, v)

                normJ = norm(Jv_val)
                βJ = normJ

                VJ = zeros(n, m+1)
                HJ = zeros(m+1, m)

                VJ[:,1] .= Jv_val / βJ

                for jj = 1:m
                    wJ = P_inv_action(fop, θ_shape, VJ[:,jj])

                    for ii = 1:jj
                        HJ[ii,jj] = dot(VJ[:,ii], wJ)
                        wJ .-= HJ[ii,jj] * VJ[:,ii]
                    end

                    HJ[jj+1,jj] = norm(wJ)

                    if HJ[jj+1,jj] < tol
                        break
                    end

                    VJ[:,jj+1] .= wJ / HJ[jj+1,jj]
                end

                col_normsJ = vec(sum(abs.(HJ), dims=1))
                jmaxJ = findlast(x -> x > 0, col_normsJ)
                jmaxJ === nothing && error("Inner GMRES failed")

                HjJ = HJ[1:jmaxJ+1, 1:jmaxJ]

                e1J = zeros(jmaxJ+1)
                e1J[1] = βJ

                sJ = HjJ \ e1J

                return -VJ[:,1:jmaxJ] * sJ
            end

  
            w = preconditioned_Jv(V[:,j])

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

            if resid_inner <= η * β
                verbose && println("  GMRES stopping criterion satisfied")
                break
            end
        end

        total_inner += inner

        col_norms = vec(sum(abs.(H), dims=1))
        jmax = findlast(x -> x > 0, col_norms)
        jmax === nothing && error("Outer GMRES failed")

        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β

        s = Hj \ e1
        Δθ = -V[:,1:jmax] * s

        θ_vec .+= Δθ

        r = vec_residual(θ_vec)

        p_inv_r = apply_Pinv(r)
        normr = norm(p_inv_r)

        k += 1
    end

    if normr <= tol
        verbose && println("Converged in $k Newton and $total_inner GMRES iterations.")
    else
        verbose && println("Did NOT converge.")
    end

    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)

    return θ_final, num_residual_evals[] -1
end

"""
The following two solvers are using a preconditioner, which can be only used in MO guage.
One with logs and other one without logs
"""


export mo_only_preconditioned_nk_solver_with_logs

function mo_only_preconditioned_nk_solver_with_logs(
    θ::Array{Float64,4},
    residual_fun::Function,
    elt::FixedPointElements,shift;
    tol=1e-8,
    max_outer=300,
    m=5,
    verbose=true,
    η=0.1
    )

    denom = elt.denom

    safe_denom = denom .+ shift * sign.(denom)
    safe_denom[abs.(safe_denom) .< shift] .= shift
    denom_vec = vec(safe_denom)

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

    function PJv(θ_vec, r, v)
        return Jv(θ_vec, r, v) ./ denom_vec
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)

    r̃ = r ./ denom_vec
    normr = norm(r̃)

    verbose && println("NK: initial ||M⁻¹r|| = $normr")

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

        V[:,1] .= r̃ / β

        verbose && println("Newton iter $k: ||M⁻¹r|| = $normr")

        gmres_current = Float64[]
        jmax = 0

        for j = 1:m

            w = PJv(θ_vec, r, V[:,j])

            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] .* V[:,i]
            end

            H[j+1,j] = norm(w)

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
        r̃ = r ./ denom_vec
        normr = norm(r̃)

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

    verbose && println("Final ||M⁻¹r|| = $normr")
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)

    return θ_final,
           newton_residuals_pre,
           newton_residuals_post,
           gmres_residuals,
           num_residual_evals[]
end

export mo_only_preconditioned_nk_solver

function mo_only_preconditioned_nk_solver(
    θ::Array{Float64,4},
    residual_fun::Function,
    elt::FixedPointElements,shift;
    tol=1e-8,
    max_outer=300,
    m=5,
    verbose=true,
    η=0.1
    )

    denom = elt.denom

    safe_denom = denom .+ shift * sign.(denom)
    safe_denom[abs.(safe_denom) .< shift] .= shift
    denom_vec = vec(safe_denom)

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

    function PJv(θ_vec, r, v)
        return Jv(θ_vec, r, v) ./ denom_vec
    end

    θ_vec = vec(copy(θ))
    r = vec_residual(θ_vec)

    r̃ = r ./ denom_vec
    normr = norm(r̃)

    verbose && println("NK: initial ||M⁻¹r|| = $normr")

    k = 0
    total_inner = 0

    while normr > tol && k < max_outer
        β = normr
        V = zeros(n, m+1)
        H = zeros(m+1, m)
        V[:,1] .= r̃ / β
        verbose && println("Newton iter $k: ||M⁻¹r|| = $normr")
        inner = 0
        for j = 1:m
            w = PJv(θ_vec, r, V[:,j])
            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end
            H[j+1,j] = norm(w)
            if H[j+1,j] < tol
                break
            end
            V[:,j+1] .= w / H[j+1,j]
            Hj = H[1:j+1,1:j]
            e1 = zeros(j+1); e1[1] = β
            resid_inner = norm(Hj * (Hj \ e1) - e1)
            verbose && println("  GMRES $j: inner = $resid_inner")
            inner += 1
            if resid_inner <= η * β
                verbose && println("  GMRES stopping criterion satisfied")
                break
            end
        end
        total_inner += inner
        jmax = findlast(!iszero, sum(abs.(H), dims=1))[2]
        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β
        s = Hj \ e1
        Δθ = -V[:,1:jmax] * s
        θ_vec .+= Δθ
        r = vec_residual(θ_vec)
        r̃ = r ./ denom_vec
        normr = norm(r̃)

        k += 1
    end

    if normr <= tol
        verbose && println("Converged in $k Newton and $total_inner GMRES iterations.")
    else
        verbose && println("Did NOT converge.")
    end

    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)

    return θ_final, num_residual_evals[]
end