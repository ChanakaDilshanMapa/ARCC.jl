using LinearAlgebra
export fp_iteration_diis
function fp_iteration_diis(amp_fun, initial_guess, t_fun;
    max_iter=100, tol=1e-8, m=6, verbose=false)

    θ = copy(initial_guess)
    converged = false
    diffs = Float64[]

    θ_list = Vector{typeof(vec(θ))}()
    r_list = Vector{typeof(vec(θ))}()

    total_iter = 0

    for outer_iter in 1:max_iter
        for inner_iter in 1:m
            t = t_fun(θ)
            θ_new = amp_fun(t)
            total_iter += 1

            if !all(isfinite.(θ_new))
                if verbose
                    println("Diverged (NaN detected) after $total_iter iterations")
                end
                return θ, converged, diffs, total_iter
            end

            diff = norm(θ_new - θ)
            push!(diffs, diff)

            if diff > 100
                if verbose
                    println("Diverged (Δ > 100) after $total_iter iterations")
                end
                return θ, converged, diffs, total_iter
            end

            if verbose
                println("Iteration $total_iter: Δ = ", diff)
            end

            push!(θ_list, vec(θ))
            push!(r_list, vec(θ_new - θ))

            θ = θ_new

            if diff < tol
                converged = true
                if verbose
                    println("Convergence achieved after $total_iter iterations")
                end
                return θ, converged, diffs, total_iter
            end
        end

        k = min(length(r_list), m)
        r_sub = r_list[end-k+1:end]
        θ_sub = θ_list[end-k+1:end]

        B = zeros(Float64, k+1, k+1)
        for i in 1:k
            for j in 1:k
                B[i,j] = dot(r_sub[i], r_sub[j])
            end
            B[i,k+1] = -1.0
            B[k+1,i] = -1.0
        end
        B[k+1,k+1] = 0.0

        rhs = zeros(Float64, k+1)
        rhs[k+1] = -1.0

        coeffs = B \ rhs
        c = coeffs[1:k]

        θ_diis = zero(vec(θ))
        for i in 1:k
            θ_diis += c[i] * θ_sub[i]
        end

        θ_new = reshape(θ_diis, size(θ))
        diff_diis = norm(θ_new - θ)
        push!(diffs, diff_diis)

        if verbose
            println("DIIS update: Δ = ", diff_diis)
        end

        if !all(isfinite.(θ_new))
            if verbose
                println("Diverged (NaN detected) after $total_iter iterations (DIIS)")
            end
            return θ, converged, diffs, total_iter
        end

        if diff_diis < tol
            converged = true
            if verbose
                println("Convergence achieved after $total_iter iterations (DIIS)")
            end
            return θ_new, converged, diffs, total_iter
        end

        θ = θ_new
    end

    if !converged
        verbose && println("Warning: Failed to converge after $total_iter iterations")
    end

    return θ, converged, diffs, total_iter
end

export fp_iteration_factory_diis
function fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol, shift, m; verbose=true)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        coulomint = make_coulomb_integrals(int, slice)
        fop = make_fock_operators(proj, f)
        fd = make_fock_diags_and_offs(fop)
        elt = make_fixed_point_elements(int, coulomint, slice, fd, n_b)

        amp_fun = ao_amps(int, elt,slice,shift)
        t_fun = theta2mo_amp(slice)

        θ_fun = theta(slice)

        θ_benchmark = θ_fun(t2)

        θ_final, diffs = fp_iteration_diis(amp_fun, initial_guess, t_fun; max_iter=max_iter, tol=tol, m=m, verbose=verbose)

        return θ_final, θ_benchmark, diffs
    end
end