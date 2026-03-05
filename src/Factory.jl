export auccd_factory

function auccd_factory(new_S, t2, nocc, n_b, Cscf, f, peris)
    return function (T)        
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        coulomint = make_coulomb_integrals(int, slice)
        fop = make_fock_operators(proj, f)
        fint = make_fock_integrals(fop, int)

        jθ_fun = coulomint.jθ
        kθ_fun = coulomint.kθ

        Go_fun = fint.Go
        Gv_fun = fint.Gv

        θ_fun = theta(slice)
        θ_benchmark = θ_fun(t2)

        jθ = jθ_fun(θ_benchmark)
        kθ = kθ_fun(θ_benchmark)

        G_o = Go_fun(θ_benchmark)
        G_v = Gv_fun(θ_benchmark)

        red = au_ccd_eqns(int, jθ, kθ, G_o, G_v)

        Z = red(θ_benchmark)
        return Z
    end        
end

export fixed_point_factory

function fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,epsilon; verbose=true)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        coulomint = make_coulomb_integrals(int, slice)
        fop = make_fock_operators(proj, f)
        fd = make_fock_diags_and_offs(fop)
        elt = make_fixed_point_elements(int, coulomint, slice, fd, n_b)

        amp_fun = ao_amps(int, elt,slice,epsilon)
        t_fun = theta2mo_amp(slice)

        θ_fun = theta(slice)
        θ_benchmark = θ_fun(t2)

        θ_final, diffs = ao_fixed_point_iteration(amp_fun, initial_guess, t_fun, max_iter, tol, verbose)

        return θ_final, θ_benchmark, diffs
    end
end

export analyzer_factory

function analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt,epsilon)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        coulomint = make_coulomb_integrals(int, slice)
        fop = make_fock_operators(proj, f)
        fd = make_fock_diags_and_offs(fop)
        elt = make_fixed_point_elements(int, coulomint, slice, fd, n_b)

        amp_fun = ao_amps(int, elt,slice,epsilon)
        t_fun = theta2mo_amp(slice)
        θ_benchmark = theta(slice)(t2)

        # spectral_radius = analyze_fixed_point(amp_fun, t_fun, θ_benchmark, purt)
        spectral_radius = estimate_spectral_radius(amp_fun, t_fun, θ_benchmark)

        if spectral_radius < 1
            conclusion = "attractor"
        elseif spectral_radius > 1
            conclusion = "repulsor"
        else
            conclusion = "inconclusive"
        end
        return conclusion, spectral_radius
    end
end

export nl_factory

function nl_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, maxiter, ftol)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        fop = make_fock_operators(proj, f)

        # Get function generators
        j_fun = j_integralθ(int)
        k_fun = k_integralθ(int)
        G_o_fun = g_o(fop, int)
        G_v_fun = g_v(fop, int)

        # Build residual function
        function build_residual(θ)
            j = j_fun(θ)
            k = k_fun(θ)
            G_o = G_o_fun(θ)
            G_v = G_v_fun(θ)
            
            eqn = au_ccd_eqns(int, j, k, G_o, G_v)
            return eqn(θ)
        end

        θ_benchmark = theta(slice)(t2)

        θ_final = nl_solve(initial_guess, build_residual; maxiter, ftol)
 
        return θ_final, θ_benchmark
    end
end

export nk_factory

function nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        fop = make_fock_operators(proj, f)

        # Get function generators
        j_fun = j_integralθ(int)
        k_fun = k_integralθ(int)
        G_o_fun = g_o(fop, int)
        G_v_fun = g_v(fop, int)

        # Build residual function
        function build_residual(θ)
            j = j_fun(θ)
            k = k_fun(θ)
            G_o = G_o_fun(θ)
            G_v = G_v_fun(θ)
            
            eqn = au_ccd_eqns(int, j, k, G_o, G_v)
            return eqn(θ)
        end

        θ_benchmark = theta(slice)(t2)

        # Enable optional Jacobian hermiticity check for debugging small problems
        θ_final, num_residual_evals = newton_krylov(initial_guess, build_residual; tol=tol, max_outer=max_outer, m=m, verbose=true)
        return θ_final, θ_benchmark, num_residual_evals
    end
end

export nk_logs_factory
export nk_logs_precond_factory

function nk_logs_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        fop = make_fock_operators(proj, f)

        j_fun = j_integralθ(int)
        k_fun = k_integralθ(int)
        G_o_fun = g_o(fop, int)
        G_v_fun = g_v(fop, int)

        function build_residual(θ)
            j = j_fun(θ)
            k = k_fun(θ)
            G_o = G_o_fun(θ)
            G_v = G_v_fun(θ)
            eqn = au_ccd_eqns(int, j, k, G_o, G_v)
            return eqn(θ)
        end

        θ_benchmark = theta(slice)(t2)
        θ_final, newton_pre, newton_post, gmres_residuals, num_residual_evals = newton_krylov_with_logs(initial_guess, build_residual; tol=tol, max_outer=max_outer, m=m, verbose=true)
        return θ_final, θ_benchmark, newton_pre, newton_post, gmres_residuals, num_residual_evals
    end
end

function nk_logs_precond_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        fop = make_fock_operators(proj, f)
        fd = make_fock_diags_and_offs(fop)
        denom = ao_denominator(fd, n_b)
        epsilon = 1e-8
        safe_denom = denom .+ epsilon * sign.(denom)
        safe_denom[abs.(safe_denom) .< epsilon] .= epsilon
        precond_fun = ao_preconditioner(safe_denom)

        j_fun = j_integralθ(int)
        k_fun = k_integralθ(int)
        G_o_fun = g_o(fop, int)
        G_v_fun = g_v(fop, int)

        function build_residual(θ)
            j = j_fun(θ)
            k = k_fun(θ)
            G_o = G_o_fun(θ)
            G_v = G_v_fun(θ)
            eqn = au_ccd_eqns(int, j, k, G_o, G_v)
            return eqn(θ)
        end

        θ_benchmark = theta(slice)(t2)
        θ_final, num_residual_evals = newton_krylov_with_logs_preconditioned(
            initial_guess,
            build_residual,
            precond_fun;
            tol=tol,
            max_outer=max_outer,
            m=m,
            verbose=true
        )
        return θ_final, θ_benchmark, num_residual_evals
    end
end

export bb_runner

function bb_runner(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, α, tol)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        fop = make_fock_operators(proj, f)

        # Get function generators
        j_fun = j_integralθ(int)
        k_fun = k_integralθ(int)
        G_o_fun = g_o(fop, int)
        G_v_fun = g_v(fop, int)

        # Build residual function
        function build_residual(θ)
            j = j_fun(θ)
            k = k_fun(θ)
            G_o = G_o_fun(θ)
            G_v = G_v_fun(θ)
            
            eqn = au_ccd_eqns(int, j, k, G_o, G_v)
            return eqn(θ)
        end

        θ_benchmark = theta(slice)(t2)

        θ_final = gradient_descent_bb(initial_guess, build_residual; tol, max_iter, verbose=true)
        
        return θ_final, θ_benchmark
    end
end
