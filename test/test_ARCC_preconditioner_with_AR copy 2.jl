using Revise, IterativeSolvers, LinearOperators 
using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "LiH_sto3g";
base_dir = joinpath(pkg_root, "test/pyscf_data", Molecule);

files = Dict(
    "n_occ_pyscf"          => glob("nocc*.npy", base_dir)[1],
    "S_pyscf"              => glob("overlap_matrix*.npy", base_dir)[1],
    "C_pyscf"              => glob("MO_coefficients*.npy", base_dir)[1],    
    "fock_ao_pyscf"        => glob("Fock_matrix*.npy", base_dir)[1],    
    "mo_eris_pyscf"        => glob("MO_ERIs*.npy", base_dir)[1],   
    "amp_ccd_pyscf"        => glob("t2_updated*.npy", base_dir)[1],
    "T_pyscf"              => glob("kinetic_energy_matrix*.npy", base_dir)[1],
    "V_en_pyscf"           => glob("nuclear_potential_matrix*.npy", base_dir)[1],    
    "ao_eris_pyscf"        => glob("ERI*.npy", base_dir)[1],
    "corr_ene_pyscf"       => glob("ccd_corr_energy*.npy", base_dir)[1],
);

T               = npzread(files["T_pyscf"]);
A               = npzread(files["V_en_pyscf"]);
eris            = npzread(files["ao_eris_pyscf"]);
nocc            = npzread(files["n_occ_pyscf"]); 
S               = npzread(files["S_pyscf"]);
C_pyscf         = npzread(files["C_pyscf"]);
fock_ao_pyscf   = npzread(files["fock_ao_pyscf"]);
mo_eris_pyscf   = npzread(files["mo_eris_pyscf"]);
corr_ene_pyscf  = npzread(files["corr_ene_pyscf"]);

n_b = size(S, 1);
new_S, new_T, new_A, new_eris = orthogonalize(S, T, A, eris);
Cscf, mo_energies__SCF = compute_C_SCF_method(nocc, new_S, new_T, new_A, new_eris, 1000, 1e-13, 1);
mo_eris = ao2mo_eris(new_eris, Cscf);

D = compute_Density_matrix(nocc, Cscf);
f = compute_Fock_matrix(new_T, new_A, new_eris, D);

nvir = n_b - nocc;
initial_guess_mo = zeros(nocc, nocc, nvir, nvir);
fock_mo = Cscf' * f * Cscf;

t2, diffs = fixed_point_iteration(update_amps_new, initial_guess_mo, mo_eris, fock_mo, 300, 1e-13, true);

################################################################
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 200;
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
max_outer_nk = 200;


function P_inv_action(F::FockOperators, θ_shape,y)
    Fo = F.Fo
    Fv = F.Fv
    y4d = reshape(y, θ_shape)
    @tensor Py[μ,ν,λ,σ] := Fv[α,λ] * y4d[μ,ν,α,σ] + Fv[α,σ] * y4d[μ,ν,λ,α] - Fo[μ,α] * y4d[α,ν,λ,σ] - Fo[ν,α] * y4d[μ,α,λ,σ]
    
    return vec(Py)
end

function inexact_newton(
    θ::Array{Float64,4},
    residual_fun::Function,
    fop;
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


    normr = norm(p_inv_r)

    verbose && println("NK: initial ||r|| = $normr")

    newton_residuals_pre  = Float64[]
    newton_residuals_post = Float64[]
    gmres_residuals       = Vector{Vector{Float64}}()

    push!(newton_residuals_pre, normr)

    k = 0
    total_inner = 0

    while normr > tol && k < max_outer

        verbose && println("Newton iter $k: ||r|| = $normr")

        function preconditioned_Jv(v)
            Jv_val = Jv(θ_vec, r, v)

            P_inv_Jv_val, _, _ = gmres_logged(
                y -> P_inv_action(fop, θ_shape, y),
                Jv_val;
                m=m,
                tol=tol,
                η=1.0
            )
            return P_inv_Jv_val
        end

        Δθ, gmres_current, jmax_newton = gmres_logged(
            preconditioned_Jv,
            -p_inv_r;
            m=m,
            tol=tol,
            η=η,
            verbose=verbose
        )

        push!(gmres_residuals, gmres_current)
        total_inner += jmax_newton

        θ_vec .+= Δθ

        r = vec_residual(θ_vec)

        p_inv_r, _, _ = gmres_logged(
            y -> P_inv_action(fop, θ_shape, y),
            r;
            m=m,
            tol=tol,
            η=1.0        
        )

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

    verbose && println("Final ||r|| = $normr")
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)

    return θ_final,
           newton_residuals_pre,
           newton_residuals_post,
           gmres_residuals,
           num_residual_evals[]
end


function pre_nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris,
                        initial_guess, max_outer, tol, m)

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

            eqn = ar_ccd_eqns(int, j, k, G_o, G_v)
            return eqn(θ)
        end

        θ_benchmark = theta(slice)(t2)

        θ_final, num_residual_evals = Precond_newton_krylov(
            initial_guess,
            build_residual,
            fop;
            tol=tol,
            max_outer=max_outer,
            m=m,
            verbose=true
        )

        return θ_final, θ_benchmark, num_residual_evals
    end
end

T_I = Matrix{Float64}(I, n_b, n_b)

run_nk = pre_nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5)

θ_final_I_l, θ_benchmark_I_l, num_evals_I_l = run_nk(T_I);
θ_final_Full, θ_benchmark_Full, num_evals_Full = run_nk(Cscf);
@test norm(θ_final_I_l - θ_benchmark_I_l)  < 1e-7
################################################################
shift_canonical = 1e-8;
run_fixed_point = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);
θ_final_fp, θ_benchmark_fp, diffs_fp = run_fixed_point(T_I);
@test norm(θ_final_fp - θ_benchmark_fp)  < 1e-7




