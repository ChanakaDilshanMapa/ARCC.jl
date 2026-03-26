using Revise, IterativeSolvers 
using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(AUCC)));
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
# initial_guess = zeros(nocc, nocc, nvir, nvir);
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 200;
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
max_outer_nk = 200;


function newton_krylov_precond(
    θ::Array{Float64,4},
    residual_fun::Function,
    F::FockOperators;   
    tol = 1e-8,
    max_outer = 300,
    m = 5,
    verbose = true,
    η = 0.1
)

    θ_shape = size(θ)
    n = prod(θ_shape)
    num_residual_evals = Ref(0)
    total_gmres_solves = 0   

    function vec_residual(θ_vec::Vector{Float64})
        θ_tensor = reshape(θ_vec, θ_shape)
        num_residual_evals[] += 1
        return vec(residual_fun(θ_tensor))
    end

    function Jv(θ_vec::Vector{Float64}, r_raw::Vector{Float64}, v::Vector{Float64})
        δ = sqrt(eps(Float64)) * (1 + norm(θ_vec)) / max(norm(v), 1e-12)
        return (vec_residual(θ_vec .+ δ .* v) .- r_raw) ./ δ
    end


    function apply_P(x::Vector{Float64})
        x_tensor = reshape(x, θ_shape)
        y_prec = newfunc(F, x_tensor)
        return vec(y_prec)
    end

    # ---------------------------
    # Apply P⁻¹ J v
    # ---------------------------
    function apply_Pinv_J(θ_vec::Vector{Float64}, r_raw::Vector{Float64}, v::Vector{Float64})
        Jv_vec = Jv(θ_vec, r_raw, v)
        y = apply_P(Jv_vec)
        return y
    end

    # ---------------------------
    # Initial residual (preconditioned)
    # ---------------------------
    θ_vec = vec(copy(θ))
    rx = vec_residual(θ_vec)   # raw residual
    r = apply_P(rx)            # preconditioned RHS
    normr = norm(r)
    verbose && println("Preconditioned NK: initial ||r|| = $normr")

    k = 0
    while normr > tol && k < max_outer

        β = normr
        V = zeros(n, m+1)
        H = zeros(m+1, m)
        V[:,1] .= r / β
        verbose && println("Newton iter $k: ||r|| = $normr")

        # ---------------------------
        # Preconditioned GMRES for the main system (big system)
        # ---------------------------
        for j = 1:m
            w = apply_Pinv_J(θ_vec, rx, V[:,j])

            # Arnoldi orthogonalization
            for i = 1:j
                H[i,j] = dot(V[:,i], w)
                w .-= H[i,j] * V[:,i]
            end

            H[j+1,j] = norm(w)
            if H[j+1,j] < tol
                break
            end

            V[:,j+1] .= w / H[j+1,j]

            # Eisenstat-Walker stopping criterion
            Hj = H[1:j+1, 1:j]
            e1 = zeros(j+1); e1[1] = β
            resid_inner = norm(Hj * (Hj \ e1) - e1)
            if resid_inner <= η * β
                break
            end
        end

        # ---------------------------
        # Count the GMRES solve for this Newton step
        # ---------------------------
        total_gmres_solves += 1

        # ---------------------------
        # Compute Δθ and update
        # ---------------------------
        jmax = findlast(!iszero, sum(abs.(H), dims=1))[2]
        Hj = H[1:jmax+1, 1:jmax]
        e1 = zeros(jmax+1); e1[1] = β
        s = Hj \ (β * (e1 ./ β))
        Δθ = -V[:,1:jmax] * s

        θ_vec .+= Δθ
        rx = vec_residual(θ_vec)  # update raw residual
        r = apply_P(rx)           # preconditioned residual for next Newton step
        normr = norm(r)
        k += 1
    end

    # ---------------------------
    # Final reporting
    # ---------------------------
    if normr <= tol
        verbose && println("Converged in $k Newton steps, total GMRES solves = $total_gmres_solves, final ||r|| = $normr")
    else
        verbose && println("Did NOT converge in $k Newton steps, total GMRES solves = $total_gmres_solves, final ||r|| = $normr")
    end
    verbose && println("Total residual evaluations: $(num_residual_evals[])")

    θ_final = reshape(θ_vec, θ_shape)
    return θ_final, num_residual_evals[], total_gmres_solves
end

function pre_nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m)
    return function (T)
        Tbar = T_bar(T, new_S)
        slice = make_slices(Cscf, T, Tbar, nocc, n_b)
        proj = make_projectors(slice)
        int = make_integrals(proj, peris)
        fop = make_fock_operators(proj, f)

        coulomint = make_coulomb_integrals(int, slice)
        fd = make_fock_diags_and_offs(fop)

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

        θ_final, num_residual_evals = newton_krylov_precond(initial_guess, build_residual; 
            fop, tol=tol, max_outer=max_outer, m=m, verbose=true)
        return θ_final, θ_benchmark, num_residual_evals
    end
end

T_I = Matrix{Float64}(I, n_b, n_b)

run_nk = pre_nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5)

θ_final_I_l, θ_benchmark_I_l, num_evals_I_l = run_nk(T_I);
# θ_final_Full, θ_benchmark_Full, num_evals_Full = run_nk(Cscf);






