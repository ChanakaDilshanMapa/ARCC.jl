using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "C2H6_6-31g";
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


T_I = Matrix{Float64}(I, n_b, n_b)
nk_test_data = Dict{String, Any}("m_values" => [10])
m_values = nk_test_data["m_values"]

m_diis = 4

solver_configs = [
    ("nk", (m_val) -> nk_solver_factory_with_logs(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val)),
    ("nk_diis", (m_val) -> nk_solver_factory_with_diis_logs(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val, m_diis)),
    ("pnk", (m_val) -> preconditioned_nk_solver_factory_with_logs(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val)),
    ("pnk_diis", (m_val) -> preconditioned_nk_solver_factory_with_diis_logs(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val, m_diis)),
]

final_summaries = Dict{String, NamedTuple{(:newton, :gmres, :final_residual, :num_evals, :res_label), Tuple{Int, Int, Float64, Int, String}}}()

for m_val in m_values
    for (solver_tag, factory_builder) in solver_configs
        run_nk_logs = factory_builder(m_val)

        θ_final_I_l, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, gmres_I_l, num_evals_I_l = run_nk_logs(T_I)
        θ_final_F_l, θ_benchmark_F_l, newton_pre_F_l, newton_post_F_l, gmres_F_l, num_evals_F_l = run_nk_logs(Cscf)

        nk_test_data["$(solver_tag)_m$(m_val)_norm_err_I"] = [norm(θ_final_I_l - θ_benchmark_I_l)]
        nk_test_data["$(solver_tag)_m$(m_val)_norm_err_F"] = [norm(θ_final_F_l - θ_benchmark_F_l)]

        res_label = startswith(solver_tag, "pnk") ? "||P⁻¹r||" : "||r||"

        final_summaries["$(solver_tag)_m$(m_val)_I"] = (
            newton=length(newton_post_I_l),
            gmres=sum(length.(gmres_I_l)),
            final_residual=isempty(newton_post_I_l) ? NaN : newton_post_I_l[end],
            num_evals=num_evals_I_l,
            res_label=res_label,
        )

        final_summaries["$(solver_tag)_m$(m_val)_F"] = (
            newton=length(newton_post_F_l),
            gmres=sum(length.(gmres_F_l)),
            final_residual=isempty(newton_post_F_l) ? NaN : newton_post_F_l[end],
            num_evals=num_evals_F_l,
            res_label=res_label,
        )
    end
end

println("\n================ Final Solver Summary ================")
for m_val in m_values
    for (solver_tag, _) in solver_configs
        for case_tag in ("I", "F")
            key = "$(solver_tag)_m$(m_val)_$(case_tag)"
            s = final_summaries[key]
            start_label = case_tag == "I" ? "MOCC" : "AOCC"
            println("------------------------------------------------------")
            println("Case: $(solver_tag) | start=$(start_label) | m=$(m_val) | m_diis=$(m_diis)")
            println("Converged in $(s.newton) Newton and $(s.gmres) GMRES iterations.")
            println("Total residual evaluations: $(s.num_evals)")
        end
    end
end


