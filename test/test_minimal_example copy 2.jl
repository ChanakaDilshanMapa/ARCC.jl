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
################################################################
# Best shift finder
rho_of_eps = function (eps::Float64)
    run_anal = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, eps)
    _, spec_r = run_anal(Cscf)
    return spec_r
end

res_opt = Optim.optimize(rho_of_eps, 0.0, 10.0, Optim.Brent())
eps_min_cont = Optim.minimizer(res_opt)
rho_min_cont = Optim.minimum(res_opt)

shift_non_canonical = eps_min_cont

################################################################
# Spectral analyze
run_analyzer = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, shift_canonical);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
conclusion_I, spec_radius_I = run_analyzer(T_I)

# AO case (Full Transformation)
conclusion_F, spec_radius_F = run_analyzer(Cscf)

run_analyzer_shifted = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, shift_non_canonical);
conclusion_F_shifted, spec_radius_F_shifted = run_analyzer_shifted(Cscf)


################################################################
run_fixed_point = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-7

# AO case (Full Transformation)
T_F = Cscf;
θ_final_F, θ_benchmark_F, diffs_F = run_fixed_point(Cscf);
@test norm(θ_final_F - θ_benchmark_F) > 10

run_fixed_point_shifted = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical; verbose=true);
θ_final_F_shifted, θ_benchmark_F_shifted, diffs_F_shifted = run_fixed_point_shifted(Cscf);
@test norm(θ_final_F_shifted - θ_benchmark_F_shifted) < 1e-7

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
fp_data_file = joinpath(save_dir, "fixed_point_results_$(Molecule).npz")
npzwrite(fp_data_file, Dict(
    "theta_final_I" => θ_final_I,
    "theta_benchmark_I" => θ_benchmark_I,
    "diffs_I" => diffs_I,
    "theta_final_F" => θ_final_F,
    "theta_benchmark_F" => θ_benchmark_F,
    "diffs_F" => diffs_F,
    "theta_final_F_shifted" => θ_final_F_shifted,
    "theta_benchmark_F_shifted" => θ_benchmark_F_shifted,
    "diffs_F_shifted" => diffs_F_shifted
))
########################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
fp_data_file = joinpath(save_dir, "fixed_point_results_$(Molecule).npz")
fp_saved = npzread(fp_data_file)
θ_final_I = fp_saved["theta_final_I"]
θ_benchmark_I = fp_saved["theta_benchmark_I"]
diffs_I = fp_saved["diffs_I"]
θ_final_F = fp_saved["theta_final_F"]
θ_benchmark_F = fp_saved["theta_benchmark_F"]
diffs_F = fp_saved["diffs_F"]
θ_final_F_shifted = fp_saved["theta_final_F_shifted"]
θ_benchmark_F_shifted = fp_saved["theta_benchmark_F_shifted"]
diffs_F_shifted = fp_saved["diffs_F_shifted"]

########################################################################
T_I = Matrix{Float64}(I, n_b, n_b)
nk_test_data = Dict{String, Any}("m_values" => [5])
for m_val in (5)
    run_nk_logs = nk_logs_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val)
    θ_final_I_l, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, gmres_I_l, num_evals_I_l = run_nk_logs(T_I)
    θ_final_F_l, θ_benchmark_F_l, newton_pre_F_l, newton_post_F_l, gmres_F_l, num_evals_F_l = run_nk_logs(Cscf)

    nk_test_data["m$(m_val)_norm_err_I"] = [norm(θ_final_I_l - θ_benchmark_I_l)]
    nk_test_data["m$(m_val)_norm_err_F"] = [norm(θ_final_F_l - θ_benchmark_F_l)]
end

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
nk_test_data_file = joinpath(save_dir, "nk_test_results_$(Molecule).npz")
npzwrite(nk_test_data_file, nk_test_data)

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
nk_test_data_file = joinpath(save_dir, "nk_test_results_$(Molecule).npz")
nk_test_saved = npzread(nk_test_data_file)

for m_val in Int.(nk_test_saved["m_values"])
    norm_err_I = nk_test_saved["m$(m_val)_norm_err_I"][1]
    norm_err_F = nk_test_saved["m$(m_val)_norm_err_F"][1]
    @test norm_err_I < 1e-7
    @test norm_err_F < 1e-7
end

