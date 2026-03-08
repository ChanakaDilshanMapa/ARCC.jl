using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(AUCC)));
Molecule = "C4H10_4-31g";
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


save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
upto_t2 = joinpath(save_dir, "mo_elements_$(Molecule).npz")
npzwrite(upto_t2, Dict(
    "t2" => t2,
    "diffs" => diffs,
    "nocc" => [nocc],
    "n_b" => [n_b],
    "new_S" => new_S,
    "new_T" => new_T,
    "new_A" => new_A,
    "new_eris" => new_eris,
    "Cscf" => Cscf,
    "mo_energies__SCF" => mo_energies__SCF,
    "mo_eris" => mo_eris,
    "D" => D,
    "f" => f,
    "nvir" => [nvir],
    "initial_guess_mo" => initial_guess_mo,
    "fock_mo" => fock_mo
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
upto_t2 = joinpath(save_dir, "mo_elements_$(Molecule).npz")
upto_t2 = npzread(upto_t2)

nocc = Int(upto_t2["nocc"][1])
n_b = Int(upto_t2["n_b"][1])
new_S = upto_t2["new_S"]
new_T = upto_t2["new_T"]
new_A = upto_t2["new_A"]
new_eris = upto_t2["new_eris"]
Cscf = upto_t2["Cscf"]
mo_energies__SCF = upto_t2["mo_energies__SCF"]
mo_eris = upto_t2["mo_eris"]
D = upto_t2["D"]
f = upto_t2["f"]
nvir = Int(upto_t2["nvir"][1])
initial_guess_mo = upto_t2["initial_guess_mo"]
fock_mo = upto_t2["fock_mo"]
t2 = upto_t2["t2"]
diffs = upto_t2["diffs"]

################################################################
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 400;
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

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
best_shift = joinpath(save_dir, "best_shift_finder_$(Molecule).npz")
npzwrite(best_shift, Dict(
    "eps_min_cont" => [eps_min_cont],
    "rho_min_cont" => [rho_min_cont],
    "shift_non_canonical" => [shift_non_canonical]
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
best_shift = joinpath(save_dir, "best_shift_finder_$(Molecule).npz")
best_shift_data = npzread(best_shift)
eps_min_cont = best_shift_data["eps_min_cont"][1]
rho_min_cont = best_shift_data["rho_min_cont"][1]
shift_non_canonical = best_shift_data["shift_non_canonical"][1]

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

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
shift_analyzer = joinpath(save_dir, "shift_analyzer_$(Molecule).npz")
npzwrite(shift_analyzer, Dict(
    "conclusion_I" => [conclusion_I == "attractor"],
    "spec_radius_I" => [spec_radius_I],
    "conclusion_F" => [conclusion_F == "attractor"],
    "spec_radius_F" => [spec_radius_F],
    "conclusion_F_shifted" => [conclusion_F_shifted == "attractor"],
    "spec_radius_F_shifted" => [spec_radius_F_shifted]
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
shift_analyzer = joinpath(save_dir, "shift_analyzer_$(Molecule).npz")
shift_analyzer_data = npzread(shift_analyzer)
conclusion_I = shift_analyzer_data["conclusion_I"][1]
spec_radius_I = shift_analyzer_data["spec_radius_I"][1]
conclusion_F = shift_analyzer_data["conclusion_F"][1]
spec_radius_F = shift_analyzer_data["spec_radius_F"][1]
conclusion_F_shifted = shift_analyzer_data["conclusion_F_shifted"][1]
spec_radius_F_shifted = shift_analyzer_data["spec_radius_F_shifted"][1]

################################################################
run_fixed_point = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
fp_time_I = @elapsed begin
    global θ_final_I, θ_benchmark_I, diffs_I
    θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I)
end
@test norm(θ_final_I - θ_benchmark_I) < 1e-7

# AO case (Full Transformation)
T_F = Cscf;
fp_time_F = @elapsed begin
    global θ_final_F, θ_benchmark_F, diffs_F
    θ_final_F, θ_benchmark_F, diffs_F = run_fixed_point(Cscf)
end
@test norm(θ_final_F - θ_benchmark_F) > 10

run_fixed_point_shifted = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical; verbose=true);
fp_time_F_shifted = @elapsed begin
    global θ_final_F_shifted, θ_benchmark_F_shifted, diffs_F_shifted
    θ_final_F_shifted, θ_benchmark_F_shifted, diffs_F_shifted = run_fixed_point_shifted(Cscf)
end
@test norm(θ_final_F_shifted - θ_benchmark_F_shifted) < 1e-7

fp_iters_I = length(diffs_I)
fp_iters_F = length(diffs_F)
fp_iters_F_shifted = length(diffs_F_shifted)

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
fp_data_file = joinpath(save_dir, "fixed_point_results_$(Molecule).npz")
npzwrite(fp_data_file, Dict(
    "theta_final_I" => θ_final_I,
    "theta_benchmark_I" => θ_benchmark_I,
    "diffs_I" => diffs_I,
    "fp_time_I" => [fp_time_I],
    "fp_iters_I" => [fp_iters_I],
    "theta_final_F" => θ_final_F,
    "theta_benchmark_F" => θ_benchmark_F,
    "diffs_F" => diffs_F,
    "fp_time_F" => [fp_time_F],
    "fp_iters_F" => [fp_iters_F],
    "theta_final_F_shifted" => θ_final_F_shifted,
    "theta_benchmark_F_shifted" => θ_benchmark_F_shifted,
    "diffs_F_shifted" => diffs_F_shifted,
    "fp_time_F_shifted" => [fp_time_F_shifted],
    "fp_iters_F_shifted" => [fp_iters_F_shifted]
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
nk_test_data = Dict{String, Any}("m_values" => [3, 5, 10])
for m_val in (3, 5, 10)
    run_nk_logs = nk_logs_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val)
    nk_time_I = @elapsed begin
        global θ_final_I_l, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, gmres_I_l, num_evals_I_l
        θ_final_I_l, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, gmres_I_l, num_evals_I_l = run_nk_logs(T_I)
    end

    nk_test_data["m$(m_val)_norm_err_I"] = [norm(θ_final_I_l - θ_benchmark_I_l)]

    nk_test_data["m$(m_val)_num_evals_I"] = [num_evals_I_l]

    newton_iters_I = length(newton_post_I_l)
    gmres_iters_by_newton_I = Int[length(g) for g in gmres_I_l]
    gmres_iters_total_I = sum(gmres_iters_by_newton_I)

    nk_test_data["m$(m_val)_newton_iters_I"] = [newton_iters_I]
    nk_test_data["m$(m_val)_gmres_iters_total_I"] = [gmres_iters_total_I]
    nk_test_data["m$(m_val)_gmres_iters_by_newton_I"] = gmres_iters_by_newton_I
    nk_test_data["m$(m_val)_nk_time_I"] = [nk_time_I]
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
    is_converged_I = norm_err_I < 1e-7
    @test is_converged_I
    num_evals_I = nk_test_saved["m$(m_val)_num_evals_I"][1]
    newton_iters_I = Int(nk_test_saved["m$(m_val)_newton_iters_I"][1])
    gmres_iters_total_I = Int(nk_test_saved["m$(m_val)_gmres_iters_total_I"][1])
    gmres_iters_by_newton_I = Int.(nk_test_saved["m$(m_val)_gmres_iters_by_newton_I"])
    nk_time_I = nk_test_saved["m$(m_val)_nk_time_I"][1]
   
    println("m=$m_val:")
    println("  Identity: residual evals=$num_evals_I, error=$(norm_err_I), converged=$(is_converged_I)")
    println("  Newton iterations (I): $(newton_iters_I)")
    println("  GMRES total iterations (I): $(gmres_iters_total_I)")
    println("  GMRES iterations per Newton (I): $(gmres_iters_by_newton_I)")
    println("  NK wall-time (I): $(nk_time_I) s")
end

