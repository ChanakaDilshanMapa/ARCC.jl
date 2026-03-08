import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim, Printf

pkg_root = dirname(dirname(pathof(AUCC)));
saved_data_root = joinpath(pkg_root, "test", "saved_data");

function infer_basis_set(molecule_name::String)
    m = match(r".*_(.+)$", molecule_name)
    return m === nothing ? "unknown" : m.captures[1]
end

function infer_molecule_label(molecule_name::String)
    m = match(r"^(.*)_([^_]+)$", molecule_name)
    return m === nothing ? molecule_name : m.captures[1]
end

function choose_saved_dataset(saved_root::String, default_name::String)
    datasets = sort(filter(name -> isdir(joinpath(saved_root, name)) && !startswith(name, "."), readdir(saved_root)))
    isempty(datasets) && error("No saved datasets found in $saved_root")

    if stdin isa Base.TTY
        println("Available datasets:")
        println(rpad("  #", 6) * rpad("Molecule", 24) * "Basis set")
        println("  " * repeat("-", 44))
        for (i, name) in enumerate(datasets)
            mol_label = infer_molecule_label(name)
            basis = infer_basis_set(name)
            idx = "[$i]"
            println(rpad("  " * idx, 6) * rpad(mol_label, 24) * basis)
        end

        print("Select dataset by index or name [$default_name]: ")
        user_input = try
            strip(readline())
        catch
            ""
        end

        if isempty(user_input)
            return default_name
        end

        if all(isdigit, user_input)
            idx = parse(Int, user_input)
            if 1 <= idx <= length(datasets)
                return datasets[idx]
            end
            println("Invalid index. Falling back to default: $default_name")
            return default_name
        end

        if user_input in datasets
            return user_input
        end

        println("Dataset '$user_input' not found. Falling back to default: $default_name")
        return default_name
    end

    return default_name
end

default_molecule = "H2_cc-pVTZ"
Molecule = choose_saved_dataset(saved_data_root, default_molecule);
Tested_basis_set = infer_basis_set(Molecule);
Molecule_label = infer_molecule_label(Molecule);
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

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule);
upto_t2 = joinpath(save_dir, "mo_elements_$(Molecule).npz");
upto_t2 = npzread(upto_t2);

nocc = Int(upto_t2["nocc"][1]);
n_b = Int(upto_t2["n_b"][1]);
new_S = upto_t2["new_S"];
new_T = upto_t2["new_T"];
new_A = upto_t2["new_A"];
new_eris = upto_t2["new_eris"];
Cscf = upto_t2["Cscf"];
mo_energies__SCF = upto_t2["mo_energies__SCF"];
mo_eris = upto_t2["mo_eris"];
D = upto_t2["D"];
f = upto_t2["f"];
nvir = Int(upto_t2["nvir"][1]);
initial_guess_mo = upto_t2["initial_guess_mo"];
fock_mo = upto_t2["fock_mo"];
t2 = upto_t2["t2"];
diffs = upto_t2["diffs"];

################################################################
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 200;
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
max_outer_nk = 200;

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule);
best_shift = joinpath(save_dir, "best_shift_finder_$(Molecule).npz");
best_shift_data = npzread(best_shift);
eps_min_cont = best_shift_data["eps_min_cont"][1];
rho_min_cont = best_shift_data["rho_min_cont"][1];
shift_non_canonical = best_shift_data["shift_non_canonical"][1];

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule);
shift_analyzer = joinpath(save_dir, "shift_analyzer_$(Molecule).npz");
shift_analyzer_data = npzread(shift_analyzer);
conclusion_I = shift_analyzer_data["conclusion_I"][1];
spec_radius_I = shift_analyzer_data["spec_radius_I"][1];
conclusion_F = shift_analyzer_data["conclusion_F"][1];
spec_radius_F = shift_analyzer_data["spec_radius_F"][1];
conclusion_F_shifted = shift_analyzer_data["conclusion_F_shifted"][1];
spec_radius_F_shifted = shift_analyzer_data["spec_radius_F_shifted"][1];

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule);
fp_data_file = joinpath(save_dir, "fixed_point_results_$(Molecule).npz");
fp_saved = npzread(fp_data_file);
θ_final_I = fp_saved["theta_final_I"];
θ_benchmark_I = fp_saved["theta_benchmark_I"];
diffs_I = fp_saved["diffs_I"];
θ_final_F = fp_saved["theta_final_F"];
θ_benchmark_F = fp_saved["theta_benchmark_F"];
diffs_F = fp_saved["diffs_F"];
θ_final_F_shifted = fp_saved["theta_final_F_shifted"];
θ_benchmark_F_shifted = fp_saved["theta_benchmark_F_shifted"];
diffs_F_shifted = fp_saved["diffs_F_shifted"];

fp_err_I = norm(θ_final_I - θ_benchmark_I);
fp_err_F = norm(θ_final_F - θ_benchmark_F);
fp_err_F_shifted = norm(θ_final_F_shifted - θ_benchmark_F_shifted);

fp_time_I = haskey(fp_saved, "fp_time_I") ? fp_saved["fp_time_I"][1] : NaN;
fp_time_F = haskey(fp_saved, "fp_time_F") ? fp_saved["fp_time_F"][1] : NaN;
fp_time_F_shifted = haskey(fp_saved, "fp_time_F_shifted") ? fp_saved["fp_time_F_shifted"][1] : NaN;

fp_iters_I = haskey(fp_saved, "fp_iters_I") ? Int(fp_saved["fp_iters_I"][1]) : length(diffs_I);
fp_iters_F = haskey(fp_saved, "fp_iters_F") ? Int(fp_saved["fp_iters_F"][1]) : length(diffs_F);
fp_iters_F_shifted = haskey(fp_saved, "fp_iters_F_shifted") ? Int(fp_saved["fp_iters_F_shifted"][1]) : length(diffs_F_shifted);

fp_converged_I = fp_err_I < 1e-7;
fp_converged_F = fp_err_F < 1e-7;
fp_converged_F_shifted = fp_err_F_shifted < 1e-7;


################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule);
nk_test_data_file = joinpath(save_dir, "nk_test_results_$(Molecule).npz");
nk_test_saved = npzread(nk_test_data_file);

nk_records = NamedTuple[];

for m_val in Int.(nk_test_saved["m_values"])
    norm_err_I = nk_test_saved["m$(m_val)_norm_err_I"][1];
    is_converged_I = norm_err_I < 1e-7;
    @test is_converged_I
    num_evals_I = nk_test_saved["m$(m_val)_num_evals_I"][1];
    newton_iters_I = Int(nk_test_saved["m$(m_val)_newton_iters_I"][1]);
    gmres_iters_total_I = Int(nk_test_saved["m$(m_val)_gmres_iters_total_I"][1]);
    gmres_iters_by_newton_I = Int.(nk_test_saved["m$(m_val)_gmres_iters_by_newton_I"]);
    nk_time_I = haskey(nk_test_saved, "m$(m_val)_nk_time_I") ? nk_test_saved["m$(m_val)_nk_time_I"][1] : NaN;

    push!(nk_records, (
        m = m_val,
        residual_evals = num_evals_I,
        error = norm_err_I,
        converged = is_converged_I,
        time_s = nk_time_I,
        newton_iters = newton_iters_I,
        gmres_total = gmres_iters_total_I,
        gmres_by_newton = gmres_iters_by_newton_I,
    ));
end

################################################################

function print_summary()
    hr = "----------------------------------------------------------------"
    fmt_time(x) = isnan(x) ? "N/A" : @sprintf("%.6f s", x)
    fmt_num(x) = @sprintf("%.6e", x)

    function print_case_block(case_name, spec_radius, attractor, amp_err, time_s, converged, iters)
        status = converged ? "Converged" : "Diverged"
        iter_text = converged ? string(iters) : "-"
        println(case_name)
        println("  Spectral radius : $(fmt_num(spec_radius))")
        println("  Attractor       : $attractor")
        println("  Amplitude error : $(fmt_num(amp_err))")
        println("  Wall time       : $(fmt_time(time_s))")
        println("  Status          : $status")
        println("  Iterations      : $iter_text")
        println(hr)
    end

    println("\n========================= SUMMARY =========================")
    println("Molecule         : $Molecule_label")
    println("Tested Basis-set : $Tested_basis_set")
    println("System size      : N_basis = $n_b, N_occ = $nocc, N_vir = $nvir")
    println()
    println(hr)
    println("Optimized shift  : Shift = $(fmt_num(eps_min_cont)), Spectral radius = $(fmt_num(rho_min_cont))")
    println(hr)
    println()
    println("Fixed-Point Iteration Summary")
    println(hr)

    print_case_block(
        "Identity case",
        spec_radius_I,
        conclusion_I,
        fp_err_I,
        fp_time_I,
        fp_converged_I,
        fp_iters_I,
    )
    println()

    print_case_block(
        "AO full case",
        spec_radius_F,
        conclusion_F,
        fp_err_F,
        fp_time_F,
        fp_converged_F,
        fp_iters_F,
    )
    println()

    print_case_block(
        "AO shifted case",
        spec_radius_F_shifted,
        conclusion_F_shifted,
        fp_err_F_shifted,
        fp_time_F_shifted,
        fp_converged_F_shifted,
        fp_iters_F_shifted,
    )

    println()
    println("Newton-Krylov Solver Summary")
    println(hr)

    w_m = 24
    w_re = 14
    w_err = 16
    w_conv = 8
    w_time = 14
    w_newton = 10
    w_gmres = 8

    println(
        rpad("No. of Krylov Subspaces", w_m) *
        rpad("Res evals", w_re) *
        rpad("Amps error", w_err) *
        rpad("Conv", w_conv) *
        rpad("Wall time", w_time) *
        rpad("Newton", w_newton) *
        rpad("GMRES", w_gmres)
    )
    println(hr)
    println()

    for r in nk_records
        err_txt = @sprintf("%.3e", r.error)
        conv_txt = r.converged ? "yes" : "no"
        time_txt = isnan(r.time_s) ? "N/A" : @sprintf("%.4f s", r.time_s)
        println(
            rpad(string(r.m), w_m) *
            rpad(string(r.residual_evals), w_re) *
            rpad(err_txt, w_err) *
            rpad(conv_txt, w_conv) *
            rpad(time_txt, w_time) *
            rpad(string(r.newton_iters), w_newton) *
            rpad(string(r.gmres_total), w_gmres)
        )
        println()
        println("GMRES per Newton: $(r.gmres_by_newton)")
        println()
    end

    println("===========================================================\n")
end

print_summary();

