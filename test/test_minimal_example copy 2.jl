using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "H2-CF_cc-pvtz";
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
tol = 1e-14; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
max_outer_nk = 200;


# ################################################################
# run_fixed_point = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);

# # MO case (Identity Transformation)
# T_I = Matrix{Float64}(I, n_b, n_b);
# θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I);
# @test norm(θ_final_I - θ_benchmark_I) < 1e-7

################################################################

run_fixed_point_diis = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical, 5; verbose=true);

T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point_diis(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-7

Tbar_I = T_bar(T_I, new_S)
slice_I = make_slices(Cscf, T_I, Tbar_I, nocc, n_b)
t2 = theta2mo_amp(slice_I)(θ_final_I)

run_fixed_point_diis = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical, 5; verbose=true);

T_F = Cscf;
θ_final_F, θ_benchmark_F, diffs_F = run_fixed_point_diis(Cscf);



