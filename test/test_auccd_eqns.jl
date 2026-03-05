using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random

pkg_root = dirname(dirname(pathof(AUCC)));
base_dir = joinpath(pkg_root, "test/pyscf_data/LiH_sto-3g");

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
Cscf, mo_energies__SCF = compute_C_SCF_method(n_b, nocc, new_S, new_T, new_A, new_eris, 100, 1e-14, 1);
mo_eris = ao2mo_eris(new_eris, Cscf);

D = compute_Density_matrix(nocc, Cscf);
f = compute_Fock_matrix(new_T, new_A, new_eris, D);

nvir = n_b - nocc;
initial_guess_mo = zeros(nocc, nocc, nvir, nvir);
fock_mo = Cscf' * f * Cscf;

t2, diffs = fixed_point_iteration(update_amps_new, initial_guess_mo, mo_eris, fock_mo, 300, 1e-14, true);
################################################################
peris = make_physaoeris(new_eris);
run_auccd = auccd_factory(new_S, t2, nocc, n_b, Cscf, f, peris);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
Z_I = run_auccd(T_I);
@test norm(Z_I) < 1e-8

# Perturbation case (Identity plus little Perturbation Transformation)
T_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q);
Z_I_plus_P = run_auccd(T_I_plus_P);
@test norm(Z_I_plus_P) < 1e-8

# AO case (Full Transformation)
T_F = Cscf;
Z_F = run_auccd(T_F);
@test norm(Z_F) < 1e-8

@test norm(Z_I) < 1e-8
@test norm(Z_I_plus_P) < 1e-8
@test norm(Z_F) < 1e-8