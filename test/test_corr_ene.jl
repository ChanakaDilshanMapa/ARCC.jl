using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random

pkg_root = dirname(dirname(pathof(ARCC)));
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

n_b = size(S, 1);
new_S, new_T, new_A, new_eris = orthogonalize(S, T, A, eris);
Cscf, mo_energies__SCF = compute_C_SCF_method(n_b, nocc, new_S, new_T, new_A, new_eris, 100, 1e-14, 1);
mo_eris = ao2mo_eris(new_eris, Cscf);

D = compute_Density_matrix(nocc, Cscf);
f = compute_Fock_matrix(new_T, new_A, new_eris, D);

nvir = n_b - nocc;
initial_guess_mo = zeros(nocc, nocc, nvir, nvir);
fock_mo = Cscf' * f * Cscf;

t2, diffs = fixed_point_iteration(update_amps_new, initial_guess_mo, mo_eris, fock_mo, 300, 1e-18);

################################################################
# Initialization
initial_guess = zeros(n_b,n_b,n_b,n_b)
max_iter = 300
tol = 1e-16 
peris = make_physaoeris(new_eris)

################################################################

# MO case (Identity Transformation)
TM_I = Matrix{Float64}(I, n_b, n_b)
θ_benchmark_I = θ_1234(TM_I, new_S, t2, nocc, n_b)

Tbar_I = T_bar(TM_I, new_S)
slice_I =  make_slices(Cscf, TM_I, Tbar_I, nocc, n_b)
proj_I = make_projectors(slice_I)
int_I = make_integrals(proj_I, peris)

corr_e_I = corr_ene(int_I, θ_benchmark_I)
# corr_e_I = corr_ene(int_I, θ_benchmark_I)
################################################################
# Perturbation case (Identity plus little Perturbation Transformation)
TM_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q)
θ_benchmark_I_plus_P = θ_1234(TM_I_plus_P, new_S, t2, nocc, n_b)

Tbar_I_plus_P = T_bar(TM_I_plus_P, new_S)
slice_I_plus_P =  make_slices(Cscf, TM_I_plus_P, Tbar_I_plus_P, nocc, n_b)
proj_I_plus_P = make_projectors(slice_I_plus_P)
int_I_plus_P = make_integrals(proj_I_plus_P, peris)

corr_e_I_plus_P = corr_ene(int_I_plus_P, θ_benchmark_I_plus_P)
################################################################
# AO case (Full Transformation)
TM_F = Cscf
θ_benchmark_F = θ_1234(TM_F, new_S, t2, nocc, n_b)

Tbar_F = T_bar(TM_F, new_S)
slice_F =  make_slices(Cscf, Tbar_F, Tbar_F, nocc, n_b)
proj_F = make_projectors(slice_F)
int_F = make_integrals(proj_F, peris)

corr_e_F = corr_ene(int_F, θ_benchmark_F)

################################################################
@test norm(corr_e_I - corr_ene_pyscf) < 1e-8
@test norm(corr_e_I_plus_P - corr_ene_pyscf) < 1e-8
@test norm(corr_e_F - corr_ene_pyscf) < 1e-8


