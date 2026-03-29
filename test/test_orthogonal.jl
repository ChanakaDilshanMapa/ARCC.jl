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

t2, diffs = fixed_point_iteration(update_amps_new, initial_guess_mo, mo_eris, fock_mo, 300, 1e-18);

TM = Matrix{Float64}(I, n_b, n_b)

####################################################################################

z_mo_orthogonal = mo_ccd_eqns(Cscf, new_S, t2, new_eris, nocc, n_b, f)
z_ao_orthogonal = ao_ccd_eqns(TM, Cscf, new_S, t2, new_eris, nocc, n_b, f)

@test norm(z_ao_orthogonal) < 1e-14
@test norm(z_mo_orthogonal) < 1e-14

t2new = rand(2,2,4,4)

z_mo_orthogonal_random = mo_ccd_eqns(Cscf, new_S, t2new, new_eris, nocc, n_b, f)
z_ao_orthogonal_random = ao_ccd_eqns(TM, Cscf, new_S, t2new, new_eris, nocc, n_b, f)
z_ao_orthogonal_random_truncated = z_ao_orthogonal_random[1:nocc, 1:nocc, nocc+1:end, nocc+1:end]

@test norm(z_ao_orthogonal_random_truncated - z_mo_orthogonal_random) < 1e-14

############################################
# Identity Case (MO case)

initial_guessnl = zeros(6,6,6,6)

Final_θ_nl_MO = solve_for_theta(TM, Cscf, new_S, new_eris, nocc, n_b, f, 
                            initial_guessnl=initial_guessnl, 
                            maxiter=300, ftol=1e-16)

Final_θ_nl_MO_truncated = Final_θ_nl_MO[1:nocc, 1:nocc, nocc+1:end, nocc+1:end]
@test norm(Final_θ_nl_MO_truncated - t2) < 1e-14

##################################################
# Cscf case (AO case)

TMC = Cscf
initial_guessnl_c = zeros(6,6,6,6)

Final_θ_nl_AO = solve_for_theta(TMC, Cscf, new_S, new_eris, nocc, n_b, f, 
                            initial_guessnl=initial_guessnl_c, 
                            maxiter=1500, ftol=1e-16)

θ1234_AO = θ_1234(TMC, new_S, t2, nocc, n_b)                            
@test norm(Final_θ_nl_AO - θ1234_AO) < 1e-14

###########################################
# Random Unitary Matrix case

function simple_random_unitary(n::Int)
    A = randn(Float64, n, n)
    return Matrix(qr(A).Q)
end

Random_unitary_mat = simple_random_unitary(6)
initial_guessnl = zeros(6,6,6,6)

Final_θ_nl_Random_unitary_mat = solve_for_theta(Random_unitary_mat, Cscf, new_S, new_eris, nocc, n_b, f, 
                            initial_guessnl=initial_guessnl, 
                            maxiter=300, ftol=1e-16)

θ1234_from_Random_unitary_mat = θ_1234(Random_unitary_mat, new_S, t2, nocc, n_b) 

@test norm(θ1234_from_Random_unitary_mat - Final_θ_nl_Random_unitary_mat) < 1e-14
############################################
# Correlation energy checks

function corr_ene(piint, theta)
    @einsum ene := (2 * piint[lambda,sigma,mu,nu] - piint[sigma,lambda,mu,nu]) * theta[mu,nu,lambda,sigma]
    return ene    
end

function piintegrals_running(TM, C, ao_eris, nocc, n_basis)
    P = ao_projector_with_TM_4_pi(TM, C, nocc)
    Q = complementary_projector_with_TM_4_pi(TM,C,nocc,n_basis)     

    v = permutedims(ao_eris, (1, 3, 2, 4))

    @einsum piint[lambda,sigma,mu,nu] := 
    P[alpha, mu] * P[beta, nu] * 
    v[gamma,delta,alpha,beta] *
    Q[lambda,gamma] * Q[sigma,delta] 

    return piint
    
end
piint_identity   = piintegrals_running(TM, Cscf, new_eris, nocc, n_b)
piint_Cscf   = piintegrals_running(TMC, Cscf, new_eris, nocc, n_b)
piint_random_unitary   = piintegrals_running(Random_unitary_mat, Cscf, new_eris, nocc, n_b)

corr_ene_from_identity      = corr_ene(piint_identity, Final_θ_nl_MO)
corr_ene_from_Cscf           = corr_ene(piint_Cscf, Final_θ_nl_AO)
corr_ene_from_random_unitary = corr_ene(piint_random_unitary, Final_θ_nl_Random_unitary_mat)

corr_ene_known = corr_ene_check2(Cscf, new_S, t2, new_eris, nocc, n_b)

@test norm(corr_ene_from_identity - corr_ene_known) < 1e-15
@test norm(corr_ene_from_Cscf - corr_ene_known) < 1e-15
@test norm(corr_ene_from_random_unitary - corr_ene_known) < 1e-15

@test norm(θ1234_from_Random_unitary_mat - Final_θ_nl_Random_unitary_mat) < 1e-14
@test norm(Final_θ_nl_AO - θ1234_AO) < 1e-14
@test norm(Final_θ_nl_MO_truncated - t2) < 1e-14



