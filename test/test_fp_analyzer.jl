using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LinearMaps, Arpack

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
Cscf, mo_energies__SCF = compute_C_SCF_method(nocc, new_S, new_T, new_A, new_eris, 1000, 1e-13, 1);
mo_eris = ao2mo_eris(new_eris, Cscf);

D = compute_Density_matrix(nocc, Cscf);
f = compute_Fock_matrix(new_T, new_A, new_eris, D);

nvir = n_b - nocc;
initial_guess_mo = zeros(nocc, nocc, nvir, nvir);
fock_mo = Cscf' * f * Cscf;

t2, diffs = fixed_point_iteration(update_amps_new, initial_guess_mo, mo_eris, fock_mo, 300, 1e-13, true);
################################################################
# Initialization
purt = 1e-6; 
peris = make_physaoeris(new_eris);

# epsilon1 = 1e-8;
# run_analyzer = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, epsilon1);

# # MO case (Identity Transformation)
# T_I = Matrix{Float64}(I, n_b, n_b);
# conclusion_I, spec_radius_I = run_analyzer(T_I)

# # # Perturbation case (Identity plus little Perturbation Transformation)
# # T_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q);
# # conclusion_I_plus_P, spec_radius_I_plus_P = run_analyzer(T_I_plus_P)

# # AO case (Full Transformation)
# conclusion_F, spec_radius_F = run_analyzer(Cscf)

epsilon2 = 2;
run_analyzer_shifted = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, epsilon2);
conclusion_F_shifted, spec_radius_F_shifted = run_analyzer_shifted(Cscf)

# Sweep epsilon and plot spectral radius with specified ranges
eps_neg1_to_0 = collect(range(-1000.0, stop=0.0, length=10))
eps_0_to_1    = collect(range(0.0, stop=1.0, length=10))
eps_1_to_10   = collect(range(1.0, stop=10.0, length=10))
eps_100       = [1000.0]
eps_values    = vcat(eps_neg1_to_0, eps_0_to_1, eps_1_to_10, eps_100)

spec_radii = similar(eps_values)
for (i, eps) in enumerate(eps_values)
    run_anal = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, eps)
    _, spec_r = run_anal(Cscf)
    spec_radii[i] = spec_r
end

plt = plot(
    eps_values,
    spec_radii;
    xlabel = "epsilon",
    ylabel = "spectral radius",
    label = nothing,
    marker = :circle,
    title = "Spectral radius vs epsilon",
    ylims = (0, 2)
)
display(plt)

savefig(plt, "rho(epsilon).png")

# Initialization
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 300;
tol = 1e-16; 
peris = make_physaoeris(new_eris);

run_fixed_point = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-8


initial_guess = zeros(n_b,n_b,n_b,n_b);
max_outer = 300;
tol = 1e-9;
m = 10;
peris = make_physaoeris(new_eris);

run_nk = nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I = run_nk(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-8

# # Perturbation case (Identity plus little Perturbation Transformation)
# T_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q);
# θ_final_I_plus_P, θ_benchmark_I_plus_P = run_nk(T_I_plus_P);
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8

# # AO case (Full Transformation)
# T_F = Cscf;
# θ_final_F, θ_benchmark_F = run_nk(Cscf);
# @test norm(θ_final_F - θ_benchmark_F) < 1e-8

# @test norm(θ_final_I - θ_benchmark_I) < 1e-8
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8
# @test norm(θ_final_F - θ_benchmark_F) < 1e-8

m = 15;
peris = make_physaoeris(new_eris);

run_nk = nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I = run_nk(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-8

# # Perturbation case (Identity plus little Perturbation Transformation)
# T_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q);
# θ_final_I_plus_P, θ_benchmark_I_plus_P = run_nk(T_I_plus_P);
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8

# # AO case (Full Transformation)
# T_F = Cscf;
# θ_final_F, θ_benchmark_F = run_nk(Cscf);
# @test norm(θ_final_F - θ_benchmark_F) < 1e-8

# @test norm(θ_final_I - θ_benchmark_I) < 1e-8
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8
# @test norm(θ_final_F - θ_benchmark_F) < 1e-8

m = 20;
peris = make_physaoeris(new_eris);

run_nk = nk_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I = run_nk(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-8

# # Perturbation case (Identity plus little Perturbation Transformation)
# T_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q);
# θ_final_I_plus_P, θ_benchmark_I_plus_P = run_nk(T_I_plus_P);
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8

# # AO case (Full Transformation)
# T_F = Cscf;
# θ_final_F, θ_benchmark_F = run_nk(Cscf);
# @test norm(θ_final_F - θ_benchmark_F) < 1e-8

# @test norm(θ_final_I - θ_benchmark_I) < 1e-8
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8
# @test norm(θ_final_F - θ_benchmark_F) < 1e-8