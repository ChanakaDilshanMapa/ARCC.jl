using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings

pkg_root = dirname(dirname(pathof(AUCC)));
base_dir = joinpath(pkg_root, "test/pyscf_data/H2O_Water_cc-pvtz");

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
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 300;
tol = 1e-13; 
peris = make_physaoeris(new_eris);

epsilon1 = 1e-8

run_fixed_point = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,epsilon1; verbose=true);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-8

# # # Perturbation case (Identity plus little Perturbation Transformation)
# # T_I_plus_P = Matrix(qr(Matrix(I,n_b,n_b) + 1e-2*ones(n_b,n_b)).Q);
# # θ_final_I_plus_P, θ_benchmark_I_plus_P, diffs_I_plus_P = run_fixed_point(T_I_plus_P);
# # @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8

# AO case (Full Transformation)
T_F = Cscf;
θ_final_F, θ_benchmark_F, diffs_F = run_fixed_point(Cscf);
@test norm(θ_final_F - θ_benchmark_F) > 1e50

epsilon2 = 2

run_fixed_point_shifted = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,epsilon2; verbose=true);

θ_final_F_shifted, θ_benchmark_F_shifted, diffs_F_shifted = run_fixed_point_shifted(Cscf);
@test norm(θ_final_F_shifted - θ_benchmark_F_shifted) < 1e-8


###########




##############

# @test norm(θ_final_I - θ_benchmark_I) < 1e-8
# @test norm(θ_final_I_plus_P - θ_benchmark_I_plus_P) < 1e-8
# @test norm(θ_final_F - θ_benchmark_F) >1e100

haslatex = (Sys.which("pdflatex") !== nothing) || (Sys.which("xelatex") !== nothing) || (Sys.which("lualatex") !== nothing)
if haslatex
    pgfplotsx()
else
    gr()
end

default(
    fontfamily = "Computer Modern",
    titlefont  = font(14, "Computer Modern"),
    guidefont  = font(12, "Computer Modern"),
    tickfont   = font(12, "Computer Modern"),
    legendfont = font(11, "Computer Modern"),
    linewidth  = 2
)

if haslatex
    xlab = L"$\\text{Iterations}$"
    ylab = L"$\\text{Residual norm}$"
    ttl  = L"$\\text{Fixed-Point Convergence}$"
    lbl_mo = L"$\\text{MO basis}$"
    lbl_ao = L"$\\text{AO basis}$"
    lbl_shift = L"$\\text{Shifted AO basis}$"
else
    xlab = L"Iterations"
    ylab = L"Residual norm"
    ttl  = L"Fixed-Point Convergence"
    lbl_mo = L"MO basis"
    lbl_ao = L"AO basis"
    lbl_shift = L"Shifted AO basis"
end

p1 = plot(
    xlabel = xlab,
    ylabel = ylab,
    title = ttl,
    yscale = :log10,
    legend = :topright,
    linewidth = 3,
    grid = true,
    size = (800, 600),
    ylims = (1e-15, 1e15),
    yticks = 10.0 .^ (-15:10:15),
    top_margin = 5 * Plots.mm,
    bottom_margin = 5 * Plots.mm,
    left_margin = 5 * Plots.mm,
    right_margin = 5 * Plots.mm
)

start_idx = 2

plot!(p1, start_idx:length(diffs_I), diffs_I[start_idx:end],
    label = lbl_mo,
      color = :blue, linestyle = :dash, linewidth = 3)

plot!(p1, start_idx:length(diffs_F), diffs_F[start_idx:end],
    label = lbl_ao,
      color = :red, linestyle = :dash, linewidth = 3)

plot!(p1, start_idx:length(diffs_F_shifted), diffs_F_shifted[start_idx:end],
    label = lbl_shift,
      color = :orange, linestyle = :dash, linewidth = 3)

display(p1)
savefig(p1, "fixed_point_convergence.png")

