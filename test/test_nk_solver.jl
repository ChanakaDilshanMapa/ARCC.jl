using Test, GTO, WTP, LinearAlgebra, AUCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, Printf

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
Cscf, mo_energies__SCF = compute_C_SCF_method(nocc, new_S, new_T, new_A, new_eris, 500, 1e-13, 1);
mo_eris = ao2mo_eris(new_eris, Cscf);

D = compute_Density_matrix(nocc, Cscf);
f = compute_Fock_matrix(new_T, new_A, new_eris, D);

nvir = n_b - nocc;
initial_guess_mo = zeros(nocc, nocc, nvir, nvir);
fock_mo = Cscf' * f * Cscf;

t2, diffs = fixed_point_iteration(update_amps_new, initial_guess_mo, mo_eris, fock_mo, 300, 1e-14, true);
################################################################

# Convergence plotting for single m value (m = 15) with separate images
initial_guess = zeros(n_b,n_b,n_b,n_b)
max_outer = 300
tol = 1e-8
m = 10

peris = make_physaoeris(new_eris);
run_nk_logs = nk_logs_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer, tol, m);

# MO (identity) transformation
T_I = Matrix{Float64}(I, n_b, n_b)
θ_final_I, θ_benchmark_I, newton_pre_I, newton_post_I, gmres_I = run_nk_logs(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-8

# AO (full) transformation
θ_final_F, θ_benchmark_F, newton_pre_F, newton_post_F, gmres_F = run_nk_logs(Cscf);
@test norm(θ_final_F - θ_benchmark_F) < 1e-8

# Prepare Newton residual data (use pre-update sequence for iteration-wise convergence)
newton_iter_I = collect(0:length(newton_pre_I)-1)
newton_iter_F = collect(0:length(newton_pre_F)-1)

# Flatten GMRES residuals across Newton iterations, track cumulative GMRES iteration counts
gmres_flat_I = reduce(vcat, gmres_I, init=Float64[])
gmres_flat_F = reduce(vcat, gmres_F, init=Float64[])
gmres_iter_I = collect(1:length(gmres_flat_I))
gmres_iter_F = collect(1:length(gmres_flat_F))

# Newton residuals plot (separate image) with dashed lines, styling aligned to fixed-point plot
p_newton = plot(
    newton_iter_I, newton_pre_I;
    label="MO Newton",
    yscale=:log10,
    xlabel="\nNewton Iterations",
    ylabel="||Z(θ)||\n",
    title = "Newton Residuals \n(m=$(m)), Water, cc-pVTZ",
    linestyle=:dash,
    markershape=:rect,
    markersize=10,
    grid=true,
    size=(800, 600),
    legend=:topright,
    ylims=(1e-9, 3),
    yticks=10.0 .^ (-9:3:1),
    titlefont=font(24),
    guidefont=font(20),
    tickfont=font(18),
    legendfont=font(20),
    top_margin=16Plots.mm,
    bottom_margin=10Plots.mm,
    left_margin=10Plots.mm,
    right_margin=10Plots.mm,
    linewidth=2.5,
)
plot!(
    p_newton, newton_iter_F, newton_pre_F;
    label="AO Newton",
    linestyle=:dash,
    markershape=:circle,
    markersize=8,
    linewidth=2.5,
)
png_newton = joinpath(pkg_root, "test", "nk_newton_m$(m).png")
savefig(p_newton, png_newton)
@info "Saved Newton convergence plot for m=$(m) to $(png_newton)"

# GMRES residuals plot (separate image) with dashed lines, styling aligned to fixed-point plot
p_gmres = plot(
    gmres_iter_I, gmres_flat_I;
    label="MO GMRES",
    yscale=:log10,
    xlabel="\nGMRES Inner Iterations (cumulative)",
    ylabel="Projected Residual\n",
    title="GMRES Residuals \n(m=$(m)), Water, cc-pVTZ",
    linestyle=:dash,
    markershape=:rect,
    markersize=10,
    grid=true,
    size=(800, 600),
    legend=:topright,
    ylims=(1e-11, 3),
    yticks=10.0 .^ (-11:3:1),
    titlefont=font(24),
    guidefont=font(20),
    tickfont=font(18),
    legendfont=font(20),
    top_margin=16Plots.mm,
    bottom_margin=10Plots.mm,
    left_margin=10Plots.mm,
    right_margin=10Plots.mm,
    linewidth=2.5,
)
plot!(
    p_gmres, gmres_iter_F, gmres_flat_F;
    label="AO GMRES",
    linestyle=:dash,
    markershape=:circle,
    markersize=8,
    linewidth=2.5,
)
png_gmres = joinpath(pkg_root, "test", "nk_gmres_m$(m).png")
savefig(p_gmres, png_gmres)
@info "Saved GMRES convergence plot for m=$(m) to $(png_gmres)"



