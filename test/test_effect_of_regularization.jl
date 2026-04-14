using Revise
using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "H2-CF-7_cc-pvtz";
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
mo = zeros(nocc, nocc, nvir, nvir);
fock_mo = Cscf' * f * Cscf;
################################################################
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 200;
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
max_outer_nk = 200;
shift_non_canonical = 1.08;

T_I = Matrix{Float64}(I, n_b, n_b)
run_fixed_point = fp_iteration_factory(new_S, mo, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);
θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I);

run_fixed_point_shifted = fp_iteration_factory(new_S, mo, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical; verbose=true);
θ_final_fp_shifted, θ_benchmark_fp_shifted, diffs_fp_shifted = run_fixed_point_shifted(T_I);

run_ink = inexact_newton_factory(new_S, mo, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
θ_final_I_l, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, num_evals_I_l = run_ink(T_I);
################################################################
# Combined convergence/divergence plot for the three requested cases
x_fp = collect(1:length(diffs_I))
x_fp_shifted = collect(1:length(diffs_fp_shifted))

# Build Newton residual convergence history for inexact Newton
# newton_post contains residuals after each Newton iteration
x_ink = collect(1:length(newton_post_I_l))
y_ink = newton_post_I_l

all_series_y = Float64[]
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_I))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_fp_shifted))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_ink))

max_x = max(length(diffs_I), length(diffs_fp_shifted), length(newton_post_I_l))

p_three = plot(
    xlabel="\nnumber of residual evaluations\n",
    ylabel="\nresidual norm\n",
    title="Effect of Regularization\n Dihydrogen (7Å,cc-pVTZ)\n",
    yscale=:log10,
    legend=:topright,
    linewidth=2,
    grid=true,
    gridlinewidth=1.5,
    gridcolor=:gray40,
    gridalpha=0.6,
    size=(1200, 800),
    titlefont=font(32, "Computer Modern"),
    guidefont=font(32, "Computer Modern"),
    tickfont=font(24, "Computer Modern"),
    legendfont=font(20, "Computer Modern"),
    top_margin=16Plots.mm,
    bottom_margin=10Plots.mm,
    left_margin=10Plots.mm,
    right_margin=10Plots.mm,
    yticks=10.0 .^ (-8:2:2)
)

plot!(
    p_three, x_fp, diffs_I;
    label=false,
    color=:orange,
    linestyle=:dash,
    linewidth=5
)

plot!(
    p_three, x_fp_shifted, diffs_fp_shifted;
    label=false,
    color=:darkblue,
    linestyle=:dash,
    linewidth=5
)

plot!(
    p_three, x_ink, y_ink;
    label=false,
    color=:darkgreen,
    linestyle=:dash,
    linewidth=5
)

# Add proxy legend entries with thinner lines than the plotted curves.
legend_lw = 2.5
plot!(p_three, [NaN], [NaN]; label="FP", color=:orange, linestyle=:dash, linewidth=legend_lw)
plot!(p_three, [NaN], [NaN]; label="SFP", color=:darkblue, linestyle=:dash, linewidth=legend_lw)
plot!(p_three, [NaN], [NaN]; label="INK", color=:darkgreen, linestyle=:dash, linewidth=legend_lw)

hline!(p_three, [tol], color=:magenta, linestyle=:solid, linewidth=5, label=false)

filtered = filter(y -> isfinite(y) && y > 0, all_series_y)
if !isempty(filtered)
    y_min = max(minimum(filtered) / 5, 1e-14)
    y_max = maximum(filtered) * 5
    plot!(p_three; ylims=(y_min, y_max), xlims=(0, max_x * 1.15))
else
    plot!(p_three; xlims=(0, max_x * 1.15))
end

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_three = joinpath(fig_dir, "effect_of_regularization_$(Molecule).pdf")
svg_three = joinpath(fig_dir, "rffect_of_regularization_$(Molecule).svg")
savefig(p_three, pdf_three)
savefig(p_three, svg_three)











