using Revise
using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "C2H6_6-31g";
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
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
shift_non_canonical = 1.08;
max_outer_nk = 200;

function random_orthogonal(n::Integer; rng=Random.default_rng())
    Q = qr(randn(rng, n, n)).Q
    return Matrix(Q)
end

T_I = Matrix{Float64}(I, n_b, n_b);
random = random_orthogonal(n_b)
run_fixed_point = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);
θ_final_mo, θ_benchmark_mo, diffs_mo = run_fixed_point(T_I);
@test norm(θ_final_mo - θ_benchmark_mo) < 1e-7

θ_final_ao, θ_benchmark_ao, diffs_ao = run_fixed_point(Cscf);
@test norm(θ_final_ao - θ_benchmark_ao) < 1e-7

θ_final_random, θ_benchmark_random, diffs_random = run_fixed_point(random);
@test norm(θ_final_random - θ_benchmark_random) < 1e-7

################################################################
run_in = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
θ_final_mo, θ_benchmark_mo, newton_pre_mo, newton_post_mo, num_evals_mo = run_in(T_I);

run_in = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
θ_final_ao, θ_benchmark_ao, newton_pre_ao, newton_post_ao, num_evals_ao = run_in(Cscf);

run_in = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
θ_final_random, θ_benchmark_random, newton_pre_random, newton_post_random, num_evals_random = run_in(random);
################################################################
# Combined convergence/divergence plot for the three requested cases
x_fp_mo = collect(1:length(diffs_mo))
x_fp_ao = collect(1:length(diffs_ao))
x_fp_random = collect(1:length(diffs_random))

# Build Newton residual convergence history for inexact Newton
# newton_post contains residuals after each Newton iteration
x_in_mo = collect(1:length(newton_post_mo))
x_in_ao = collect(1:length(newton_post_ao))
x_in_random = collect(1:length(newton_post_random))

all_series_y = Float64[]
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_mo))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_ao))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_random))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, newton_post_mo))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, newton_post_ao))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, newton_post_random))

max_x = maximum((
    length(diffs_mo),
    length(diffs_ao),
    length(diffs_random),
    length(newton_post_mo),
    length(newton_post_ao),
    length(newton_post_random)
))

p_three = plot(
    xlabel="\nnumber of residual evaluations\n",
    ylabel="\nresidual norm\n",
    title="Effect of Guage\n Ethane (6-31G)\n",
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
    p_three, x_fp_mo, diffs_mo;
    label=false,
    color=:orange,
    linestyle=:dash,
    linewidth=5
)

plot!(
    p_three, x_fp_ao, diffs_ao;
    label=false,
    color=:lightgreen,
    linestyle=:dash,
    linewidth=5
)

plot!(
    p_three, x_fp_random, diffs_random;
    label=false,
    color=:darkgreen,
    linestyle=:dash,
    linewidth=5
)

plot!(
    p_three, x_in_mo, newton_post_mo;
    label=false,
    color=:brown,
    linestyle=:dash,
    linewidth=6,
    dash_pattern="on 0.70cm off 0.30cm"
)

plot!(
    p_three, x_in_ao, newton_post_ao;
    label=false,
    color=:black,
    linestyle=:dash,
    linewidth=5,
    dash_pattern="on 0.45cm off 0.30cm"
)

plot!(
    p_three, x_in_random, newton_post_random;
    label=false,
    color=:gray,
    linestyle=:dash,
    linewidth=4,
    dash_pattern="on 0.20cm off 0.30cm"
)

# Add proxy legend entries with thinner lines than the plotted curves.
legend_lw = 2.5
plot!(p_three, [NaN], [NaN]; label="FP MO", color=:orange, linestyle=:dash, linewidth=legend_lw)
plot!(p_three, [NaN], [NaN]; label="IN MO", color=:brown, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 0.70cm off 0.30cm")
plot!(p_three, [NaN], [NaN]; label="FP AO", color=:lightgreen, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 1.25cm off 0.32cm")
plot!(p_three, [NaN], [NaN]; label="IN AO", color=:black, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 0.45cm off 0.30cm")
plot!(p_three, [NaN], [NaN]; label="FP Random", color=:darkgreen, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 0.22cm off 0.18cm")
plot!(p_three, [NaN], [NaN]; label="IN Random", color=:gray, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 0.20cm off 0.30cm")

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
pdf_three = joinpath(fig_dir, "effect_of_gauge_$(Molecule).pdf")
svg_three = joinpath(fig_dir, "effect_of_gauge_$(Molecule).svg")
savefig(p_three, pdf_three)
savefig(p_three, svg_three)











