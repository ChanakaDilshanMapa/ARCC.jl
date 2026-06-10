using Revise,Test, LinearAlgebra, ARCC, CCD, NPZ, Glob, TensorOperations, Plots, Random, LaTeXStrings

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
fock_mo = Cscf' * f * Cscf;

nvir = n_b - nocc;
zeros_mo = zeros(nocc, nocc, nvir, nvir);
zeros_ao = zeros(n_b,n_b,n_b,n_b);

max_iter = 200;
max_outer_nk = 200;
tol = 1e-8; 
purt = 1e-6;
shift_canonical = 1e-8;

function random_orthogonal(n::Integer; rng=Random.default_rng())
    Q = qr(randn(rng, n, n)).Q
    return Matrix(Q)
end

T_I = Matrix{Float64}(I, n_b, n_b);
random = random_orthogonal(n_b);
peris = make_physaoeris(new_eris);

################################################################
t2, diffs = fixed_point_iteration(update_amps_new, zeros_mo, mo_eris, fock_mo, 300, 1e-13, true);
run_mp2 = mp2_amp_factory(new_S, zeros_mo, nocc, n_b, Cscf, f, peris, zeros_ao, 1, tol,shift_canonical; verbose=false);

t_mp2 = run_mp2(T_I);
θ_mp2_ao = run_mp2(Cscf);
θ_mp2_random = run_mp2(random);

################################################################
θ_final_mo, θ_benchmark_mo, diffs_mo = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, t_mp2, max_iter, tol,shift_canonical; verbose=true)(T_I);
θ_final_ao, θ_benchmark_ao, diffs_ao  = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, θ_mp2_ao, max_iter, tol,shift_canonical; verbose=true)(Cscf);
θ_final_random, θ_benchmark_random, diffs_random = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, θ_mp2_random, max_iter, tol,shift_canonical; verbose=true)(random);

################################################################
θ_final_mo, θ_benchmark_mo, newton_pre_mo, newton_post_mo, num_evals_mo  = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, t_mp2, max_outer_nk, tol, 4)(T_I);
θ_final_ao, θ_benchmark_ao, newton_pre_ao, newton_post_ao, num_evals_ao = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, θ_mp2_ao, max_outer_nk, tol, 4)(Cscf);
θ_final_random, θ_benchmark_random, newton_pre_random, newton_post_random, num_evals_random = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, θ_mp2_random, max_outer_nk, tol, 4)(random);

################################################################
x_fp_mo = collect(0:length(diffs_mo) - 1)
x_fp_ao = collect(0:length(diffs_ao) - 1)
x_fp_random = collect(0:length(diffs_random) - 1)

x_in_mo = collect(0:length(newton_post_mo) - 1)
x_in_ao = collect(0:length(newton_post_ao) - 1)
x_in_random = collect(0:length(newton_post_random) - 1)

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

text_pt = 11

figure_width = 420
figure_height = Int(round(figure_width * 0.66))

p_three = plot(
    xlabel="\nnumber of residual evaluations",
    ylabel="residual norm\n",
    yscale=:log10,
    legend=(0.78, 0.92),
    linewidth=1.5,
    grid=true,
    gridlinewidth=1.0,
    gridcolor=:gray40,
    gridalpha=0.6,
    size=(figure_width, figure_height),
    xguidefont=font(text_pt, "Computer Modern"),
    yguidefont=font(text_pt, "Computer Modern"),
    xtickfontsize=text_pt,
    ytickfontsize=text_pt,
    xtickfontfamily="Computer Modern",
    ytickfontfamily="Computer Modern",
    legendfontsize=text_pt,
    legendfontfamily="Computer Modern",
    titlefontsize=text_pt,
    titlefontfamily="Computer Modern",
    top_margin=0Plots.mm,
    bottom_margin=0Plots.mm,
    left_margin=0Plots.mm,
    right_margin=0Plots.mm,
    yticks=10.0 .^ (-8:2:2)
)

plot!(
    p_three, x_fp_mo, diffs_mo;
    label=false,
    color=:orange,
    linestyle=:solid,
    linewidth=2.5
)

plot!(
    p_three, x_fp_ao, diffs_ao;
    label=false,
    color=:darkgreen,
    linestyle=:dash,
    linewidth=2.5
)

plot!(
    p_three, x_fp_random, diffs_random;
    label=false,
    color=:lightgreen,
    linestyle=:dashdot,
    linewidth=2.5
)

plot!(
    p_three, x_in_mo, newton_post_mo;
    label=false,
    color=:gray,
    seriestype=:scatter,
    marker=:+,
    markerstrokewidth=2,
    markersize=6
)

plot!(
    p_three, x_in_ao, newton_post_ao;
    label=false,
    color=:brown,
    seriestype=:scatter,
    marker=:x,
    markerstrokewidth=3,
    markersize=4
)

plot!(
    p_three, x_in_random, newton_post_random;
    label=false,
    color=:black,
    seriestype=:scatter,
    marker=:dot,
    markerstrokewidth=0,
    markersize=2
)

legend_lw = 2.5

plot!(p_three, [NaN], [NaN]; label="FP MO", color=:orange, linestyle=:solid, linewidth=legend_lw)
plot!(p_three, [NaN], [NaN]; label="INK MO", color=:gray, seriestype=:scatter, marker=:+, markerstrokewidth=2, markersize=4)
plot!(p_three, [NaN], [NaN]; label="FP AO", color=:darkgreen, linestyle=:dash, linewidth=1.8)
plot!(p_three, [NaN], [NaN]; label="INK AO", color=:brown, seriestype=:scatter, marker=:x, markerstrokewidth=3, markersize=4)
plot!(p_three, [NaN], [NaN]; label="FP Ra", color=:lightgreen, linestyle=:dashdot, linewidth=legend_lw)
plot!(p_three, [NaN], [NaN]; label="INK Ra", color=:black, seriestype=:scatter, marker=:pixel, markerstrokewidth=2, markersize=2)

hline!(p_three, [tol], color=:magenta, linestyle=:dash, linewidth=1.5, label=false)

filtered = filter(y -> isfinite(y) && y > 0, all_series_y)
if !isempty(filtered)
    y_min = max(minimum(filtered) / 5, 1e-14)
    y_max = maximum(filtered) * 5
    plot!(p_three; ylims=(y_min, y_max), xlims=(0, 26))
else
    plot!(p_three; xlims=(0, 26))
end

plot!(p_three; left_margin=0Plots.mm, right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm)

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_three = joinpath(fig_dir, "effect_of_gauge_$(Molecule).pdf")
svg_three = joinpath(fig_dir, "effect_of_gauge_$(Molecule).svg")

savefig(p_three, pdf_three)
savefig(p_three, svg_three)