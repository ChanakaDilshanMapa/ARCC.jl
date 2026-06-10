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
max_outer_nk = 200;

rho_of_eps = function (eps::Float64)
    run_anal = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, eps)
    _, spec_r = run_anal(Cscf)
    return spec_r
end

res_opt = Optim.optimize(rho_of_eps, 0.0, 10.0, Optim.Brent())
eps_min_cont = Optim.minimizer(res_opt)
rho_min_cont = Optim.minimum(res_opt)
shift_non_canonical = eps_min_cont

T_I = Matrix{Float64}(I, n_b, n_b);

run_mp2 = mp2_amp_factory(new_S, initial_guess_mo, nocc, n_b, Cscf, f, peris, initial_guess, 1, tol,shift_canonical; verbose=false);

t_mp2 = run_mp2(T_I);

run_ink = inexact_newton_factory(new_S, t2, nocc, n_b, Cscf, f, peris, t_mp2, max_outer_nk, tol, 5);
θ_final_ink, θ_benchmark_ink, newton_pre_ink, newton_post_ink, num_evals_ink = run_ink(T_I);
@test norm(θ_final_ink - θ_benchmark_ink) < 1e-7

run_pnk = preconditioned_nk_solver_factory_with_logs(new_S, t2, nocc, n_b, Cscf, f, peris, t_mp2, max_outer_nk, tol, 15, 30);
θ_final_pnk, θ_benchmark_pnk, newton_pre_pnk, newton_post_pnk, gmres_residuals_pnk, num_evals_pnk = run_pnk(T_I);
@test norm(θ_final_pnk - θ_benchmark_pnk) < 1e-7

run_sfp = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, t_mp2, max_iter, tol,shift_non_canonical; verbose=true);
θ_final_sfp, θ_benchmark_sfp, diffs_sfp = run_sfp(T_I);
@test norm(θ_final_sfp - θ_benchmark_sfp) < 1e-7

run_sfp_plus_diis = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, t_mp2, max_iter, tol, shift_non_canonical, 5; verbose=true);
θ_final_sfp_plus_diss, θ_benchmark_sfp_plus_diss, diffs_sfp_plus_diss_raw = run_sfp_plus_diis(T_I);
@test norm(θ_final_sfp_plus_diss - θ_benchmark_sfp_plus_diss) < 1e-7

diffs_sfp_plus_diss = if diffs_sfp_plus_diss_raw isa AbstractVector
    diffs_sfp_plus_diss_raw
else
    # Backward-compatible fallback when factory returns convergence flag instead of residual history.
    Tbar = T_bar(T_I, new_S)
    slice = make_slices(Cscf, T_I, Tbar, nocc, n_b)
    proj = make_projectors(slice)
    int = make_integrals(proj, peris)
    coulomint = make_coulomb_integrals(int, slice)
    fop = make_fock_operators(proj, f)
    fd = make_fock_diags_and_offs(fop)
    elt = make_fixed_point_elements(int, coulomint, slice, fd, n_b)
    amp_fun = ao_amps(int, elt, slice, shift_non_canonical)
    t_fun = theta2mo_amp(slice)

    _, _, diffs_recovered, _ = fp_iteration_diis(
        amp_fun,
        initial_guess,
        t_fun;
        max_iter=max_iter,
        tol=tol,
        m=5,
        verbose=true,
    )
    diffs_recovered
end
################################################################
common_r0 = if !isempty(diffs_sfp) && isfinite(diffs_sfp[1]) && diffs_sfp[1] > 0
    diffs_sfp[1]
elseif !isempty(diffs_sfp_plus_diss) && isfinite(diffs_sfp_plus_diss[1]) && diffs_sfp_plus_diss[1] > 0
    diffs_sfp_plus_diss[1]
elseif !isempty(newton_pre_ink) && isfinite(newton_pre_ink[1]) && newton_pre_ink[1] > 0
    newton_pre_ink[1]
elseif !isempty(newton_pre_pnk) && isfinite(newton_pre_pnk[1]) && newton_pre_pnk[1] > 0
    newton_pre_pnk[1]
else
    1.0
end

y_sfp = vcat([common_r0], diffs_sfp)
y_sfp_plus_diis = vcat([common_r0], diffs_sfp_plus_diss)
y_ink = vcat([common_r0], newton_post_ink)
y_pnk = vcat([common_r0], newton_post_pnk)

x_sfp = collect(0:length(diffs_sfp))
x_sfp_plus_diis = collect(0:length(diffs_sfp_plus_diss))
x_ink = collect(0:length(newton_post_ink))
gmres_counts_pnk = [length(residuals) for residuals in gmres_residuals_pnk]
x_pnk = vcat(0, cumsum(1 .+ gmres_counts_pnk))

all_series_y = Float64[]
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_sfp))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_sfp_plus_diis))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_ink))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_pnk))

max_x = max(maximum(x_sfp), maximum(x_sfp_plus_diis), maximum(x_ink), maximum(x_pnk))

# Figure styling (target ~0.5\textwidth) and unified fonts
text_pt = 11
figure_width = 420
figure_height = Int(round(figure_width * 0.66))

p_two = plot(
    xlabel="\nnumber of residual evaluations",
    ylabel="residual norm\n",
    yscale=:log10,
    legend=(0.68, 0.88),
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
    p_two, x_sfp, y_sfp;
    label=false,
    color=:blue,
    linestyle=:dash,
    linewidth=2.5,
    dash_pattern="on 0.70cm off 0.30cm"
)

plot!(
    p_two, x_sfp_plus_diis, y_sfp_plus_diis;
    label=false,
    color=:lightblue,
    linestyle=:solid,
    linewidth=2.5
)

plot!(
    p_two, x_ink, y_ink;
    label=false,
    color=:gray,
    seriestype=:scatter,
    marker=:+,
    markerstrokewidth=2,
    markersize=6
)

plot!(
    p_two, x_pnk, y_pnk;
    label=false,
    color=:olive,
    linestyle=:dashdot,
    linewidth=2.5
)

legend_lw = 2.5
plot!(p_two, [NaN], [NaN]; label="SFP", color=:blue, linestyle=:dash, linewidth=1.5, dash_pattern="on 0.25cm off 0.20cm")
plot!(p_two, [NaN], [NaN]; label="SFP+DIIS", color=:lightblue, linestyle=:solid, linewidth=legend_lw)
plot!(p_two, [NaN], [NaN]; label="INK", color=:gray, seriestype=:scatter, marker=:+, markerstrokewidth=2, markersize=6)
plot!(p_two, [NaN], [NaN]; label="PNK", color=:olive, linestyle=:dashdot, linewidth=legend_lw)

hline!(p_two, [tol], color=:magenta, linestyle=:dash, linewidth=1.5, label=false)

filtered = filter(y -> isfinite(y) && y > 0, all_series_y)
if !isempty(filtered)
    y_min = max(minimum(filtered) / 5, 1e-14)
    y_max = maximum(filtered) * 5
    plot!(p_two; ylims=(y_min, y_max), xlims=(0, max_x * 1.15))
else
    plot!(p_two; xlims=(0, max_x * 1.15))
end

# tighten margins similar to matplotlib tight_layout
plot!(p_two; left_margin=0Plots.mm, right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm)

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_two = joinpath(fig_dir, "effect_of_jacobian_inverse_$(Molecule).pdf")
svg_two = joinpath(fig_dir, "effect_of_jacobian_inverse_$(Molecule).svg")
savefig(p_two, pdf_two)
savefig(p_two, svg_two)
