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
run_sfp = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical; verbose=true);
θ_final_sfp, θ_benchmark_sfp, diffs_sfp = run_sfp(T_I);
@test norm(θ_final_sfp - θ_benchmark_sfp) < 1e-7

run_sfp_plus_diis = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol, shift_non_canonical, 5; verbose=true);
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
x_sfp = collect(1:length(diffs_sfp))
x_sfp_plus_diis = collect(1:length(diffs_sfp_plus_diss))

all_series_y = Float64[]
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_sfp))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, diffs_sfp_plus_diss))

max_x = max(length(diffs_sfp), length(diffs_sfp_plus_diss))

p_two = plot(
    xlabel="\nnumber of residual evaluations\n",
    ylabel="\nresidual norm\n",
    title="Effect of Jacobian Inverse\n Ethane (6-31G)\n",
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
    p_two, x_sfp, diffs_sfp;
    label=false,
    color=:orange,
    linestyle=:dash,
    linewidth=5,
    dash_pattern="on 0.70cm off 0.30cm"
)

plot!(
    p_two, x_sfp_plus_diis, diffs_sfp_plus_diss;
    label=false,
    color=:darkblue,
    linestyle=:solid,
    linewidth=5,
    dash_pattern="on 0.45cm off 0.30cm"
)

legend_lw = 2.5
plot!(p_two, [NaN], [NaN]; label="SFP", color=:orange, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 0.70cm off 0.30cm")
plot!(p_two, [NaN], [NaN]; label="SFP+DIIS", color=:darkblue, linestyle=:dash, linewidth=legend_lw, dash_pattern="on 0.45cm off 0.30cm")

hline!(p_two, [tol], color=:magenta, linestyle=:solid, linewidth=5, label=false)

filtered = filter(y -> isfinite(y) && y > 0, all_series_y)
if !isempty(filtered)
    y_min = max(minimum(filtered) / 5, 1e-14)
    y_max = maximum(filtered) * 5
    plot!(p_two; ylims=(y_min, y_max), xlims=(0, max_x * 1.15))
else
    plot!(p_two; xlims=(0, max_x * 1.15))
end

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_two = joinpath(fig_dir, "effect_of_jacobian_inverse_fp_$(Molecule).pdf")
svg_two = joinpath(fig_dir, "effect_of_jacobian_inverse_fp_$(Molecule).svg")
savefig(p_two, pdf_two)
savefig(p_two, svg_two)
