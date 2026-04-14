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
################################################################
run_nk = nk_solver_factory_with_logs(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
θ_final_nk, θ_benchmark_nk, newton_pre_nk, newton_post_nk, num_evals_nk = run_nk(random);
@test norm(θ_final_nk - θ_benchmark_nk) < 1e-7

run_pnk = preconditioned_nk_solver_factory_with_logs(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
θ_final_pnk, θ_benchmark_pnk, newton_pre_pnk, newton_post_pnk, num_evals_pnk = run_pnk(random);
@test norm(θ_final_pnk - θ_benchmark_pnk) < 1e-7
################################################################
x_nk = collect(1:length(newton_post_nk))
x_pnk = collect(1:length(newton_post_pnk))

r0 = if !isempty(newton_pre_nk)
    newton_pre_nk[1]
elseif !isempty(newton_pre_pnk)
    newton_pre_pnk[1]
else
    1.0
end

y_nk_plot = vcat([r0], newton_post_nk)
y_pnk_plot = vcat([r0], newton_post_pnk)
x_nk_plot = collect(0:length(newton_post_nk))
x_pnk_plot = collect(0:length(newton_post_pnk))

all_series_y = Float64[]
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_nk_plot))
append!(all_series_y, filter(y -> isfinite(y) && y > 0, y_pnk_plot))

max_x = max(length(x_nk_plot), length(x_pnk_plot))

p_two = plot(
    xlabel="\nnumber of residual evaluations\n",
    ylabel="\nresidual norm\n",
    title="Effect of Preconditioning\n Ethane (6-31G)\n",
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
    p_two, x_nk_plot, y_nk_plot;
    label=false,
    color=:orange,
    linestyle=:dash,
    linewidth=5
)

plot!(
    p_two, x_pnk_plot, y_pnk_plot;
    label=false,
    color=:darkblue,
    linestyle=:dash,
    linewidth=5
)

legend_lw = 2.5
plot!(p_two, [NaN], [NaN]; label="NK", color=:orange, linestyle=:dash, linewidth=legend_lw)
plot!(p_two, [NaN], [NaN]; label="PNK", color=:darkblue, linestyle=:dash, linewidth=legend_lw)

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
pdf_two = joinpath(fig_dir, "effect_of_preconditioning_$(Molecule).pdf")
svg_two = joinpath(fig_dir, "effect_of_preconditioning_$(Molecule).svg")
savefig(p_two, pdf_two)
savefig(p_two, svg_two)











