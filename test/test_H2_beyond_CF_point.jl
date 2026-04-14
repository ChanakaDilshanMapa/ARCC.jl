using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "H2-CF-6_cc-pvtz";
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
################################################################
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 300;
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
max_outer_nk = 200;
################################################################
T_I = Matrix{Float64}(I, n_b, n_b)
run_pnk_logs = preconditioned_nk_solver_factory_with_logs(new_S, initial_guess_mo, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 10);
θ_final_I_lp, θ_benchmark_I_lp, newton_pre_I_lp, newton_post_I_lp, gmres_I_lp, num_evals_I_lp = run_pnk_logs(T_I);
θ_final_F_lp, θ_benchmark_F_lp, newton_pre_F_lp, newton_post_F_lp, gmres_F_lp, num_evals_F_lp = run_pnk_logs(Cscf)

shift_non_canonical = 1.08

run_fixed_point_diis_shifted = fp_iteration_factory_diis(new_S, initial_guess_mo, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical, 5; verbose=true);
θ_final_fp_diis_shifted, θ_benchmark_fp_diis_shifted, diffs_fp_diis_shifted = run_fixed_point_diis_shifted(T_I);
@test norm(θ_final_fp_diis_shifted - θ_benchmark_fp_diis_shifted) < 1e-7


Tbar = T_bar(T_I, new_S)
slice = make_slices(Cscf, T_I, Tbar, nocc, n_b)
proj = make_projectors(slice)
int = make_integrals(proj, peris)
fop = make_fock_operators(proj, f)

j_fun = j_integralθ(int)
k_fun = k_integralθ(int)
G_o_fun = g_o(fop, int)
G_v_fun = g_v(fop, int)

function build_residual(θ)
    j = j_fun(θ)
    k = k_fun(θ)
    G_o = G_o_fun(θ)
    G_v = G_v_fun(θ)

    eqn = ar_ccd_eqns(int, j, k, G_o, G_v)
    return eqn(θ)
end

residual = build_residual(θ_final_fp_diis_shifted)

e1 = corr_ene(int, θ_final_I_lp)
e2 = corr_ene(int, θ_final_fp_diis_shifted)
norm(e1 - e2)

Tbar = T_bar(T_I, new_S)
slice = make_slices(Cscf, T_I, Tbar, nocc, n_b)
proj = make_projectors(slice)
int = make_integrals(proj, peris)

shift_non_canonical = 1.08

run_fixed_point_diis_shifted = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical, 5; verbose=true);
θ_final_fp_diis_shifted, θ_benchmark_fp_diis_shifted, diffs_fp_diis_shifted = run_fixed_point_diis_shifted(T_I);
@test norm(θ_final_fp_diis_shifted - θ_benchmark_fp_diis_shifted) < 1e-7


function is_unitary(U; atol=1e-10)
    I_mat = Matrix{eltype(U)}(I, size(U,1), size(U,2))
    return isapprox(U' * U, I_mat, atol=atol)
end

println(is_unitary(Cscf))


Tbar_I = T_bar(T_I, new_S)
slice_I = make_slices(Cscf, T_I, Tbar_I, nocc, n_b)
t2 = theta2mo_amp(slice_I)(θ_final_I_lp)
theta_c = theta(slice_I)(t2)
norm(theta_c - θ_final_F_lp)
################################################################
# Best shift finder
rho_of_eps = function (eps::Float64)
    run_anal = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, eps)
    _, spec_r = run_anal(Cscf)
    return spec_r
end

res_opt = Optim.optimize(rho_of_eps, 0.0, 10.0, Optim.Brent())
eps_min_cont = Optim.minimizer(res_opt)
rho_min_cont = Optim.minimum(res_opt)

shift_non_canonical = eps_min_cont
shift_non_canonical = 6.5
shift_canonical = 1e-8;
################################################################
# fixed-point iteration
run_fixed_point = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);
θ_final_fp, θ_benchmark_fp, diffs_fp = run_fixed_point(T_I);
@test norm(θ_final_fp - θ_benchmark_fp) > 5

# level-shifted fixed-point iteration
run_fixed_point_shifted = fp_iteration_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical; verbose=true);
θ_final_fp_shifted, θ_benchmark_fp_shifted, diffs_fp_shifted = run_fixed_point_shifted(T_I);
@test norm(θ_final_fp_shifted - θ_benchmark_fp_shifted) < 1e-7

# fixed-point iteration with DIIS
run_fixed_point_diis = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical, 5; verbose=true);
θ_final_fp_diis, θ_benchmark_fp_diis, diffs_fp_diis = run_fixed_point_diis(T_I);
@test norm(θ_final_fp_diis - θ_benchmark_fp_diis) < 1e-7

# level-shifted fixed-point iteration with DIIS
run_fixed_point_diis_shifted = fp_iteration_factory_diis(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical, 5; verbose=true);
θ_final_fp_diis_shifted, θ_benchmark_fp_diis_shifted, diffs_fp_diis_shifted = run_fixed_point_diis_shifted(T_I);
@test norm(θ_final_fp_diis_shifted - θ_benchmark_fp_diis_shifted) < 1e-7
########################################################################
# Single combined plot: overlay Fixed Point and all NK curves
start_idx = 1

function build_eval_series(newton_pre, gmres_logs)
    combined = Float64[]
    x_evals = Int[]
    iter_count = 1

    for i in eachindex(newton_pre)
        push!(combined, newton_pre[i])
        push!(x_evals, iter_count)
        iter_count += 1
        if i <= length(gmres_logs)
            for gmres_val in gmres_logs[i]
                push!(combined, gmres_val)
                push!(x_evals, iter_count)
                iter_count += 1
            end
        end
    end

    return x_evals, combined
end

########################################################################
x_evals_Ip, combined_Ip = build_eval_series(newton_pre_I_lp, gmres_I_lp)
p_all = plot(
    xlabel="\nnumber of residual evaluations\n", 
    ylabel="\nresidual norm\n", 
    title="H2 with Bond Distance 10Å (cc-pVTZ)\n", 
    yscale=:log10, 
    legend=:topright, 
    linewidth=5, 
    grid=true, 
    gridlinewidth=1.5,
    gridcolor=:gray40,
    gridalpha=0.6,
    size=(1200, 800), 
    titlefont=font(36, "Computer Modern"), 
    guidefont=font(36, "Computer Modern"), 
    tickfont=font(24, "Computer Modern"), 
    legendfont=font(24, "Computer Modern"), 
    top_margin=16Plots.mm, 
    bottom_margin=10Plots.mm, 
    left_margin=10Plots.mm, 
    right_margin=10Plots.mm,
    # legend_marker_scale=9.0,
    # legend_line_width=0.01
)

color_pnk_m10 = "#001BDB"

function valid_log_series(y::AbstractVector{<:Real}, start_idx::Int=1)
    xs = Int[]
    ys = Float64[]
    for i in start_idx:length(y)
        yi = Float64(y[i])
        if isfinite(yi) && yi > 0
            push!(xs, i)
            push!(ys, yi)
        end
    end
    return xs, ys
end

fp_x, fp_y = valid_log_series(diffs_fp, start_idx)
fp_shift_x, fp_shift_y = valid_log_series(diffs_fp_shifted, start_idx)
fp_diis_x, fp_diis_y = valid_log_series(diffs_fp_diis, start_idx)
fp_diis_shift_x, fp_diis_shift_y = valid_log_series(diffs_fp_diis_shifted, start_idx)
pnk_x, pnk_y = valid_log_series(combined_Ip, 1)

plot!(
    p_all, fp_x, fp_y;
    label = "",
    color = :darkgreen,
    linestyle = :dash,
    linewidth = 5,
    dash_pattern = "on 1cm off 0.5cm"
)
plot!(
    p_all, fp_shift_x, fp_shift_y;
    label = "",
    color = :orange,
    linestyle = :dash,
    linewidth = 5,
    dash_pattern = "on 1cm off 0.5cm on 0.25cm off 0.5cm"
)
plot!(
    p_all, fp_diis_x, fp_diis_y;
    label = "",
    color = :magenta,
    linestyle = :dash,
    linewidth = 5,
    dash_pattern = "on 0.25cm off 0.35cm"
)
plot!(
    p_all, fp_diis_shift_x, fp_diis_shift_y;
    label = "",
    color = :gray,
    linestyle = :dash,
    linewidth = 5,
    dash_pattern = "on 0.25cm off 0.35cm on 0.25cm off 0.35cm"
)
plot!(
    p_all, pnk_x, pnk_y;
    label = "",
    color = color_pnk_m10,
    linestyle = :dash,
    dash_pattern = "on 0.25cm off 0.35cm",
    linewidth = 5
)

# Proxy legend handles: keep legend samples solid while plotted curves keep their styles.
plot!(p_all, [NaN], [NaN], label="FP", color=:darkgreen, linestyle=:solid, linewidth=3)
plot!(p_all, [NaN], [NaN], label="Level-Shifted FP", color=:orange, linestyle=:solid, linewidth=3)
plot!(p_all, [NaN], [NaN], label="FPDIIS", color=:magenta, linestyle=:solid, linewidth=3)
plot!(p_all, [NaN], [NaN], label="LS-FPDIIS", color=:gray, linestyle=:solid, linewidth=3)
plot!(p_all, [NaN], [NaN], label="Preconditioned NK", color=color_pnk_m10, linestyle=:solid, linewidth=3)
 
filtered = Float64[]
append!(filtered, fp_y)
append!(filtered, fp_shift_y)
append!(filtered, fp_diis_y)
append!(filtered, fp_diis_shift_y)
append!(filtered, pnk_y)
if !isempty(filtered)
    sorted_vals = sort(filtered)
    n_vals = length(sorted_vals)
    idx_lo = max(1, floor(Int, 0.01 * n_vals))
    idx_hi = max(1, ceil(Int, 0.99 * n_vals))
    y_min = max(sorted_vals[idx_lo] / 5, 1e-8)
    y_max = sorted_vals[idx_hi] * 5
    # Optional: set sparse log ticks
    lo_exp = floor(Int, log10(y_min))
    hi_exp = ceil(Int, log10(y_max))
    tick_step = max(1, ceil(Int, (hi_exp - lo_exp) / 10))
    yticks_vals = 10.0 .^ collect(lo_exp:tick_step:hi_exp)
    max_x = maximum((isempty(fp_x) ? 1 : maximum(fp_x), isempty(fp_shift_x) ? 1 : maximum(fp_shift_x), isempty(fp_diis_x) ? 1 : maximum(fp_diis_x), isempty(fp_diis_shift_x) ? 1 : maximum(fp_diis_shift_x), isempty(pnk_x) ? 1 : maximum(pnk_x)))
    plot!(p_all; ylims=(y_min, y_max), xlims=(0, max_x * 1.25), yticks=yticks_vals)
else
    max_x = maximum((length(diffs_fp), length(diffs_fp_shifted), length(diffs_fp_diis), length(diffs_fp_diis_shifted), length(combined_Ip)))
    plot!(p_all; xlims=(0, max_x * 1.25))
end

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_all = joinpath(fig_dir, "beyond_CF_convergence_all_in_one_$(Molecule).pdf")
savefig(p_all, pdf_all)
svg_all = joinpath(fig_dir, "beyond_CF_convergence_all_in_one_$(Molecule).svg")
savefig(p_all, svg_all)
