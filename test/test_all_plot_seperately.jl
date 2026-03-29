using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim

pkg_root = dirname(dirname(pathof(ARCC)));
Molecule = "C2H6_Ethane_6-31G";
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


save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
upto_t2 = joinpath(save_dir, "mo_elements_$(Molecule).npz")
npzwrite(upto_t2, Dict(
    "t2" => t2,
    "diffs" => diffs,
    "nocc" => [nocc],
    "n_b" => [n_b],
    "new_S" => new_S,
    "new_T" => new_T,
    "new_A" => new_A,
    "new_eris" => new_eris,
    "Cscf" => Cscf,
    "mo_energies__SCF" => mo_energies__SCF,
    "mo_eris" => mo_eris,
    "D" => D,
    "f" => f,
    "nvir" => [nvir],
    "initial_guess_mo" => initial_guess_mo,
    "fock_mo" => fock_mo
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
upto_t2 = joinpath(save_dir, "mo_elements_$(Molecule).npz")
upto_t2 = npzread(upto_t2)

nocc = Int(upto_t2["nocc"][1])
n_b = Int(upto_t2["n_b"][1])
new_S = upto_t2["new_S"]
new_T = upto_t2["new_T"]
new_A = upto_t2["new_A"]
new_eris = upto_t2["new_eris"]
Cscf = upto_t2["Cscf"]
mo_energies__SCF = upto_t2["mo_energies__SCF"]
mo_eris = upto_t2["mo_eris"]
D = upto_t2["D"]
f = upto_t2["f"]
nvir = Int(upto_t2["nvir"][1])
initial_guess_mo = upto_t2["initial_guess_mo"]
fock_mo = upto_t2["fock_mo"]
t2 = upto_t2["t2"]
diffs = upto_t2["diffs"]

################################################################
initial_guess = zeros(n_b,n_b,n_b,n_b);
max_iter = 200;
tol = 1e-8; 
peris = make_physaoeris(new_eris);
purt = 1e-6;
shift_canonical = 1e-8;
max_outer_nk = 200;
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

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
best_shift = joinpath(save_dir, "best_shift_finder_$(Molecule).npz")
npzwrite(best_shift, Dict(
    "eps_min_cont" => [eps_min_cont],
    "rho_min_cont" => [rho_min_cont],
    "shift_non_canonical" => [shift_non_canonical]
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
best_shift = joinpath(save_dir, "best_shift_finder_$(Molecule).npz")
best_shift_data = npzread(best_shift)
eps_min_cont = best_shift_data["eps_min_cont"][1]
rho_min_cont = best_shift_data["rho_min_cont"][1]
shift_non_canonical = best_shift_data["shift_non_canonical"][1]

################################################################
# Spectral analyze
run_analyzer = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, shift_canonical);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
conclusion_I, spec_radius_I = run_analyzer(T_I)

# AO case (Full Transformation)
conclusion_F, spec_radius_F = run_analyzer(Cscf)

run_analyzer_shifted = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, shift_non_canonical);
conclusion_F_shifted, spec_radius_F_shifted = run_analyzer_shifted(Cscf)

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
shift_analyzer = joinpath(save_dir, "shift_analyzer_$(Molecule).npz")
npzwrite(shift_analyzer, Dict(
    "conclusion_I" => [conclusion_I == "attractor"],
    "spec_radius_I" => [spec_radius_I],
    "conclusion_F" => [conclusion_F == "attractor"],
    "spec_radius_F" => [spec_radius_F],
    "conclusion_F_shifted" => [conclusion_F_shifted == "attractor"],
    "spec_radius_F_shifted" => [spec_radius_F_shifted]
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
shift_analyzer = joinpath(save_dir, "shift_analyzer_$(Molecule).npz")
shift_analyzer_data = npzread(shift_analyzer)
conclusion_I = shift_analyzer_data["conclusion_I"][1]
spec_radius_I = shift_analyzer_data["spec_radius_I"][1]
conclusion_F = shift_analyzer_data["conclusion_F"][1]
spec_radius_F = shift_analyzer_data["spec_radius_F"][1]
conclusion_F_shifted = shift_analyzer_data["conclusion_F_shifted"][1]
spec_radius_F_shifted = shift_analyzer_data["spec_radius_F_shifted"][1]

################################################################
run_fixed_point = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_canonical; verbose=true);

# MO case (Identity Transformation)
T_I = Matrix{Float64}(I, n_b, n_b);
θ_final_I, θ_benchmark_I, diffs_I = run_fixed_point(T_I);
@test norm(θ_final_I - θ_benchmark_I) < 1e-7

# AO case (Full Transformation)
T_F = Cscf;
θ_final_F, θ_benchmark_F, diffs_F = run_fixed_point(Cscf);
@test norm(θ_final_F - θ_benchmark_F) > 10

run_fixed_point_shifted = fixed_point_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_iter, tol,shift_non_canonical; verbose=true);
θ_final_F_shifted, θ_benchmark_F_shifted, diffs_F_shifted = run_fixed_point_shifted(Cscf);
@test norm(θ_final_F_shifted - θ_benchmark_F_shifted) < 1e-7

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
fp_data_file = joinpath(save_dir, "fixed_point_results_$(Molecule).npz")
npzwrite(fp_data_file, Dict(
    "theta_final_I" => θ_final_I,
    "theta_benchmark_I" => θ_benchmark_I,
    "diffs_I" => diffs_I,
    "theta_final_F" => θ_final_F,
    "theta_benchmark_F" => θ_benchmark_F,
    "diffs_F" => diffs_F,
    "theta_final_F_shifted" => θ_final_F_shifted,
    "theta_benchmark_F_shifted" => θ_benchmark_F_shifted,
    "diffs_F_shifted" => diffs_F_shifted
))
########################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
fp_data_file = joinpath(save_dir, "fixed_point_results_$(Molecule).npz")
fp_saved = npzread(fp_data_file)
θ_final_I = fp_saved["theta_final_I"]
θ_benchmark_I = fp_saved["theta_benchmark_I"]
diffs_I = fp_saved["diffs_I"]
θ_final_F = fp_saved["theta_final_F"]
θ_benchmark_F = fp_saved["theta_benchmark_F"]
diffs_F = fp_saved["diffs_F"]
θ_final_F_shifted = fp_saved["theta_final_F_shifted"]
θ_benchmark_F_shifted = fp_saved["theta_benchmark_F_shifted"]
diffs_F_shifted = fp_saved["diffs_F_shifted"]

########################################################################
# FIXED POINT FIGURE
start_idx = 1
p_fp = plot(
    xlabel="\nnumber of residual evaluations\n", 
    ylabel="\nresidual norm\n", 
    title="Ethane (6-31G)\n", 
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
    right_margin=10Plots.mm
)

plot!(
    p_fp, start_idx:length(diffs_I), diffs_I[start_idx:end], 
    label="", color=:darkgreen, linestyle=:dash, linewidth=5
)
plot!(
    p_fp, start_idx:length(diffs_F), diffs_F[start_idx:end], 
    label="", color=:magenta, linestyle=:dash, linewidth=5
)
plot!(
    p_fp, start_idx:length(diffs_F_shifted), diffs_F_shifted[start_idx:end], 
    label="", color=:orange, linestyle=:dash, linewidth=5
)

# Add invisible thin lines for legend only
plot!(p_fp, [NaN], [NaN], label="FP MO basis", color=:darkgreen, linestyle=:solid, linewidth=3)
plot!(p_fp, [NaN], [NaN], label="FP AO basis", color=:magenta, linestyle=:solid, linewidth=3)
plot!(p_fp, [NaN], [NaN], label="FP Shifted AO basis", color=:orange, linestyle=:solid, linewidth=3)

# Set limits for FP plot
fp_series_y = Float64[]
append!(fp_series_y, filter(y -> isfinite(y) && y > 0, diffs_I[start_idx:end]))
append!(fp_series_y, filter(y -> isfinite(y) && y > 0, diffs_F[start_idx:end]))
append!(fp_series_y, filter(y -> isfinite(y) && y > 0, diffs_F_shifted[start_idx:end]))
fp_max_x = maximum((length(diffs_I), length(diffs_F), length(diffs_F_shifted)))

if !isempty(fp_series_y)
    sorted_vals = sort(fp_series_y)
    n_vals = length(sorted_vals)
    idx_lo = max(1, floor(Int, 0.01 * n_vals))
    idx_hi = max(1, ceil(Int, 0.99 * n_vals))
    y_min = max(sorted_vals[idx_lo] / 5, 1e-8)
    y_max = sorted_vals[idx_hi] * 5
    lo_exp = floor(Int, log10(y_min))
    hi_exp = ceil(Int, log10(y_max))
    tick_step = max(1, ceil(Int, (hi_exp - lo_exp) / 10))
    yticks_vals = 10.0 .^ collect(lo_exp:tick_step:hi_exp)
    plot!(p_fp; ylims=(y_min, y_max), xlims=(0, fp_max_x * 1.25), yticks=yticks_vals)
else
    plot!(p_fp; xlims=(0, fp_max_x * 1.25))
end

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_fp = joinpath(fig_dir, "convergence_fixed_point_$(Molecule).pdf")
savefig(p_fp, pdf_fp)
svg_fp = joinpath(fig_dir, "convergence_fixed_point_$(Molecule).svg")
savefig(p_fp, svg_fp)

################################################################
nk_data = Dict{Int, NamedTuple{(:x_I, :combined_I, :x_F, :combined_F), Tuple{Vector{Int}, Vector{Float64}, Vector{Int}, Vector{Float64}}}}()

# Initialize variables for NK convergence tracking
all_series_y = Float64[]
max_x = 0
start_idx = 1

T_I = Matrix{Float64}(I, n_b, n_b)
for m_val in (3, 5, 10)
    run_nk_logs = nk_logs_factory(new_S, t2, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, m_val)
    θ_final_I_l, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, gmres_I_l, num_evals_I_l = run_nk_logs(T_I)
    θ_final_F_l, θ_benchmark_F_l, newton_pre_F_l, newton_post_F_l, gmres_F_l, num_evals_F_l = run_nk_logs(Cscf)

    @test norm(θ_final_I_l - θ_benchmark_I_l) < 1e-7
    @test norm(θ_final_F_l - θ_benchmark_F_l) < 1e-7

    combined_I = Float64[]
    combined_F = Float64[]
    x_evals_I = Int[]  
    x_evals_F = Int[]  
    
    iter_count = 1
    for i in 1:length(newton_pre_I_l)
        push!(combined_I, newton_pre_I_l[i])
        push!(x_evals_I, iter_count)
        iter_count += 1
        if i <= length(gmres_I_l)
            for gmres_val in gmres_I_l[i]
                push!(combined_I, gmres_val)
                push!(x_evals_I, iter_count)
                iter_count += 1
            end
        end
    end

    iter_count = 1
    for i in 1:length(newton_pre_F_l)
        push!(combined_F, newton_pre_F_l[i])
        push!(x_evals_F, iter_count)
        iter_count += 1
        if i <= length(gmres_F_l)
            for gmres_val in gmres_F_l[i]
                push!(combined_F, gmres_val)
                push!(x_evals_F, iter_count)
                iter_count += 1
            end
        end
    end

    nk_data[m_val] = (x_I=x_evals_I, combined_I=combined_I, x_F=x_evals_F, combined_F=combined_F)
    append!(all_series_y, filter(y -> isfinite(y) && y > 0, combined_I))
    append!(all_series_y, filter(y -> isfinite(y) && y > 0, combined_F))
    max_x = maximum((max_x, maximum(x_evals_I), maximum(x_evals_F)))
end

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
nk_data_file = joinpath(save_dir, "nk_results_$(Molecule).npz")
npzwrite(nk_data_file, Dict(
    "start_idx" => [start_idx],
    "max_x" => [max_x],
    "all_series_y" => all_series_y,
    "diffs_I" => diffs_I,
    "diffs_F" => diffs_F,
    "diffs_F_shifted" => diffs_F_shifted,
    "m3_x_I" => nk_data[3].x_I,
    "m3_combined_I" => nk_data[3].combined_I,
    "m3_x_F" => nk_data[3].x_F,
    "m3_combined_F" => nk_data[3].combined_F,
    "m5_x_I" => nk_data[5].x_I,
    "m5_combined_I" => nk_data[5].combined_I,
    "m5_x_F" => nk_data[5].x_F,
    "m5_combined_F" => nk_data[5].combined_F,
    "m10_x_I" => nk_data[10].x_I,
    "m10_combined_I" => nk_data[10].combined_I,
    "m10_x_F" => nk_data[10].x_F,
    "m10_combined_F" => nk_data[10].combined_F
))

################################################################
nk_data_file = joinpath(save_dir, "nk_results_$(Molecule).npz")
nk_saved = npzread(nk_data_file)
start_idx = Int(nk_saved["start_idx"][1])
max_x = Int(nk_saved["max_x"][1])
all_series_y = nk_saved["all_series_y"]
diffs_I = nk_saved["diffs_I"]
diffs_F = nk_saved["diffs_F"]
diffs_F_shifted = nk_saved["diffs_F_shifted"]
nk_data = Dict(
    3 => (x_I=nk_saved["m3_x_I"], combined_I=nk_saved["m3_combined_I"], x_F=nk_saved["m3_x_F"], combined_F=nk_saved["m3_combined_F"]),
    5 => (x_I=nk_saved["m5_x_I"], combined_I=nk_saved["m5_combined_I"], x_F=nk_saved["m5_x_F"], combined_F=nk_saved["m5_combined_F"]),
    10 => (x_I=nk_saved["m10_x_I"], combined_I=nk_saved["m10_combined_I"], x_F=nk_saved["m10_x_F"], combined_F=nk_saved["m10_combined_F"])
)

################################################################
# NEWTON-KRYLOV FIGURE
p_nk = plot(
    xlabel="\nnumber of residual evaluations\n", 
    ylabel="\nresidual norm\n", 
    title="Ethane (6-31G)\n", 
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
    right_margin=10Plots.mm
)

color_map_mo = Dict(3=>"#D02000", 5=>"#6f4f1d", 10=>"#001BDB")
color_map_ao = Dict(3=>"#F08080", 5=>"#9c7248", 10=>"#8080FD")
shape_map_ao = Dict(3=>:circle, 5=>:circle, 10=>:circle)
shape_map_mo = Dict(3=>:rect, 5=>:rect, 10=>:rect)
markersize_mo = Dict(3=>9, 5=>9, 10=>9)
markersize_ao = Dict(3=>5, 5=>5, 10=>5)

for m_val in (3, 5, 10)
    data = nk_data[m_val]
    plot!(
        p_nk, data.x_I, data.combined_I;
        label = "",
        color = color_map_mo[m_val],
        linestyle = :dash,
        linewidth = 5,
        dash_pattern="on 1cm off 0.5cm"
    )
    plot!(
        p_nk, data.x_F, data.combined_F + 10 * 10 .^(-float(data.x_F));
        label = "",
        color = color_map_ao[m_val],
        linestyle = :dash,
        linewidth = 5,
        dash_pattern="on 1cm off 0.5cm on 0.25 cm off 0.5cm"
    )
end

# Add invisible thin lines for legend only
plot!(p_nk, [NaN], [NaN], label="NK MO (m=3)", color=color_map_mo[3], linestyle=:solid, linewidth=2)
plot!(p_nk, [NaN], [NaN], label="NK AO (m=3)", color=color_map_ao[3], linestyle=:solid, linewidth=2)
plot!(p_nk, [NaN], [NaN], label="NK MO (m=5)", color=color_map_mo[5], linestyle=:solid, linewidth=2)
plot!(p_nk, [NaN], [NaN], label="NK AO (m=5)", color=color_map_ao[5], linestyle=:solid, linewidth=2)
plot!(p_nk, [NaN], [NaN], label="NK MO (m=10)", color=color_map_mo[10], linestyle=:solid, linewidth=2)
plot!(p_nk, [NaN], [NaN], label="NK AO (m=10)", color=color_map_ao[10], linestyle=:solid, linewidth=2)

# Set limits for NK plot
nk_series_y = Float64[]
for m_val in (3, 5, 10)
    append!(nk_series_y, filter(y -> isfinite(y) && y > 0, nk_data[m_val].combined_I))
    append!(nk_series_y, filter(y -> isfinite(y) && y > 0, nk_data[m_val].combined_F))
end
nk_max_x = max_x

if !isempty(nk_series_y)
    sorted_vals = sort(nk_series_y)
    n_vals = length(sorted_vals)
    idx_lo = max(1, floor(Int, 0.01 * n_vals))
    idx_hi = max(1, ceil(Int, 0.99 * n_vals))
    y_min = max(sorted_vals[idx_lo] / 5, 1e-8)
    y_max = sorted_vals[idx_hi] * 5
    lo_exp = floor(Int, log10(y_min))
    hi_exp = ceil(Int, log10(y_max))
    tick_step = max(1, ceil(Int, (hi_exp - lo_exp) / 10))
    yticks_vals = 10.0 .^ collect(lo_exp:tick_step:hi_exp)
    plot!(p_nk; ylims=(y_min, y_max), xlims=(0, nk_max_x * 1.25), yticks=yticks_vals)
else
    plot!(p_nk; xlims=(0, nk_max_x * 1.25))
end

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_nk = joinpath(fig_dir, "convergence_newton_krylov_$(Molecule).pdf")
savefig(p_nk, pdf_nk)
svg_nk = joinpath(fig_dir, "convergence_newton_krylov_$(Molecule).svg")
savefig(p_nk, svg_nk)

################################################################
epsilon = 1e-2 .* 2.0 .^(0:15)

spec_rs = similar(epsilon)
for (i, ep) in enumerate(epsilon)
    run_anal = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, ep)
    _, spec_r = run_anal(Cscf)
    spec_rs[i] = spec_r
end

save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
isdir(save_dir) || mkpath(save_dir)
rho_data_file = joinpath(save_dir, "spectral_radius_results_$(Molecule).npz")
npzwrite(rho_data_file, Dict(
    "epsilon" => epsilon,
    "spec_rs" => spec_rs
))

################################################################
save_dir = joinpath(pkg_root, "test", "saved_data", Molecule)
rho_data_file = joinpath(save_dir, "spectral_radius_results_$(Molecule).npz")
rho_saved = npzread(rho_data_file)
epsilon = rho_saved["epsilon"]
spec_rs = rho_saved["spec_rs"]

plt_rho = plot(
    epsilon,
    spec_rs;
    xscale = :log10,
    xlabel = "\nshift\n",
    ylabel = "\nspectral radius\n",
    title = "Spectral radius vs. shift, Ethane (6-31G)\n",
    linewidth = 3,
    color=:darkblue,
    legend = false,
    grid = true,
    gridlinewidth = 1.5,
    gridcolor = :gray40,
    gridalpha = 0.6,
    size = (1200, 800),
    titlefont = font(36, "Computer Modern"),
    guidefont = font(36, "Computer Modern"),
    tickfont = font(24, "Computer Modern"),
    legendfont = font(24, "Computer Modern"),
    top_margin = 16Plots.mm,
    bottom_margin = 10Plots.mm,
    left_margin = 10Plots.mm,
    right_margin = 10Plots.mm,
    yticks = 0.0:0.2:2.0,
    marker=:circle,
    markersize=10
)

hline!(plt_rho, [1.0], color = :magenta, linestyle = :solid, linewidth = 3, label = false)

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_rho = joinpath(fig_dir, "spectral_radius_$(Molecule).pdf")
savefig(plt_rho, pdf_rho)
svg_rho = joinpath(fig_dir, "spectral_radius_$(Molecule).svg")
savefig(plt_rho, svg_rho)



