using Revise
using Test, GTO, WTP, LinearAlgebra, ARCC, CCD, NPZ, Glob, Einsum, TensorOperations, Plots, NLsolve, Random, LaTeXStrings, Optim
using Printf

pkg_root = dirname(dirname(pathof(ARCC)))
fig_dir = joinpath(pkg_root, "test/figures", "H2_bond_scan_cc-pvtz")
isdir(fig_dir) || mkpath(fig_dir)

function process_one_molecule(mol_name::String)
    println("Processing: $mol_name")

    r_match = match(r"^H2[-_]CF[-_]([0-9.]+)_cc-pvtz$", mol_name)
    r_match === nothing && error("Could not extract bond distance from folder name: $mol_name")
    bond_distance = parse(Float64, r_match.captures[1])

    base_dir = joinpath(pkg_root, "test/pyscf_data", mol_name)

    pick_first(patterns::Vector{String}) = begin
        for p in patterns
            hits = glob(p, base_dir)
            if !isempty(hits)
                return hits[1]
            end
        end
        error("Missing file for patterns $(patterns) in $base_dir")
    end

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

    D = compute_Density_matrix(nocc, Cscf);
    f = compute_Fock_matrix(new_T, new_A, new_eris, D);

    nvir = n_b - nocc;
    mo = zeros(nocc, nocc, nvir, nvir);
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

    run_ink = inexact_newton_factory(new_S, mo, nocc, n_b, Cscf, f, peris, initial_guess, max_outer_nk, tol, 5);
    θ_final, θ_benchmark_I_l, newton_pre_I_l, newton_post_I_l, num_evals_I_l = run_ink(T_I);                            

    Tbar = T_bar(T_I, new_S)
    slice = make_slices(Cscf, T_I, Tbar, nocc, n_b)
    t2_ink = theta2mo_amp(slice)(θ_final)

    run_anal = analyzer_factory(new_S, t2_ink, nocc, n_b, Cscf, f, peris, purt, 1e-8)
    _, spec_r = run_anal(T_I)

    return spec_r, bond_distance
end

molecules = [
    "H2-CF-1.00_cc-pvtz",
    "H2-CF-1.32_cc-pvtz",
    "H2-CF-1.63_cc-pvtz",
    "H2-CF-1.95_cc-pvtz",
    "H2-CF-2.26_cc-pvtz",
    "H2-CF-2.58_cc-pvtz",
    "H2-CF-2.89_cc-pvtz",
    "H2-CF-3.21_cc-pvtz",
    "H2-CF-3.53_cc-pvtz",
    "H2-CF-3.84_cc-pvtz",
    "H2-CF-4.16_cc-pvtz",
    "H2-CF-4.47_cc-pvtz",
    "H2-CF-4.79_cc-pvtz",
    "H2-CF-5.11_cc-pvtz",
    "H2-CF-5.42_cc-pvtz",
    "H2-CF-5.74_cc-pvtz",
    "H2-CF-6.05_cc-pvtz",
    "H2-CF-6.37_cc-pvtz",
    "H2-CF-6.68_cc-pvtz",
    "H2-CF-7.00_cc-pvtz"
]

R = Float64[]
spec_rs = Float64[]

for mol in molecules
    spec_r, bond_distance = process_one_molecule(mol)
    push!(R, bond_distance)
    push!(spec_rs, spec_r)
    println(@sprintf("R = %.3f Å | spectral radius = %.6e", bond_distance, spec_r))
end

y_tick_min = floor(minimum(spec_rs) * 10) / 10
y_tick_max = ceil(maximum(spec_rs) * 10) / 10
y_tick_max <= y_tick_min && (y_tick_max = y_tick_min + 0.1)
ytick_span = y_tick_max - y_tick_min
ytick_step = ytick_span > 30 ? 5.0 : ytick_span > 15 ? 2.0 : 1.0
yticks_vals = collect(y_tick_min:ytick_step:y_tick_max)
ylims_vals = (y_tick_min - 0.05 * ytick_span, y_tick_max + 0.05 * ytick_span)

# Figure styling (target ~0.5\textwidth) and unified fonts
text_pt = 11
figure_width = 420
figure_height = Int(round(figure_width * 0.66))

plt = plot(
    R,
    spec_rs;
    xlabel = "\nbond distance (Å)",
    ylabel = "spectral radius\n",
    linewidth = 1.5,
    color = :darkblue,
    legend = false,
    grid = true,
    ygrid = true,
    gridlinewidth = 1.0,
    gridcolor = :gray40,
    gridalpha = 0.6,
    size = (figure_width, figure_height),
    titlefont = font(text_pt, "Computer Modern"),
    guidefont = font(text_pt, "Computer Modern"),
    tickfont = font(text_pt, "Computer Modern"),
    legendfont = font(text_pt, "Computer Modern"),
    top_margin = 0Plots.mm,
    bottom_margin = 0Plots.mm,
    left_margin = 0Plots.mm,
    right_margin = 0Plots.mm,
    ylims = ylims_vals,
    yticks = yticks_vals,
    marker = :circle,
    markersize = 5
)

hline!(plt, [1.0], color = :magenta, linestyle = :dash, linewidth = 1.5, label = false)

# tighten margins similar to matplotlib tight_layout
plot!(plt; left_margin=0Plots.mm, right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm)


pdf_rho = joinpath(fig_dir, "spectral_radius_H2_cc-pvtz.pdf")
svg_rho = joinpath(fig_dir, "spectral_radius_H2_cc-pvtz.svg")
savefig(plt, pdf_rho)
savefig(plt, svg_rho)


############

using Plots

# ============================================================
# Data
# ============================================================

R = [
    1.00,
    1.32,
    1.63,
    1.95,
    2.26,
    2.58,
    2.89,
    3.21,
    3.53,
    3.84,
    4.16,
    4.47,
    4.79,
    5.11,
    5.42,
    5.74,
    6.05,
    6.37,
    6.68,
    7.00
]

s2 = [
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000,
    2.000000
]

gap = [
    0.650821,
    0.532080,
    0.434556,
    0.358066,
    0.299018,
    0.253397,
    0.217890,
    0.189952,
    0.167732,
    0.149899,
    0.135467,
    0.123684,
    0.113960,
    0.105835,
    0.098954,
    0.093045,
    0.087901,
    0.083368,
    0.079327,
    0.075690
]

# ============================================================
# Figure styling
# ============================================================

text_pt = 11

figure_width = 420
figure_height = Int(round(figure_width * 0.66))

# ============================================================
# Main plot : <S²>
# ============================================================

p = plot(
    R,
    s2;

    xlabel = "\nbond distance (Å)",
    ylabel = L"\langle S^2 \rangle",

    color = :darkblue,
    linewidth = 1.5,

    marker = :none,

    legend = (0.72, 0.82),

    grid = true,
    ygrid = true,
    gridlinewidth = 1.0,
    gridcolor = :gray40,
    gridalpha = 0.6,

    size = (figure_width, figure_height),

    # Fonts
    titlefont = font(text_pt, "Computer Modern"),
    guidefont = font(text_pt, "Computer Modern"),
    tickfont = font(text_pt, "Computer Modern"),
    legendfont = font(text_pt, "Computer Modern"),

    # Margins
    top_margin = 0Plots.mm,
    bottom_margin = 0Plots.mm,
    left_margin = 0Plots.mm,
    right_margin = 0Plots.mm,

    label = L"\langle S^2 \rangle"
)

# ============================================================
# Add HOMO-LUMO gap on secondary y-axis
# ============================================================

plot!(
    twinx(),
    R,
    gap;

    color = :green,
    linewidth = 1.5,
    linestyle = :dash,

    marker = :none,

    ylabel = "gap (Hartree)",

    legend = false,

    guidefont = font(text_pt, "Computer Modern"),
    tickfont = font(text_pt, "Computer Modern"),

    label = false
)

plot!(p, [NaN], [NaN]; label = "H-L gap", color = :green, linewidth = 1.5, linestyle = :dash)

# ============================================================
# Tight margins
# ============================================================

plot!(
    p;
    left_margin = 0Plots.mm,
    right_margin = 0Plots.mm,
    top_margin = 0Plots.mm,
    bottom_margin = 0Plots.mm
)

# ============================================================
# Save figures
# ============================================================

savefig(p, "combined_s2_gap_H2.pdf")

savefig(p, "combined_s2_gap_H2.svg")