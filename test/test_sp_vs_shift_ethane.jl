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

peris = make_physaoeris(new_eris);
purt = 1e-6;

text_pt = 11
figure_width = 420
figure_height = Int(round(figure_width * 0.66))

epsilon = 1e-2 .* 2.0 .^(0:15)

spec_rs = similar(epsilon)
for (i, ep) in enumerate(epsilon)
    run_anal = analyzer_factory(new_S, t2, nocc, n_b, Cscf, f, peris, purt, ep)
    _, spec_r = run_anal(Cscf)
    spec_rs[i] = spec_r
end

plt_rho = plot(
    epsilon, spec_rs;
    xscale=:log10,
    xlabel="\nshift",
    ylabel="spectral radius\n",
    linewidth=2.5,
    color=:darkblue,
    legend=false,
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
    yticks=0.0:0.2:2.0,
    marker=:circle,
    markersize=4
)

hline!(plt_rho, [1.0], color=:magenta, linestyle=:dash, linewidth=1.5, label=false)

plot!(plt_rho; left_margin=0Plots.mm, right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0.8Plots.mm)

fig_dir = joinpath(pkg_root, "test/figures", Molecule)
isdir(fig_dir) || mkpath(fig_dir)
pdf_rho = joinpath(fig_dir, "spectral_radius_$(Molecule).pdf")
savefig(plt_rho, pdf_rho)
svg_rho = joinpath(fig_dir, "spectral_radius_$(Molecule).svg")
savefig(plt_rho, svg_rho)

println("Minimum spectral radius: ", minimum(spec_rs))