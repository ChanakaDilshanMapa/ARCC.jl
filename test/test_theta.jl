using AUCC, NPZ, Glob, Test, LinearAlgebra, Einsum

pkg_root = dirname(dirname(pathof(AUCC)))
base_dir = joinpath(pkg_root, "test/pyscf_data/equi_geom")

files = Dict(               
    "nocc_pyscf"           => glob("nocc_*.npy", base_dir)[1],
    "S_pyscf"              => glob("overlap_matrix_*.npy", base_dir)[1],
    "C_pyscf"              => glob("MO_coefficients_*.npy", base_dir)[1],
    "amp_ccd_pyscf"        => glob("t2_updated_*.npy", base_dir)[1],        
)

nocc_pyscf              = npzread(files["nocc_pyscf"])
S_pyscf                 = npzread(files["S_pyscf"])
C_pyscf                 = npzread(files["C_pyscf"])
amp_ccd_pyscf           = npzread(files["amp_ccd_pyscf"])

C = C_pyscf
nocc = nocc_pyscf
S = S_pyscf
t2 = amp_ccd_pyscf

n_basis = size(S,1)

t = mo2ao_amps_theta_check(C, S, t2, nocc, n_basis)

@test norm(t -t2) < 1e-15