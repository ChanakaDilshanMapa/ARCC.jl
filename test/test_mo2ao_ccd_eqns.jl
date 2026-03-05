using AUCC, NPZ, Glob, Test, LinearAlgebra, Einsum

pkg_root = dirname(dirname(pathof(AUCC)))
base_dir = joinpath(pkg_root, "test/pyscf_data/equi_geom")

files = Dict(               
    "nocc_pyscf"           => glob("nocc_*.npy", base_dir)[1],
    "S_pyscf"              => glob("overlap_matrix_*.npy", base_dir)[1],
    "C_pyscf"              => glob("MO_coefficients_*.npy", base_dir)[1],
    "amp_ccd_pyscf"        => glob("t2_updated_*.npy", base_dir)[1],
    "ao_eris_pyscf"        => glob("ERI_*.npy", base_dir)[1],   
    "Fock_pyscf"           => glob("Fock_matrix_*.npy", base_dir)[1],
)

nocc_pyscf              = npzread(files["nocc_pyscf"])
S_pyscf                 = npzread(files["S_pyscf"])
C_pyscf                 = npzread(files["C_pyscf"])
amp_ccd_pyscf           = npzread(files["amp_ccd_pyscf"])
ao_eris_pyscf           = npzread(files["ao_eris_pyscf"])
f_pyscf                 = npzread(files["Fock_pyscf"])

C = C_pyscf
nocc = nocc_pyscf
S = S_pyscf
t2 = amp_ccd_pyscf
ao_eris = ao_eris_pyscf
f = f_pyscf

n_basis = size(S,1)

z_ao = mo2ao_ccd_eqns(C, S, t2, ao_eris, nocc, n_basis, f)

@test norm(z_ao) < 1e-14


Cnew = inv(S)
t2new = rand(2,2,4,4)

z = mo_ccd_eqns(C, S, t2new, ao_eris, nocc, n_basis, f)
z_ao2 = mo2ao_ccd_eqns2(z, Cnew, S, nocc, n_basis)

z2 = z_ao2[1:nocc, 1:nocc, nocc+1:end, nocc+1:end]
@test norm(z2 - z) < 1e-14

################################

Cnew2 = inv(sqrt(S))
t2new2 = rand(2,2,4,4)

z3 = mo_ccd_eqns(C, S, t2new2, ao_eris, nocc, n_basis, f)
z_ao3 = mo2ao_ccd_eqns3(z3, Cnew2, S, nocc, n_basis)

z4 = z_ao3[1:nocc, 1:nocc, nocc+1:end, nocc+1:end]
@test norm(z3 - z4) < 1e-14
