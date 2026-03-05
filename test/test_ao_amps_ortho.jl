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

P = ao_projector(C, nocc)
Q = complementary_projector(C,nocc,n_basis)

P_bar = ao_projector_bar(C, S, nocc)
Q_bar = complementary_projector_bar(C, S , nocc, n_basis)

theta = mo2ao_amps_theta(C, S, t2, nocc, n_basis)

@einsum temp1[mu,alpha] := S[mu,rho] * P[rho,alpha]
@einsum temp2[nu,beta] := S[nu,epsilon] * P[epsilon,beta]
@einsum temp3[lambda,gamma] := S[neta,lambda] * Q[gamma,neta]
@einsum temp4[sigma,delta] := S[tau,sigma] * Q[delta,tau]
@einsum check1[mu,nu,lambda,sigma] := temp1[mu,alpha] * temp2[nu,beta] * temp3[lambda,gamma] * temp4[sigma,delta] * theta[alpha,beta,gamma,delta]

@einsum temp5[mu,alpha] := S[mu,rho] * Q[rho,alpha]
@einsum temp6[nu,beta] := S[nu,epsilon] * P[epsilon,beta]
@einsum temp7[lambda,gamma] := S[neta,lambda] * Q[gamma,neta]
@einsum temp8[sigma,delta] := S[tau,sigma] * Q[delta,tau]
@einsum check2[mu,nu,lambda,sigma] := temp5[mu,alpha] * temp6[nu,beta] * temp7[lambda,gamma] * temp8[sigma,delta] * theta[alpha,beta,gamma,delta]

@einsum check3[mu,nu,lambda,sigma] := P_bar[mu,alpha] * P_bar[nu,beta] * Q_bar[lambda, gamma] * Q_bar[sigma, delta] * theta[alpha,beta,gamma,delta]

@einsum check4[mu,nu,lambda,sigma] := P_bar[mu,alpha] * Q_bar[nu,beta] * Q_bar[lambda, gamma] * Q_bar[sigma, delta] * theta[alpha,beta,gamma,delta]

@test norm(theta - check1) < 1e-15
@test norm(check2) < 1e-15

@test norm(theta - check3) < 1e-15
@test norm(check4) < 1e-15


