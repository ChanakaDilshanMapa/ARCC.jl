export ao2mo_eris

function ao2mo_eris(ao_eris, C)
   	@einsum temp_1[i,nu,lambda,sigma]   := C[mu,i]      * ao_eris[mu,nu,lambda,sigma]
	@einsum temp_2[i,j,lambda,sigma]    := C[nu,j]      * temp_1[i,nu,lambda,sigma]
	@einsum temp_3[i,j,a,sigma]         := C[lambda,a]  * temp_2[i,j,lambda,sigma] 
	@einsum mo_eris[i,j,a,b]            := C[sigma,b]   * temp_3[i,j,a,sigma]
    return mo_eris
end

export phys_aoeris

function phys_aoeris(aoeris)
    return permutedims(aoeris, (1, 3, 2, 4))
end

export piintegrals

function piintegrals(proj::Projectors, peris::PhysAOEris)
    P = proj.P 
    Q = proj.Q   
    v = peris.physaoeris 

    @tensor tmp1[gamma,delta,mu,beta] := v[gamma,delta,alpha,beta] * P[alpha,mu]
    @tensor tmp2[gamma,delta,mu,nu]  := tmp1[gamma,delta,mu,beta] * P[beta,nu]
    @tensor tmp3[lambda,delta,mu,nu] := Q[lambda,gamma] * tmp2[gamma,delta,mu,nu]
    @tensor piint[lambda,sigma,mu,nu] := tmp3[lambda,delta,mu,nu] * Q[sigma,delta]
    
    return piint    
end

export bigpi

function bigpi(proj::Projectors, peris::PhysAOEris)
    piint = piintegrals(proj, peris)
    bpi = 2 * piint .- permutedims(piint, (2,1,3,4))        
   
    return bpi
end

export a_integral
export b_integral
export c_integral
export c_integral_permuted
export d_integral

function a_integral(proj::Projectors, peris::PhysAOEris)
    P = proj.P 
    Pbar = proj.Pbar
    v = peris.physaoeris    

    @tensor temp1[μ,ε,η,τ] := Pbar[μ,ρ] * v[ρ,ε,η,τ]
    @tensor temp2[μ,ν,η,τ] := temp1[μ,ε,η,τ] * Pbar[ν,ε]
    @tensor temp3[μ,ν,α,τ] := temp2[μ,ν,η,τ] * P[η,α]
    @tensor a[μ,ν,α,β] := temp3[μ,ν,α,τ] * P[τ,β]

    return a
end

function b_integral(proj::Projectors, peris::PhysAOEris)  
    Q = proj.Q
    Qbar = proj.Qbar
    v = peris.physaoeris 
    
    @tensor temp1[α,ε,η,τ] := Q[α,ρ] * v[ρ,ε,η,τ]
    @tensor temp2[α,β,η,τ] := temp1[α,ε,η,τ] * Q[β,ε]
    @tensor temp3[α,β,λ,τ] := temp2[α,β,η,τ] * Qbar[λ,η]
    @tensor b[α,β,λ,σ] := temp3[α,β,λ,τ] * Qbar[σ,τ]

    return b
end

function c_integral(proj::Projectors, peris::PhysAOEris) 
    P = proj.P
    Pbar = proj.Pbar 
    Q = proj.Q
    Qbar = proj.Qbar
    v = peris.physaoeris 
   
    @tensor temp1[μ,ε,η,τ] := Pbar[μ,ρ] * v[ρ,ε,η,τ]
    @tensor temp2[μ,α,η,τ] := temp1[μ,ε,η,τ] * Q[α,ε]
    @tensor temp3[μ,α,γ,τ] := temp2[μ,α,η,τ] * P[η,γ]
    @tensor c[μ,α,γ,λ] := temp3[μ,α,γ,τ] * Qbar[λ,τ]

    return c
end

function c_integral_permuted(proj::Projectors, peris::PhysAOEris) 
    P = proj.P
    Pbar = proj.Pbar 
    Q = proj.Q
    Qbar = proj.Qbar
    v = peris.physaoeris 

    @tensor temp1[μ,ε,τ,η] := Pbar[μ,ρ] * v[ρ,ε,τ,η]
    @tensor temp2[μ,α,τ,η] := temp1[μ,ε,τ,η] * Q[α,ε]
    @tensor temp3[μ,α,λ,η] := temp2[μ,α,τ,η] * Qbar[λ,τ]
    @tensor cperm[μ,α,λ,γ] := temp3[μ,α,λ,η] * P[η,γ]

    return cperm
end

function d_integral(proj::Projectors, peris::PhysAOEris)
    Pbar = proj.Pbar 
    Qbar = proj.Qbar
    v = peris.physaoeris 

    @tensor temp1[μ,ε,η,τ] := Pbar[μ,ρ] * v[ρ,ε,η,τ]
    @tensor temp2[μ,ν,η,τ] := temp1[μ,ε,η,τ] * Pbar[ν,ε]
    @tensor temp3[μ,ν,λ,τ] := temp2[μ,ν,η,τ] * Qbar[λ,η]
    @tensor d[μ,ν,λ,σ] := temp3[μ,ν,λ,τ] * Qbar[σ,τ]
        
    return d
end

export j_integral
export k_integral

function j_integral(int::Integrals, slice::Slices)
    c_permuted = int.cperm
    piint = int.piint
    θ_fun =  theta(slice)

    return t -> begin
        θ = θ_fun(t)
        j = copy(c_permuted)
        @tensor j[mu, alpha, lambda, gamma] -= 0.5 * θ[mu, delta, beta, lambda] * piint[alpha, beta, gamma, delta]
        @tensor j[mu, alpha, lambda, gamma] += 0.5 * θ[mu, delta, lambda, beta] * (2 * piint[alpha, beta, gamma, delta] - piint[beta, alpha, gamma, delta])
        return j
    end
end

function k_integral(int::Integrals, slice::Slices)
    c = int.c
    piint = int.piint
    θ_fun =  theta(slice)

    return t -> begin
        θ = θ_fun(t)
        k = copy(c)
        @tensor k[mu,alpha,gamma,lambda] -= 0.5 * θ[mu,delta,beta,lambda] * piint[beta,alpha,gamma,delta]
        return k
    end   
end

export j_integralθ
export k_integralθ

function j_integralθ(int::Integrals)
    c_permuted = int.cperm
    piint = int.piint

    return θ -> begin
        j = copy(c_permuted)
        @tensor j[mu, alpha, lambda, gamma] -= 0.5 * θ[mu, delta, beta, lambda] * piint[alpha, beta, gamma, delta]
        @tensor j[mu, alpha, lambda, gamma] += 0.5 * θ[mu, delta, lambda, beta] * (2 * piint[alpha, beta, gamma, delta] - piint[beta, alpha, gamma, delta])
        return j
    end
end

function k_integralθ(int::Integrals)
    c = int.c
    piint = int.piint

    return θ -> begin
        k = copy(c)
        @tensor k[mu,alpha,gamma,lambda] -= 0.5 * θ[mu,delta,beta,lambda] * piint[beta,alpha,gamma,delta]
        return k
    end   
end