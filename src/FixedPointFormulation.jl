export bigR
export deltaF
export ao_denominator
export ao_amps

function bigR(int::Integrals, coulombint::CoulombIntegrals, slice::Slices)
    a = int.a
    b = int.b
    πint = int.piint

    j_fun = coulombint.j
    k_fun = coulombint.k

    θ_fun =  theta(slice)

    return t -> begin
        θ = θ_fun(t)  
        j = j_fun(t)
        k = k_fun(t)
 
        @tensor R[μ,ν,λ,σ] := a[μ,ν,α,β] * θ[α,β,λ,σ]

        @tensor R[μ,ν,λ,σ] += θ[μ,ν,α,β] * b[α,β,λ,σ]

        tmp = similar(R)
        @tensor tmp[μ,ν,γ,δ] := θ[μ,ν,α,β] * πint[α,β,γ,δ]
        @tensor R[μ,ν,λ,σ] += tmp[μ,ν,γ,δ] * θ[γ,δ,λ,σ]

        temp = similar(R)
        @tensor temp[α,λ] := (2 * θ[γ,δ,λ,β] - θ[γ,δ,β,λ]) * πint[α,β,γ,δ]
        @tensor tmp[μ,ν,λ,σ] := temp[α,λ] * θ[μ,ν,α,σ]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        @tensor temp[μ,α] := (2 * θ[μ,β,γ,δ] - θ[μ,β,δ,γ]) * πint[γ,δ,α,β]
        @tensor tmp[μ,ν,λ,σ] := temp[μ,α] * θ[α,ν,λ,σ]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        @tensor tmp[μ,ν,λ,σ] := j[μ,α,λ,γ] * (2 * θ[γ,ν,α,σ] - θ[γ,ν,σ,α])
        R += tmp + permutedims(tmp, (2,1,4,3))
    
        @tensor tmp[μ,ν,λ,σ] := k[μ,α,γ,λ] * θ[γ,ν,α,σ]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        @tensor tmp[μ,ν,λ,σ] := k[μ,α,γ,σ] * θ[γ,ν,λ,α]
        R -= tmp + permutedims(tmp, (2,1,4,3))

        return R
    end
end

function deltaF(fd::FockDiagOffs, slice::Slices)
    Fo_off = fd.Fooff
    Fv_off = fd.Fvoff   

    θ_fun =  theta(slice)
    return t -> begin
        θ = θ_fun(t)
        @tensor ΔF[μ,ν,λ,σ] := Fv_off[α,λ] * θ[μ,ν,α,σ] + Fv_off[α,σ] * θ[μ,ν,λ,α] - Fo_off[μ,α] * θ[α,ν,λ,σ] - Fo_off[ν,α] * θ[μ,α,λ,σ]        
    end    
    return ΔF
end

function ao_denominator(fd::FockDiagOffs, nbasis)
    fo_diag = fd.Fodiag
    fv_diag = fd.Fvdiag

    denom = reshape(fo_diag, nbasis, 1, 1, 1) .+ 
            reshape(fo_diag, 1, nbasis, 1, 1) .- 
            reshape(fv_diag, 1, 1, nbasis, 1) .- 
            reshape(fv_diag, 1, 1, 1, nbasis)
    
    return denom
end

function ao_amps(int::Integrals, elt::FixedPointElements, slice::Slices, shift)
    d = int.d  
    R_fun = elt.R       
    delF_fun = elt.ΔF           
    denom = elt.denom   
    
    θ_fun =  theta(slice)

    safe_denom = denom .+ shift * sign.(denom)
    safe_denom[abs.(safe_denom) .< shift] .= shift
 
    return t -> begin
        ΔF = delF_fun(t)
        R = R_fun(t)
        θ = θ_fun(t)

        amps =  ((d .+ ΔF .+ R) .+ ((safe_denom .- denom) .* θ)) ./ safe_denom
        
        return amps
    end
end