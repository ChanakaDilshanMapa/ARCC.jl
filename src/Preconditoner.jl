export newfunc

# function newfunc(fock_mo, Δt, nocc, nvir)    
    
#     foo = fock_mo[1:nocc, 1:nocc]
#     fvv = fock_mo[nocc+1:nocc+nvir, nocc+1:nocc+nvir]
#     # Pad foo and fvv to (6,6) with zeros if needed
#     foo_padded = zeros(eltype(foo), 6, 6)
#     foo_padded[1:size(foo,1), 1:size(foo,2)] = foo
#     fvv_padded = zeros(eltype(fvv), 6, 6)
#     fvv_padded[1:size(fvv,1), 1:size(fvv,2)] = fvv
#     foo = foo_padded
#     fvv = fvv_padded


#     @tensor result[i,j,a,b] := fvv[a,q] * Δt[i,j,q,b] + fvv[b,q] * Δt[i,j,a,q] - foo[p,i] * Δt[p,j,a,b] - foo[p,j] * Δt[i,p,a,b]
#     # @tensor result[i,j,a,b] := - fvv[a,q] * Δt[i,j,q,b] - fvv[b,q] * Δt[i,j,a,q] + foo[p,i] * Δt[p,j,a,b] + foo[p,j] * Δt[i,p,a,b]
#     # @tensor result[i,j,a,b] := fvv[a,q] * Δt[j,i,q,b] + fvv[b,q] * Δt[j,i,a,q] - foo[p,i] * Δt[p,j,b,a] + foo[p,j] * Δt[i,p,a,b]

#     result = 4 * result 
#     # result = 1 * result 

#     return result
# end

function newfunc(F::FockOperators, y)
    Fo = F.Fo
    Fv = F.Fv  

    @tensor PreC[μ,ν,λ,σ] := Fv[α,λ] * y[μ,ν,α,σ] + Fv[α,σ] * y[μ,ν,λ,α] - Fo[μ,α] * y[α,ν,λ,σ] - Fo[ν,α] * y[μ,α,λ,σ]        
    
    return PreC
end



# function newfunc(fd::FockDiagOffs,nbasis,shift, Δθ)

#     fo_diag = fd.Fodiag
#     fv_diag = fd.Fvdiag

#     denom = reshape(fo_diag, nbasis, 1, 1, 1) .+ 
#             reshape(fo_diag, 1, nbasis, 1, 1) .- 
#             reshape(fv_diag, 1, 1, nbasis, 1) .- 
#             reshape(fv_diag, 1, 1, 1, nbasis)

#     safe_denom = denom .+ shift * sign.(denom)
#     safe_denom[abs.(safe_denom) .< shift] .= shift

#     PreC = zeros(nbasis,nbasis,nbasis,nbasis)
#     # PreC .= -denom .* Δθ    
#     PreC .= -safe_denom .* Δθ    
#     return 4 .* PreC 
# end


# function newfunc(fd::FockDiagOffs,nbasis, shift, Δθ)
#     Fo_off = fd.Fooff
#     Fv_off = fd.Fvoff
#     fo_diag = fd.Fodiag
#     fv_diag = fd.Fvdiag

#     denom = reshape(fo_diag, nbasis, 1, 1, 1) .+ 
#             reshape(fo_diag, 1, nbasis, 1, 1) .- 
#             reshape(fv_diag, 1, 1, nbasis, 1) .- 
#             reshape(fv_diag, 1, 1, 1, nbasis)

#     # safe_denom = denom .+ shift * sign.(denom)
#     # safe_denom[abs.(safe_denom) .< shift] .= shift 

    
#     @tensor PreC[μ,ν,λ,σ] := Fv_off[α,λ] * Δθ[μ,ν,α,σ]
#     @tensor PreC[μ,ν,λ,σ] += Fv_off[α,σ] * Δθ[μ,ν,λ,α]
#     @tensor PreC[μ,ν,λ,σ] -= Fo_off[μ,α] * Δθ[α,ν,λ,σ]
#     @tensor PreC[μ,ν,λ,σ] -= Fo_off[ν,α] * Δθ[μ,α,λ,σ]
#     PreC .-= denom .* Δθ
#     # @tensor PreC[μ,ν,λ,σ] := Fv_off[α,λ] * Δθ[μ,ν,α,σ] + Fv_off[α,σ] * Δθ[μ,ν,λ,α] - Fo_off[μ,α] * Δθ[α,ν,λ,σ] - Fo_off[ν,α] * Δθ[μ,α,λ,σ]        
    
#     return PreC 
#     # return PreC .+ safe_denom .* Δθ
# end
