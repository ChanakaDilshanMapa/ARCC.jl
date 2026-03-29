export P_inv_action

function P_inv_action(F::FockOperators, θ_shape,y)
    Fo = F.Fo
    Fv = F.Fv
    y4d = reshape(y, θ_shape)
    @tensor Py[μ,ν,λ,σ] := Fv[α,λ] * y4d[μ,ν,α,σ] + Fv[α,σ] * y4d[μ,ν,λ,α] - Fo[μ,α] * y4d[α,ν,λ,σ] - Fo[ν,α] * y4d[μ,α,λ,σ]
    
    return vec(Py)
end