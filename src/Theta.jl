export theta

function theta(slice::Slices)
    Tbarocc, Tbarvir = slice.Tbarocc, slice.Tbarvir

    return t -> begin
    @tensor temp1[μ,j,a,b] := Tbarocc[μ,i] * t[i,j,a,b]
    @tensor temp2[μ,ν,a,b] := temp1[μ,j,a,b] * Tbarocc[ν,j]
    @tensor temp3[μ,ν,λ,b] := temp2[μ,ν,a,b] * Tbarvir[λ,a]
    @tensor θ[μ,ν,λ,σ] := temp3[μ,ν,λ,b] * Tbarvir[σ,b]
    
    return θ
    end
end

export theta2mo_amp

function theta2mo_amp(slice::Slices)
    return θ -> begin
        @tensor temp1[i,nu,lambda,sigma] := slice.Tocc[mu,i] * θ[mu,nu,lambda,sigma]
        @tensor temp2[i,j,lambda,sigma] := temp1[i,nu,lambda,sigma] * slice.Tocc[nu,j]
        @tensor temp3[i,j,a,sigma] := temp2[i,j,lambda,sigma] * slice.Tvir[lambda,a]
        @tensor t[i,j,a,b] := temp3[i,j,a,sigma] * slice.Tvir[sigma,b]

        return t
    end
end