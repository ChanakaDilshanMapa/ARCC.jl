export g_o
export g_v
export ar_ccd_eqns

function g_o(fop::FockOperators, int::Integrals)
    F_o = fop.Fo
    piint = int.piint  

    return θ -> begin
        G_o = deepcopy(F_o)       
        @tensor G_o[mu,alpha] += θ[mu,beta,gamma,delta] * (2 * piint[gamma,delta,alpha,beta] - piint[delta,gamma,alpha,beta])   
        return G_o  
    end 
end

function g_v(fop::FockOperators, int::Integrals)
    F_v = fop.Fv
    piint = int.piint 

    return θ -> begin
        G_v = deepcopy(F_v)       
        @tensor G_v[alpha,lambda] -= θ[gamma,delta,lambda,beta] * (2 * piint[alpha,beta,gamma,delta] - piint[beta,alpha,gamma,delta])   
        return G_v  
    end 
end


function ar_ccd_eqns(int::Integrals, j, k , G_o, G_v)
    a = int.a
    b = int.b
    d = int.d
    piint = int.piint

    return θ -> begin
        Z = copy(d) 

        @tensor Z[mu,nu,lambda,sigma] += a[mu,nu,alpha,beta] * θ[alpha,beta,lambda,sigma]

        @tensor Z[mu,nu,lambda,sigma] += θ[mu,nu,alpha,beta] * b[alpha,beta,lambda,sigma]
        
        tmp = similar(Z)
        @tensor tmp[mu,nu,gamma,delta] := θ[mu,nu,alpha,beta] * piint[alpha,beta,gamma,delta]
        @tensor Z[mu,nu,lambda,sigma] += tmp[mu,nu,gamma,delta] * θ[gamma,delta,lambda,sigma] 

        @tensor tmp[mu,nu,lambda,sigma] := G_v[alpha,lambda] * θ[mu,nu,alpha,sigma]
        Z += tmp 
        Z += permutedims(tmp, (2, 1, 4, 3))

        @tensor tmp[mu,nu,lambda,sigma] := G_o[mu,alpha] * θ[alpha,nu,lambda,sigma]
        Z -= tmp
        Z -= permutedims(tmp, (2, 1, 4, 3))

        @tensor tmp[mu,nu,lambda,sigma] := j[mu,alpha,lambda,gamma] * (2 * θ[gamma,nu,alpha,sigma] - θ[gamma,nu,sigma,alpha])
        Z += tmp
        Z += permutedims(tmp, (2,1,4,3))

        @tensor tmp[mu,nu,lambda,sigma] := k[mu,alpha,gamma,lambda] * θ[gamma,nu,alpha,sigma]
        Z -= tmp
        Z -= permutedims(tmp, (2,1,4,3))

        @tensor tmp[mu,nu,lambda,sigma] := k[mu,alpha,gamma,sigma] * θ[gamma,nu,lambda,alpha]
        Z -= tmp
        Z -= permutedims(tmp, (2,1,4,3))

        return Z        
    end    
end

export mo2ar_eqns

function mo2ar_eqns(ZMO, slice::Slices)    
    Tbarocc, Tbarvir = slice.Tbarocc, slice.Tbarvir

    @tensor temp_1[mu,j,a,b]         := Tbarocc[mu,i]      * ZMO[i,j,a,b]
	@tensor temp_2[mu,nu,a,b]        := Tbarocc[nu,j]      * temp_1[mu,j,a,b]
	@tensor temp_3[mu,nu,lambda,b]   := Tbarvir[lambda,a]  * temp_2[mu,nu,a,b] 
	@tensor ZAU[mu,nu,lambda,sigma]  := Tbarvir[sigma,b]   * temp_3[mu,nu,lambda,b] 

    return ZAR
end



