export corr_ene

function corr_ene(int::Integrals, θ)
    piint = int.piint
    
    ene = 0.0

    @tensor ene += 2 * piint[lambda,sigma,mu,nu]  * θ[mu,nu,lambda,sigma]
    @tensor ene -=     piint[sigma,lambda,mu,nu]  * θ[mu,nu,lambda,sigma]    
    
    return ene    
end

