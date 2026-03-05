export fock_o
export fock_v

function fock_o(proj::Projectors, fock_ao)
    P = proj.P 
    Pbar = proj.Pbar

    @tensor tmp[mu,tau] := Pbar[mu,rho] * fock_ao[rho,tau]
    @tensor F_o[mu,alpha] := tmp[mu,tau] * P[tau,alpha]

    return F_o
    
end

function fock_v(proj::Projectors, fock_ao)
    Q = proj.Q 
    Qbar = proj.Qbar

    @tensor tmp[lambda,rho] := Qbar[lambda,tau] * fock_ao[rho,tau]
    @tensor F_v[alpha,lambda] := tmp[lambda,rho] * Q[alpha,rho]

    return F_v
    
end

export fock_o_diag , fock_v_diag

function fock_o_diag(fop::FockOperators)
    Fo = fop.Fo
    Fo_diag = diag(Fo)
    
    return Fo_diag
end

function fock_v_diag(fop::FockOperators)
    Fv = fop.Fv
    Fv_diag = diag(Fv)
    
    return Fv_diag
end

export fock_o_off, fock_v_off

function fock_o_off(fop::FockOperators)
    Fo = fop.Fo
    Fo_off = Fo - Diagonal(Fo)
    
    return Fo_off
end

function fock_v_off(fop::FockOperators)
    Fv = fop.Fv
    Fv_off = Fv - Diagonal(Fv)    

    return Fv_off
end



