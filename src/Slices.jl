export T_bar

function T_bar(T, S)    
   	return @tensor Tbar[μ,i] := S[μ,α] * T[α,i]	
end

############################################################################################
export C_occ, C_vir

function C_occ(C, nocc)
    return C[:, 1:nocc]
end

function C_vir(C, nocc, nbasis)
    return C[:,nocc + 1:nbasis]
end

############################################################################################

export T_occ, T_vir

function T_occ(T, nocc)
    return T[:, 1:nocc]
end

function T_vir(T, nocc, nbasis)
    return T[:,nocc + 1:nbasis]
end

############################################################################################

export Tbar_occ, Tbar_vir

function Tbar_occ(Tbar, nocc)
    return Tbar[:, 1:nocc]
end

function Tbar_vir(Tbar, nocc, nbasis)
    return Tbar[:,nocc + 1:nbasis]
end