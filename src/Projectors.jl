export ao_projector, complementary_projector

function ao_projector(slice::Slices)
   	return @tensor P[mu,nu] := slice.Cocc[mu,i] * slice.Tocc[nu,i]	
end

function complementary_projector(slice::Slices)  		
   	return @tensor Q[mu,nu] := slice.Tvir[mu,a] * slice.Cvir[nu,a]
end

export ao_projector_bar, complementary_projector_bar

function ao_projector_bar(slice::Slices)
   return @tensor Pbar[mu,nu] := slice.Tbarocc[mu,i] * slice.Cocc[nu,i]    
end

function complementary_projector_bar(slice::Slices)      
    return @tensor Q_bar[lambda,sigma] := slice.Tbarvir[lambda,a] * slice.Cvir[sigma,a]    
end

