export basis_tranform, orthogonalize

function basis_tranform(S, T, A, eris, V)
    n_b = size(T, 2)
    new_S = V' * S * V
    I = UniformScaling(1.0)
    new_S = norm(new_S - I) < 1e-12 ? I(n_b) : new_S
    new_T = V' * T * V
    new_A = V' * A * V

    new_eris = deepcopy(eris)
    for (p, q) in product(1:n_b, 1:n_b)
        new_eris[p, q, :, :] = V' * new_eris[p, q, :, :] * V
    end

    for (r, s) in product(1:n_b, 1:n_b)
        new_eris[:, :, r, s] = V' * new_eris[:, :, r, s] * V
    end
   
    return new_S, new_T, new_A, new_eris
end

function orthogonalize(S, T, A, eris) #ZMS can be done linear scaling
    # Computing S^{-1/2}
    eig_val, eig_vec = eigen(Hermitian(S))
    V = eig_vec * Diagonal(eig_val)^(-1 / 2) * eig_vec'
    return basis_tranform(S, T, A, eris, V)
end