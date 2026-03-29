export nl_solve

function nl_solve(θ_init::Array{Float64,4}, residual_fun::Function;
                                 maxiter=100, ftol=1e-12, verbose=false)
    θ_shape = size(θ_init)
    function f!(F, θ_flat)
        θ = reshape(θ_flat, θ_shape)
        F .= vec(residual_fun(θ))
    end
    solution = nlsolve(f!, vec(copy(θ_init));
                       method=:anderson,
                       ftol=ftol,
                       xtol=ftol,
                       iterations=maxiter,
                       show_trace=verbose,
                       store_trace=verbose)
    θ_final = reshape(solution.zero, θ_shape)
    return θ_final
end

