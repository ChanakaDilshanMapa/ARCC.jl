export fp_iteration

function fp_iteration(amp_fun, initial_guess, t_fun, max_iter, tol, verbose=false)    
    θ = copy(initial_guess)
    converged = false
    diffs = Float64[]
    
    for iter in 1:max_iter 
        t = t_fun(θ)

        θ_new = amp_fun(t)    
 
        if !all(isfinite.(θ_new))
            if verbose
                println("Diverged (Nan detected) after $iter iterations")
            end
            break
        end

        diff = norm(θ_new - θ)
        push!(diffs, diff)

        if diff > 100
            if verbose
                println("Diverged (Δ > 100) after $iter iterations")
            end
            break
        end

        if verbose
            println("Iteration $iter: Δ = ", diff)
        end        

        θ = θ_new          
        
        if diff < tol
            converged = true
            if verbose
                println("Convergence achieved after $iter iterations")
            end
            break
        end
    end
    
    if !converged
        verbose && println("Warning: Failed to converge after $max_iter iterations")
    end
    
    return θ, diffs
end