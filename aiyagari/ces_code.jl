using Dierckx, LinearAlgebra, Optim, QuantEcon

function create_model_CES(;na = 101, nz = 15)

    # Create NamedTuple
    p = (;A = 0.01, # total factor productivity
          α = 0.33, # capital share
          δ = 0.08, # depreciation rate
          r_lb = 0.0, # Lower bound on interest rate
          r_ub = 0.1, # Upper bound on interest rate
          r_iter = 0.0, # Interest rate
          ϕ = 0.0, # Borrowing constraint
          w = 0.01, # Wage
          β = 0.9, # Discount factor
          γ = 2.0, # Risk aversion (must be >1)
          nz = nz, # Number of productivity grid points
          ρ = 0.9, # AR(1) persistence
          μ = 0.0, # AR(1) mean
          σ = 0.003, # AR(1) standard deviation
          θ = 3.0, # Expanding grid parameter
          na = na, # Number of capital grid points
          lb = 0.0, # Lower bound of capital grid
          ub = 1_000.0, # Upper bound of capital grid
          max_iter = 1_000, # Maximum number of iterations
          how_iter = 25, # Number of Howard policy iterations
          print_skip = 100, # Print frequency
          toler = 4e-7, # Tolerance
          toler_prices = 1e-3, # tolerance
          max_iter_prices = 100, # maximum no. of iterations
          print_skip_prices = 1, # how often to print
          agrid = zeros(Float64, na), # Capital grid
          zgrid = zeros(Float64, nz), # Productivity grid
          Π = zeros(Float64, nz, nz)) # Transition matrix
          
          # Create capital grid
          temp_grid = LinRange(0, 1, p.na)
          agrid = p.lb .+ (p.ub - p.lb) .* temp_grid .^ p.θ
          
          # Create productivity grid
          mc = rouwenhorst(p.nz, p.μ, p.ρ, p.σ)
          zgrid = exp.(mc.state_values)
          Π = mc.p

          # Return NamedTuple
          p = (;p..., agrid = agrid, zgrid = zgrid, Π = Π)

    return p
end

function utility_CES(c, p)
    (; γ) = p
    return c^(1-γ)
end

function interpV_CES(Avals, v_slice, p)
    (; γ) = p
    interp_v = Spline1D(Avals, v_slice, k=1, bc="extrapolate")
    transformed_interp(x) = interp_v(x)^(1.0-γ)
    return transformed_interp
end

function optimise_CES(Avals, Zvals, v_init, v_new, policy, Π, p)
    (; β, na, nz, lb, γ) = p
    @Threads.threads for j in 1:nz
        expected_value = v_init * Π[j,:]
        interpolation = interpV_CES(Avals, expected_value, p)
        for i in 1:na
            obj_CES(ap) = - (((1-β) * utility_CES(resources(Avals, Zvals, i, j, p) - ap, p) + β * interpolation(ap))^(1/(1-γ)))
            ub = resources(Avals, Zvals, i, j, p)  
            res = optimize(obj_CES, lb, ub)
            policy[i,j] = res.minimizer
            v_new[i,j] = -res.minimum
        end
    end
    return v_new, policy
end

function howard_CES(v, policy, Π, Agrid, Zgrid, p)
    (; β, na, nz, how_iter, γ) = p
    for _ in 1:how_iter
        for j in 1:nz
            exp_val = v * Π[j,:]
            interp_e_val = interpV_CES(Agrid, exp_val, p)
            for i in 1:na
                obj_CES(ap) = (((1-β) * utility_CES(resources(Agrid, Zgrid, i, j, p) - ap, p) + β * interp_e_val(ap)))^(1/(1-γ))
                v[i,j] = obj_CES(policy[i,j])
            end
        end
    end
    return v
end  

function vfi_CES(v_init, v_new, policy, p)
    (; agrid, zgrid, Π, max_iter, toler, print_skip) = p
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < max_iter))
        v_new, policy = optimise_CES(agrid, zgrid, v_init, v_new, policy, Π, p)
        error = maximum(abs.(v_new - v_init) ./ (1 .+ abs.(v_new)))
        v_init .= v_new
        if iter % print_skip == 0
            println("--------------------")
            println("Iteration: $iter, Error: $error")
        end
        iter += 1
    end
    println("--------------------")
    println("Converged in $iter iterations")
    println("--------------------")

    return v_new, policy, agrid

end

function hpi_CES(p)
    (; na, nz, agrid, zgrid, Π, max_iter, toler, print_skip) = p
    v_init = ones(na, nz)
    v_new = similar(v_init)
    policy = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < max_iter))
        v_new, policy = optimise_CES(agrid, zgrid, v_init, v_new, policy, Π, p)
        v_new = howard_CES(v_new, policy, Π, agrid, zgrid, p)
        error = maximum(abs.(v_new - v_init) ./ (1 .+ abs.(v_new)))
        v_init .= v_new
        if iter % print_skip == 0
            println("--------------------")
            println("Iteration: $iter, Error: $error")
        end
        iter += 1
    end
    println("--------------------")
    println("Converged in $iter iterations")
    println("--------------------")

    return v_new, policy, agrid

end