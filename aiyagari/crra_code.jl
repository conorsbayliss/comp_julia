function create_model_CRRA(;na = 101, nz = 15)

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
          γ = 2.0, # Risk aversion
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

model_CRRA = create_model_CRRA()

function utility(c, pars)
    (; γ) = pars
    if γ == 1.0
        return log(c)
    else
        return c^(1-γ)/(1-γ)
    end
end

function ar1(pars)
    (;ρ, μ, σ, nz) = pars
    mc = QuantEcon.rouwenhorst(nz, μ, ρ, σ)
    return mc.p, mc.state_values
end

function exp_grid(pars)
    (; na, θ, lb, ub) = pars
    grid = LinRange(0.0,1.0,na)
    exp_grid = lb .+ (ub - lb) .* grid.^θ
    return exp_grid
end

function resources(Avals, Zvals, j, i, pars)
    (; r_iter, w) = pars
    return (1+r_iter)*Avals[j] + (w*exp(Zvals[i]))
end

function interpV(Avals, v_slice)
    interp_v = Spline1D(Avals, v_slice, k=3, bc="extrapolate")
    return interp_v
end

function optimise(Avals, Zvals, v_init, v_new, policy, Π, pars)
    (; β, na, nz, lb) = pars
    for i in 1:nz
        expected_value = v_init * Π[i,:]
        interpolation = interpV(Avals, expected_value)
        for j in 1:na
            obj(ap) = - (utility(resources(Avals, Zvals, j, i, pars) - ap, pars) + β * interpolation(ap))
            ub = resources(Avals, Zvals, j, i, pars)  
            res = optimize(obj, lb, ub)
            policy[j,i] = res.minimizer
            v_new[j,i] = -res.minimum
        end
    end
    return v_new, policy
end

function howard(v, policy, Π, Avals, Zvals, pars)
    (; β, na, nz, how_iter) = pars
    for _ in 1:how_iter
        for j in 1:nz
            exp_val = v * Π[j,:]
            interp_e_val = interpV(Avals, exp_val)
            for i in 1:na
                obj(ap) = (utility(resources(Avals, Zvals, i, j, pars) - ap, pars) + β * interp_e_val(ap))
                v[i,j] = obj(policy[i,j])
            end
        end
    end
    return v
end   

function vfi(v_init, policy, Π, Zvals, Avals, pars)
    (; maxiter, toler, print_skip, r_iter, w) = pars
    v_new = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < maxiter))
        v_new, policy = optimise(Avals, Zvals, v_init, v_new, policy, Π, pars)
        error = maximum(abs.(v_new - v_init) ./ (1 .+ abs.(v_new)))
        v_init .= v_new
        if iter % print_skip == 0
            println("--------------------")
            println("Iteration: $iter, Error: $error")
        end
        iter += 1
    end
    println("--------------------")
    println("Converged in $iter iterations for r = $r_iter and w = $w")
    println("--------------------")
    return v_new, policy
end

function hpi(v_init, policy, Π, Zvals, Avals, pars)
    (; maxiter, toler, print_skip, r_iter, w) = pars
    v_new = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < maxiter))
        v_new, policy = optimise(Avals, Zvals, v_init, v_new, policy, Π, pars)
        v_new = howard(v_new, policy, Π, Avals, Zvals, pars)
        error = maximum(abs.(v_new - v_init) ./ (1 .+ abs.(v_new)))
        v_init .= v_new
        if iter % print_skip == 0
            println("--------------------")
            println("Iteration: $iter, Error: $error")
        end
        iter += 1
    end
    println("--------------------")
    println("Converged in $iter iterations for r = $r_iter and w = $w")
    println("--------------------")
    return v_new, policy
end