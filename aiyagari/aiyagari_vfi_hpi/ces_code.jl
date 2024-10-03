### Note, when running Aiyagari with CES utility, ###
### some of the required functions are in         ###
### crra_code.jl since they are common to both    ###
### implementations. Specifically, the functions: ###
### - resources,                                  ###
### - labour_supply, and                          ###
### - invariant_distribution                      ###

using Dierckx, LinearAlgebra, Optim, QuantEcon

function create_CES_model(;na = 101, 
                           nz = 19, 
                           γ = 2.0, 
                           β = 0.9,
                           how_iter = 25)

    p = (;
          ### N ###
          n = na * nz,
          na = na,
          nz = nz,

          ### Deep structural parameters ###
          α = 0.33, # Capital share
          A = 0.01, # TFP
          δ = 0.08, # Depreciation rate
          β = β, # Discount factor
          γ = γ, # Risk aversion

          ### Prices ###
          r_lb = 0.0, # Lower bound for r
          r_ub = 0.00, # Upper bound for r
          r_iter = 0.0, # Initialise r
          w = 1.0, # Wage

          ### AR(1) process ###
          ρ = 0.9, # Persistence
          μ = 0.0, # Mean
          σ = 0.003, # Standard deviation

          ### Grids ###
          θ = 4.0, # Grid expansion parameter
          ϕ = 0.0, # Borrowing constraint
          lb = 0.0, # Lower bound for assets
          ub = 1_000.0, # Upper bound for assets

          ### Other, VFI/HPI ###
          toler = 4e-7, # Tolerance for VFI/hpi
          print_skip = 100, # Print every n iterations
          max_iter = 1_000, # Maximum number of iterations
          how_iter = how_iter, # Number of Howard iterations

          ### Other, prices ###
          toler_prices = 1e-3, # Tolerance for prices
          max_iter_prices = 100, # Maximum number of iterations over prices
          print_skip_prices = 1, # Print every n iterations

          ### Create matrices ###
          agrid = zeros(Float64, na), # Asset grid
          zgrid = zeros(Float64, nz), # Productivity grid
          Π = zeros(Float64, nz, nz)) # Transition matrix

    ### Create capital grid ###
    temp_grid  = LinRange(0.0, 1.0, p.na)
    agrid = p.lb .+ (p.ub - p.lb) * (temp_grid .^ p.θ) 

    ### Create productivity grid ###
    mc = rouwenhorst(p.nz, p.μ, p.ρ, p.σ)
    zgrid = mc.state_values
    Π = mc.p

    ### Price bounds ###
    r_lb = 0.0
    r_ub = 1/p.β - 1
    ### Return new NamedTuple ###
    p = (;p..., agrid = agrid, zgrid = zgrid, Π = Π, r_lb = r_lb, r_ub = r_ub)

    return p

end

function utility_CES(c, p)
    (; γ) = p
    return c^(1-γ)
end

function interpV_CES(v_slice, p)
    (; γ, agrid) = p
    interp_v = Spline1D(agrid, v_slice, k=1, bc="extrapolate")
    transformed_interp(x) = interp_v(x)^(1.0-γ)
    return transformed_interp
end

function optimise_CES(v_init, v_new, policy, p)
    (; β, na, nz, lb, γ, Π) = p
    Threads.@threads for j in 1:nz
        expected_value = v_init * Π[j,:]
        interpolation = interpV_CES(expected_value, p)
        for i in 1:na
            obj_CES(ap) = - (((1-β) * utility_CES(resources(i, j, p) - ap, p) + β * interpolation(ap))^(1/(1-γ)))
            ub = resources(i, j, p)  
            res = optimize(obj_CES, lb, ub)
            policy[i,j] = res.minimizer
            v_new[i,j] = -res.minimum
        end
    end
    return v_new, policy
end

function howard_CES(v, policy, p)
    (; β, na, nz, how_iter, γ, Π) = p
    v_how = similar(v)
    for _ in 1:how_iter
        for j in 1:nz
            exp_val = v * Π[j,:]
            interp_e_val = interpV_CES(exp_val, p)
            for i in 1:na
                obj_CES(ap) = (((1-β) * utility_CES(resources(i, j, p) - ap, p) + β * interp_e_val(ap)))^(1/(1-γ))
                v_how[i,j] = obj_CES(policy[i,j])
            end
        end
    end
    return v
end

function vfi_CES(v_init, policy, p)
    (; max_iter, toler, print_skip) = p
    v_new = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < max_iter))
        v_new, policy = optimise_CES(v_init, v_new, policy, p)
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
    return v_new, policy
end

function hpi_CES(v_init, policy, p)
    (; max_iter, toler, print_skip) = p
    v_new = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < max_iter))
        v_new, policy = optimise_CES(v_init, v_new, policy, p)
        v_new = howard_CES(v_new, policy, p)
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
    return v_new, policy 
end