using Dierckx, Optim, LinearAlgebra, QuantEcon

function create_EGM_model_aiyagari(;na = 101, nz = 19)
        # Create NamedTuple
         p = (; 
        ## Number of states ##
         n = na * nz, # Total number of states
         na = na, # Number of capital grid points
         nz = nz, # Number of productivity grid points
        ## Structural parameters ##
         α = 0.33, # Capital share
         β = 0.9, # Discount factor
         A = 0.1, # Productivity 
         γ = 1.0, # Risk aversion
         δ = 0.08, # Depreciation rate
        ## Grid parameters ##
         θ = 4, # Grid expansion parameter
         lb = 0, # Lower bound of capital grid
         ub = 2_000.0, # Upper bound of capital grid
         ρ = 0.9, # Persistence of productivity
         μ = 0.0, # Mean of productivity
         σ = 0.003, # Standard deviation of productivity
        ## Initiliase grids ##
         agrid = LinRange(0.0,1.0,na), # Capital grid
         zgrid = LinRange(0.0,1.0,nz), # Productivity grid
         Π = zeros(Float64, nz, nz), # Transition matrix
        ## Initialise prices and borrowing constraint ##
         w = 0.018920262276054915, # wage
         R_lb = 0.0, # Lower bound of interest rate
         R_ub = 0.0, # Upper bound of interest rate
         R_iter = 1.11, # Initial guess for interest rate
        ## Other ##
         toler_pol = 1e-6, # Tolerance on policies
         toler_prices = 1e-3, # Tolerance on prices
         maxiter_pol = 500, # Maximum number of iterations on policies
         maxiter_prices = 100, # Maximum number of iterations on prices
         print_skip_pol = 500, # Print every x iterations in policy step
         print_skip_prices = 1) # Print every y iterations in value step
        ## Rouwenhorst ##
         mc = rouwenhorst(p.nz, p.μ, p.ρ, p.σ)
         zgrid, Π = mc.state_values, mc.p
        ## Create capital grid ##
         temp_grid = LinRange(0.0,1.0,na)
         agrid = p.lb .+ (p.ub - p.lb) * (temp_grid .^ p.θ)
        ## Return new NamedTuple ##
         p = (p..., agrid = agrid, zgrid = zgrid, Π = Π, R_lb = 1-p.δ, R_ub = 1/p.β)
        return p
end

function utility(c)
    if model.γ == 1.0
        return log(c)
    else
        return (c^(1-model.γ)-1) / (1-model.γ)
    end
end

function marginal_utility(c)
    return c^(-model.γ)
end

function inverse_marginal_utility(u)
    return u^(-1/model.γ)
end

function resources(i, j)
    return (model.R_iter) * model.agrid[i] + model.w * exp(model.zgrid[j])
end

function invariant_distribution(M, O, X, Y, Inv, policy, p)
    (; Π, agrid, n, na, nz) = p
    for i in 1:na
        for j in 1:nz
            if policy[i,j] <= agrid[1]
                M[i,j,1,:] = Π[j,:]
            elseif policy[i,j] >= agrid[end]
                M[i,j,end,:] = Π[j,:]
            else
                index = findfirst(x -> x > policy[i,j], agrid)
                π = (agrid[index] - policy[i,j]) / (agrid[index] - agrid[index-1])
                M[i,j,index-1,:] = π * Π[j,:]
                M[i,j,index,:] = (1-π) * Π[j,:]
            end
        end
    end
    O = reshape(M, n, n)
    for i in 1:na
        O[i,:] = O[i,:] / sum(O[i,:])
    end
    X = reduce(vcat, [I(n) - O', ones(1,n)])
    Y = [zeros(n);1]
    Inv = X \ Y
    Inv[Inv .< 0] .= 0
    Inv = Inv / sum(Inv)
    return Inv
end

function labour_supply(p)
    (; Π, zgrid, nz) = p
    L = zeros(nz)
    L2 = reduce(vcat, [I(nz) - Π', ones(1,nz)])
    L3 = [zeros(nz);1]
    L = L2 \ L3
    L[L .< 0] .= 0
    L = L / sum(L)
    L = L' * zgrid
    return L
end

function initial_guess(p)
    (; na, nz) = p
    a_init = zeros(na,nz)
    for i in 1:na
        for j in 1:nz
            a_init[i,j] = 1/2 * resources(i,j)
        end
    end
    return a_init
end
    
function egm_find_policies(a_init, assets_today, policy, c, rearrange_budget, p)
     # Unpack parameters
     (;β, Π, na, nz, toler_pol, print_skip_pol, maxiter_pol, R_iter, agrid, w, zgrid) = p
     # Initialise matrix
     rearrange_budget = agrid .- w * exp.(zgrid')
     # Set initial error and iteration counter
     error_pol = toler_pol
     iter_pol = 0
     if iter_pol == 0
        println("/// Finding Policy Functions... ///")
     end
    while (error_pol >= toler_pol) && (iter_pol <= maxiter_pol)
        # Calculate new consumption levels
        cons = inverse_marginal_utility.((β * R_iter) .* (marginal_utility.(a_init) * Π'))
        # Use the budget constraint to find today's assets
        assets_today = (rearrange_budget .+ cons) ./ (R_iter)
        for j in 1:nz
            spline = Spline1D(assets_today[:,j], agrid, k=3, bc="extrapolate")
            policy[:,j] = spline.(agrid)
        end
        policy = max.(policy, 0.0)
        # Calculate error
        error_pol = maximum((abs.(policy - a_init)) ./ (1 .+ abs.(a_init)))
        # Keep track of iteration and error
        if iter_pol % print_skip_pol == 0
            println("--------------------")
            println("Iteration: $iter_pol, Error: $error_pol")
        end
        # Go to the next iteration
        a_init = copy(policy)
        iter_pol += 1
    end
    # Check if converged
    if iter_pol == maxiter_pol
        println("--------------------")
        println("Failed to converge in $maxiter_pol iterations")
    else
        println("--------------------")
        println("/// Found Policy Functions ///")
    end
    # Calculate consumption
    c = (R_iter) .* agrid .+ (w .* exp.(zgrid)') .- policy
    # Return consumption and savings policy functions
    return policy, c
end

function find_equilibrium(p)
    # Unpack parameters
    (; A, α, δ, R_lb, R_ub, R_iter, n, na, nz, agrid, zgrid, toler_prices, print_skip_prices, maxiter_prices) = p
    # Initialise matrices for stationary distribution calculation
    M = zeros(na, nz, na, nz)
    O = zeros(n, n)
    X = zeros(n+1, n)
    Y = zeros(n+1)
    Dist = zeros(n)
    L = labour_supply(p)
    wealth = zeros(na, nz)
    # Initialise matrices for EGM step
    a_init = initial_guess(p)
    assets_today = zeros(na, nz)
    policy = copy(assets_today)
    G = zeros(n)
    c = copy(assets_today)
    rearrange_budget = copy(assets_today)
    # Initialise capital supply and demand
    Ks, Kd = 1.0, 1.0
    # Initialise iteration counter and error
    iter_prices = 0
    error_prices = toler_prices + 1.0
    if iter_prices == 0
        println("Starting price iteration")
    end
    while ((error_prices >= toler_prices) && (iter_prices <= maxiter_prices))
        println("//////////////////////")
        println("Price iteration: $iter_prices")
        println("R_lb = $R_lb, R_ub = $R_ub")
        R_iter = (p.R_lb + p.R_ub) / 2
        Kd = ((A^α * L ^ (1 - α)) / (R_iter + δ - 1)) ^ 1 / (1 - α)
        w_iter = (1 - α) * A * (Kd / L) ^ α
        p = (; p..., w = w_iter, R_iter = R_iter)
        policy, c = egm_find_policies(a_init, assets_today, policy, c, rearrange_budget, p)
        Dist = invariant_distribution(M, O, X, Y, Dist, policy, p)
        G = reshape(policy, n, 1)
        Ks = dot(Dist', G)
        diff = Ks - Kd
        error_prices = abs(diff)
        if diff > 0
            p = (; p..., R_ub = R_iter)
        elseif diff < 0
            p = (; p..., R_lb = R_iter)
        end
        if iter_prices % print_skip_prices == 0
            println("%%%%%%%%%%%%%%%")
            println("Price iteration: $iter_prices, R = $R_iter, w = $w_iter, Ks = $Ks, Kd = $Kd, diff = $diff")
            println("%%%%%%%%%%%%%%%")
        end
        a_init = copy(policy)
        iter_prices += 1
    end
    Dist = reshape(Dist, na, nz)
    for i in 1:na
        for j in 1:nz
            wealth[i, j] = p.w * exp(zgrid[j]) + p.R_iter * agrid[i]
        end
    end
    println("r = $R_iter, w = $w_iter")
    println("%%%%%%%%%%%%%%%")
    return Dist, wealth, policy
end

model = create_EGM_model_aiyagari()

dist, wealth, pol = find_equilibrium(model)