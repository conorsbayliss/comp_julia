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

function resources(i, j, p)
    (; r_iter, w, agrid, zgrid) = p
    return (1+r_iter)*agrid[i] + (w * exp(zgrid[j]))
end

function interpV_CES(v_slice, p)
    (; γ, agrid) = p
    interp_v = Spline1D(agrid, v_slice, k=1, bc="extrapolate")
    transformed_interp(x) = interp_v(x)^(1.0-γ)
    return transformed_interp
end

function invariant_distribution(M, O, X, Y, Inv, policy, p)
    (; n, na, nz, agrid, Π) = p
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
    (; nz, zgrid, Π) = p
    L = zeros(nz)
    L2 = reduce(vcat, [I(nz) - Π', ones(1,nz)])
    L3 = [zeros(nz);1]
    L = L2 \ L3
    L[L .< 0] .= 0
    L = L / sum(L)
    L = L' * zgrid
    return L
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
    (; max_iter, toler, print_skip, r_iter, w) = p
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
    println("Converged in $iter iterations for r = $r_iter and w = $w")
    println("--------------------")
    return v_new, policy
end

function hpi_CES(v_init, policy, p)
    (; max_iter, toler, print_skip, r_iter, w) = p
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
    println("Converged in $iter iterations for r = $r_iter and w = $w")
    println("--------------------")
    return v_new, policy 
end

function equilibrium_vfi_CES(p)
    (; n, na, nz, toler_prices, max_iter_prices, print_skip_prices, A, α, ϕ, δ, zgrid, agrid) = p 
    n = na * nz
    M = zeros(na, nz, na, nz)
    O = zeros(n, n)
    X = zeros(n+1,n)
    Y = zeros(n+1)
    Inv = zeros(n)
    v_init = zeros(na,nz)
    for j in 1:nz
        v_init[:,j] = agrid .^ (1-α)
    end
    policy = similar(v_init)
    wealth = zeros(na, nz)
    capital_demand = []
    capital_supply = []
    interest_rates = []
    L = labour_supply(p)
    Ks, Kd = 1, 1
    iter = 0
    error = toler_prices + 1
    if iter == 0
        println("Iterating on prices...")
    end
    while ((error > toler_prices) && (iter < max_iter_prices))
        println("////////////////////")
        println("Price Iteration: $iter")
        r_iter = (p.r_lb + p.r_ub) / 2
        push!(interest_rates, r_iter)
        Kd = ((A^α * L ^ (1-α)) / (r_iter + δ))^1/(1-α)
        push!(capital_demand, Kd)
        w_iter = (1-α) * A * (Kd/L)^α
        Φ = w_iter * (exp(minimum(zgrid))/r_iter)
        if ϕ > 0
            ϕ_iter = min(Φ, ϕ)
            p = (; p..., ϕ = ϕ_iter, r_iter = r_iter, w = w_iter)
        else
            p = (; p..., r_iter = r_iter, w = w_iter)
        end
        v_new, policy = vfi_CES(v_init, policy, p)
        Invariant = invariant_distribution(M, O, X, Y, Inv, policy, p)
        G = reshape(policy .- ϕ, n, 1)
        Ks = dot(Invariant', G)
        push!(capital_supply, Ks)
        diff = Ks - Kd
        error = abs(diff)
        if diff > 0
            p = (; p..., r_ub = r_iter)
        else
            p = (; p..., r_lb = r_iter)
        end
        if iter % print_skip_prices == 0
            println("%%%%%%%%%%%%%%%%%%%%")
            println("Iter = $iter, Ks = $Ks, Kd = $Kd, diff = $diff")
            println("%%%%%%%%%%%%%%%%%%%%")
        end
        v_init .= v_new
        iter += 1
    end
    Invariant = reshape(Invariant, na, nz)
    for i in 1:na
        for j in 1:nz
            wealth[i,j] = p.w * exp(zgrid[j]) + (1 + p.r_iter) * agrid[i]
        end
    end
    println("r = $(p.r_iter), w = $(p.w)")
    println("%%%%%%%%%%%%%%%%%%%%")
    return v_init, policy, Invariant, wealth, capital_demand, capital_supply, interest_rates
end

function equilibrium_hpi_CES(p)
    (; n, na, nz, toler_prices, max_iter_prices, print_skip_prices, A, α, ϕ, δ, zgrid, agrid) = p
    M = zeros(na, nz, na, nz)
    O = zeros(n, n)
    X = zeros(n+1,n)
    Y = zeros(n+1)
    Inv = zeros(n)
    v_init = zeros(na,nz)
    for j in 1:nz
        v_init[:,j] = agrid .^ (1-α)
    end
    policy = similar(v_init)
    wealth = zeros(na,nz)
    L = labour_supply(p)
    capital_demand = []
    capital_supply = []
    interest_rates = []
    Ks, Kd = 1, 1
    iter = 0
    error = toler_prices + 1
    if iter == 0
        println("Iterating on prices...")
    end
    while ((error > toler_prices) && (iter < max_iter_prices))
        println("////////////////////")
        println("Price Iteration: $iter")
        push!(interest_rates, r_iter)
        r_iter = (p.r_lb + p.r_ub) / 2
        Kd = ((A^α * L ^ (1-α)) / (r_iter + δ))^1/(1-α)
        push!(capital_demand, Kd)
        w_iter = (1-α) * A * (Kd/L)^α
        Φ = w_iter * (exp(minimum(Zvals))/r_iter)
        if ϕ > 0
            ϕ_iter = min(Φ, exp(minimum(Zvals)))
            p = (; p..., ϕ = ϕ_iter, r_iter = r_iter, w = w_iter)
        else
            p = (; p..., r_iter = r_iter, w = w_iter)
        end
        v_new, policy = hpi_CES(v_init, policy, p)
        Invariant = invariant_distribution(M, O, X, Y, Inv, policy, p)
        G = reshape(policy .- ϕ, n, 1)
        Ks = dot(Invariant', G)
        push!(capital_supply, Ks)
        diff = Ks - Kd
        error = abs(diff)
        if diff > 0
            p = (; p..., r_ub = r_iter)
        else
            p = (; p..., r_lb = r_iter)
        end
        if iter % print_skip_prices == 0
            println("%%%%%%%%%%%%%%%%%%%%")
            println("Iter = $iter, Ks = $Ks, Kd = $Kd, diff = $diff")
            println("%%%%%%%%%%%%%%%%%%%%")
        end
        v_init .= v_new
        iter += 1
    end
    Invariant = reshape(Invariant, na, nz)
    for i in 1:na
        for j in 1:nz
            wealth[i,j] = p.w * exp(zgrid[j]) + (1 + p.r_iter) * agrid[i]
        end
    end
    println("r = $(p.r_iter), w = $(p.w)")
    println("%%%%%%%%%%%%%%%%%%%%")
    return v_init, policy, Invariant, wealth
end

function gini_coeff(distribution, wealth, p)
    (; n, na, nz) = p
    n = na * nz
    gini_w = zeros(n)
    gini_dist = zeros(n)
    lorenz = zeros(n)
    wealth = reshape(wealth, n)
    distribution = reshape(distribution, n)
    sorted_wealth = sort(wealth, rev = false)
    sorted_indices = sortperm(wealth)
    sorted_distribution = distribution[sorted_indices]
    for i in 1:n
        gini_w[i] = sorted_wealth[i] * sorted_distribution[i]
        if i == 1
            lorenz[i] = gini_w[i]
            gini_dist[i] = sorted_distribution[i]
        else
            lorenz[i] = lorenz[i-1] + gini_w[i]
            gini_dist[i] = gini_dist[i-1] + sorted_distribution[i]
        end
    end
    lorenz = lorenz / lorenz[end]
    gini_dist = gini_dist / gini_dist[end]
    area_under_lorenz = 0.0
    for i in 2:length(lorenz)
        area_under_lorenz += 0.5 * (lorenz[i] + lorenz[i-1]) * (gini_dist[i] - gini_dist[i-1])
    end
    Gini_round = 1 - 2 * area_under_lorenz
    Gini_round = round(Gini_round, digits = 4)
    return lorenz, gini_dist, Gini_round
end