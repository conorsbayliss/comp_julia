using Dierckx, LinearAlgebra, Optim, QuantEcon

function create_CRRA_model(;na = 101, nz = 19, γ = 1.0, β = 0.9)

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
          how_iter = 10, # Number of Howard iterations

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

function utility(c, p)
    (; γ) = p
    if γ == 1.0
        return log(c)
    else
        return c^(1-γ)/(1-γ)
    end
end

function resources(agrid, zgrid, j, i, p)
    (; r_iter, w) = p
    return (1+r_iter)*agrid[j] + (w * exp(zgrid[i]))
end

function interpV(agrid, v_slice)
    interp_v = Spline1D(agrid, v_slice, k=3, bc="extrapolate")
    return interp_v
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

function optimise(v_init, v_new, policy, p)
    (; β, na, nz, lb, agrid, zgrid, Π) = p
    Threads.@threads for i in 1:nz
        expected_value = v_init * Π[i,:]
        interpolation = interpV(agrid, expected_value)
        for j in 1:na
            obj(ap) = - (utility(resources(agrid, zgrid, j, i, p) - ap, p) + β * interpolation(ap))
            ub = resources(agrid, zgrid, j, i, p)  
            res = optimize(obj, lb, ub)
            policy[j,i] = res.minimizer
            v_new[j,i] = -res.minimum
        end
    end
    return v_new, policy
end

function howard(v, policy, p)
    (; β, na, nz, how_iter, agrid, zgrid, Π) = p
    for _ in 1:how_iter
        for j in 1:nz
            exp_val = v * Π[j,:]
            interp_e_val = interpV(agrid, exp_val)
            for i in 1:na
                obj(ap) = (utility(resources(agrid, zgrid, i, j, p) - ap, p) + β * interp_e_val(ap))
                v[i,j] = obj(policy[i,j])
            end
        end
    end
    return v
end

function vfi(v_init, policy, p)
    (; max_iter, toler, print_skip, r_iter, w) = p
    v_new = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < max_iter))
        v_new, policy = optimise(v_init, v_new, policy, p)
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

function hpi(v_init, policy, p)
    (; max_iter, toler, print_skip) = p
    v_new = similar(v_init)
    error = toler + 1
    iter = 0
    if iter == 0
        println("Iterating...")
    end
    while ((error > toler) && (iter < max_iter))
        v_new, policy = optimise(v_init, v_new, policy,p)
        v_new = howard(v_new, policy,p)
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

function equilibrium_vfi_crra(p)
    (; n, na, nz, toler_prices, max_iter_prices, print_skip_prices, A, α, ϕ, δ, zgrid, agrid) = p
    M = zeros(na, nz, na, nz)
    O = zeros(n, n)
    X = zeros(n+1,n)
    Y = zeros(n+1)
    Inv = zeros(n)
    v_init = zeros(na,nz)
    policy = similar(v_init)
    wealth = zeros(na,nz)
    capital_demand = []
    capital_supply = []
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
        Kd = ((A^α * L ^ (1-α)) / (r_iter + δ))^1/(1-α)
        push!(capital_demand, Kd)
        w_iter = (1-α) * A * (Kd/L)^α
        Φ = w_iter * (exp(minimum(zgrid))/r_iter)
        if ϕ > 0
            ϕ_iter = min(Φ, exp(minimum(zgrid)))
            p = (; p..., ϕ = ϕ_iter, r_iter = r_iter, w = w_iter)
        else
            p = (; p..., r_iter = r_iter, w = w_iter)
        end
        v_new, policy = vfi(v_init, policy, p)
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
    return v_init, policy, Invariant, wealth, capital_demand, capital_supply
end