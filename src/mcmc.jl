"""
    mcmc(gp::GPBase; kwargs...)

Run MCMC algorithms provided by the Klara package for estimating the hyperparameters of
Gaussian process `gp`.
"""
function mcmc(gp::GPBase; nIter::Int=1000, burn::Int=1, thin::Int=1, ε::Float64=0.1,
              Lmin::Int=5, Lmax::Int=15, lik::Bool=true, noise::Bool=true,
              domean::Bool=true, kern::Bool=true)
    precomp = init_precompute(gp)
    params_kwargs = get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)
    count = 0
    function calc_target(gp::GPBase, θ::AbstractVector) #log-target and its gradient
        count += 1
        try
            set_params!(gp, θ; params_kwargs...)
            update_target_and_dtarget!(gp, precomp; params_kwargs...)
            return true
        catch err
            if !all(isfinite.(θ))
                return false
            elseif isa(err, ArgumentError)
                return false
            elseif isa(err, LinearAlgebra.PosDefException)
                return false
            else
                throw(err)
            end
        end
    end


    θ_cur = get_params(gp; params_kwargs...)
    D = length(θ_cur)
    leapSteps = 0                   #accumulator to track number of leap-frog steps
    post = Array{Float64}(undef, nIter, D)     #posterior samples
    post[1,:] = θ_cur

    @assert calc_target(gp, θ_cur)
    target_cur, grad_cur = gp.target, gp.dtarget

    num_acceptances = 0
    for t in 1:nIter
        θ, target, grad = θ_cur, target_cur, grad_cur

        ν_cur = randn(D)
        ν = ν_cur + 0.5 * ε * grad

        reject = false
        L = rand(Lmin:Lmax)
        leapSteps +=L
        for l in 1:L
            θ += ε * ν
            if  !calc_target(gp,θ)
                reject=true
                break
            end
            target, grad = gp.target, gp.dtarget
            ν += ε * grad
        end
        ν -= 0.5*ε * grad

        if reject
            post[t,:] = θ_cur
        else
            α = target - 0.5 * ν'ν - target_cur + 0.5 * ν_cur'ν_cur
            u = log(rand())

            if u < α
                num_acceptances += 1
                θ_cur = θ
                target_cur = target
                grad_cur = grad
            end
            post[t,:] = θ_cur
        end
    end
    post = post[burn:thin:end,:]
    set_params!(gp, θ_cur; params_kwargs...)
    @printf("Number of iterations = %d, Thinning = %d, Burn-in = %d \n", nIter,thin,burn)
    @printf("Step size = %f, Average number of leapfrog steps = %f \n", ε,leapSteps/nIter)
    println("Number of function calls: ", count)
    @printf("Acceptance rate: %f \n", num_acceptances/nIter)
    return post'
end


"""
    adaptivemcmc(gp::GPBase; kwargs...)

Run MCMC algorithms provided by the Klara package for estimating the hyperparameters of
Gaussian process `gp`.
"""
function adaptivemcmc(gp::G; nIter::Int=1000, burn::Int=1, thin::Int=1, ε::Float64=0.1,
              Lmin::Int=5, Lmax::Int=15, lik::Bool=true, noise::Bool=true,
              domean::Bool=true, kern::Bool=true) where{G<:GPBase}
    precomp = init_precompute(gp)
    params_kwargs = get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)
    count = 0
    function calc_target(gp::GPBase, θ::AbstractVector) #log-target and its gradient
        count += 1
        try
            set_params!(gp, θ; params_kwargs...)
            update_target_and_dtarget!(gp, precomp; params_kwargs...)
            return true
        catch err
            if !all(isfinite.(θ))
                return false
            elseif isa(err, ArgumentError)
                return false
            elseif isa(err, LinearAlgebra.PosDefException)
                return false
            else
                throw(err)
            end
        end
    end


    θ_cur = get_params(gp; params_kwargs...)
    D = length(θ_cur)
    leapSteps = 0                   #accumulator to track number of leap-frog steps
    post = Array{Float64}(undef, nIter, D+1)     #posterior samples
    post[1,:] = vcat(θ_cur,gp.target) # order is: v(n), log lik params (1 for gaussian), mean (dim z/x), kernel(dim x + 1), logpost

    @assert calc_target(gp, θ_cur)
    target_cur, grad_cur = gp.target, gp.dtarget

    num_acceptances = 0
    starttime = time()
    for t in 1:nIter
        if (t % (nIter/10.0) < 1. / nIter)
          runtime = (time() - starttime)/60.0
          println("$t of $nIter: $runtime minutes")
        end
        θ, target, grad = θ_cur, target_cur, grad_cur

        ν_cur = randn(D)
        ν = ν_cur + 0.5 * ε * grad

        reject = false
        L = rand(Lmin:Lmax)
        leapSteps +=L
        for l in 1:L
            θ += ε * ν
            if  !calc_target(gp,θ)
                reject=true
                break
            end
            target, grad = gp.target, gp.dtarget
            ν += ε * grad
        end
        ν -= 0.5*ε * grad

        if reject
            post[t,:] = θ_cur
        else
            α = target - 0.5 * ν'ν - target_cur + 0.5 * ν_cur'ν_cur
            #println(α)
            u = log(rand())

            if u < α
                num_acceptances += 1
                θ_cur = θ
                target_cur = target
                grad_cur = grad
            end
            post[t,:] = vcat(θ_cur,gp.target)
        end
        # adaptive stages ε
        if t < burn
        # adapt ε
            if (num_acceptances / t) < 0.8
                ε *= 0.99
            else
                ε *= 1.01
            end
        end

    end
    post = post[(burn+1):thin:end,:]
    set_params!(gp, θ_cur; params_kwargs...)
    @printf("Number of iterations = %d, Thinning = %d, Burn-in = %d \n", nIter,thin,burn)
    @printf("Step size = %f, Average number of leapfrog steps = %f \n", ε,leapSteps/nIter)
    println("Number of function calls: ", count)
    @printf("Acceptance rate: %f \n", num_acceptances/nIter)
    return post'
end
