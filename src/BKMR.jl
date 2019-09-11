#todo:

# Main GaussianProcess type

mutable struct BKMR{X<:AbstractMatrix,Z<:AbstractMatrix,Y<:AbstractVector{<:Real},M<:Mean,K<:Kernel,L<:Likelihood,
                    CS<:CovarianceStrategy, D<:KernelData} <: GPBase
    # Observation data
    "Input observations: gaussian process function"
    x::X
    "Input observations: fixed effects (mean) function"
    z::Z
    "Output observations"
    y::Y
    # Model
    "Mean object"
    mean::M
    "Kernel object"
    kernel::K
    "Likelihood"
    lik::L
    "Strategy for computing or approximating covariance matrices"
    covstrat::CS
    # Auxiliary data
    "Dimension of inputs: gaussian process function"
    dim::Int
    "Dimension of inputs: fixed effects (mean) function"
    dimz::Int
    "Number of observations: gaussian process function"
    nobs::Int
    "Number of observations: fixed effects (mean) function"
    nobsz::Int
    "Auxiliary observation data (to speed up calculations): fixed effects (mean) function"
    data::D
#to do: need for fixed effect?
    "Latent (whitened) variables - N(0,1)"
    v::Vector{Float64}
    "Mean values"
    μ::Vector{Float64}
    "`(k + exp(2*obsNoise))`"
    cK::PDMat{Float64,Matrix{Float64}}
    "Log-likelihood"
    ll::Float64
    "Gradient of log-likelihood"
    dll::Vector{Float64}
    "Log-target (marginal log-likelihood + log priors)"
    target::Float64
    "Gradient of log-target (gradient of marginal log-likelihood + gradient of log priors)"
    dtarget::Vector{Float64}

    function BKMR{X,Z,Y,M,K,L,CS,D}(x::X, z::Z, y::Y, mean::M, kernel::K, lik::L, covstrat::CS, data::D) where {X,Z,Y,M,K,L,CS,D}
        dim, nobs = size(x)
        dimz, nobsz = size(z)
        length(y) == nobs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        nobsz == nobs || throw(ArgumentError("Input (X) and input (Z) observations must have consistent dimensions."))
        kmr = new{X,Z,Y,M,K,L,CS,D}(x, z, y, mean, kernel, lik, covstrat, dim, dimz, nobs, nobsz, data, zeros(nobs))
        initialise_target!(kmr)
    end
end

function BKMR(x::AbstractMatrix,z::AbstractMatrix, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood, covstrat::CovarianceStrategy)
    data = KernelData(kernel, x, x, covstrat)
    return BKMR{typeof(x),typeof(z),typeof(y),typeof(mean),typeof(kernel),typeof(lik),typeof(covstrat),typeof(data)}(
                x, z, y, mean, kernel, lik, covstrat, data)
end

# todo: default to mean zero GP if no Z is specified
#function BKMR(x::AbstractMatrix,z::AbstractMatrix, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood, covstrat::CovarianceStrategy)
#    data = KernelData(kernel, x, x, covstrat)
#    return GPMC{typeof(x),typeof(y),typeof(mean),typeof(kernel),typeof(lik),typeof(covstrat),typeof(data)}(
#                x, y, mean, kernel, lik, covstrat, data)
#end


"""
    BKMR(x, z, y, mean, kernel, lik)

Fit a Bayesian kernel machine (BKM) to a set of training points. The Gaussian process with
non-Gaussian observations is defined in terms of its user-defined likelihood function,
mean and covariance (kernel) functions.

The non-Gaussian likelihood is handled by a Monte Carlo method. The latent function
values are represented by centered (whitened) variables ``f(x) = m(x) + Lv`` where
``v ∼ N(0, I)`` and ``LLᵀ = K_θ``.

# Arguments:
- `x::AbstractVecOrMat{Float64}`: Input observations: gaussian process function
- `z::AbstractVecOrMat{Float64}`: Input observations: fixed effects
- `y::AbstractVector{<:Real}`: Output observations
- `mean::Mean`: Mean function
- `kernel::Kernel`: Covariance function
- `lik::Likelihood`: Likelihood function
"""
#function BKMR(x::AbstractMatrix,z::AbstractMatrix, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood)
#    covstrat = FullCovariance()
#    return BKMR(x,z, y, mean, kernel, lik, covstrat)
#end

function BKMR(x::X,z::Z, y::Y, mean::M, kernel::K, lik::L) where{X<:AbstractMatrix,Z<:AbstractMatrix,Y<:AbstractVector{<:Real},M<:Mean,K<:Kernel, L<:Likelihood}
    covstrat = FullCovariance()
    return BKMR(x,z, y, mean, kernel, lik, covstrat)
end

BKMR(x::AbstractMatrix,z::AbstractVector, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood) =
    BKMR(x, z', y, mean, kernel, lik)

BKMR(x::AbstractVector,z::AbstractMatrix, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood) =
    BKMR(x', z, y, mean, kernel, lik)

BKMR(x::AbstractVector,z::AbstractVector, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood) =
    BKMR(x', z', y, mean, kernel, lik)

"""
    KMR(x, y, mean::Mean, kernel::Kernel, lik::Likelihood)

Fit a Kernel machine that is defined by its `mean`, its `kernel`, and its likelihood
function `lik` to a set of training points `x` and `y`.

See also: [`GPMC`](@ref)
"""
KMR(x::AbstractVecOrMat{Float64}, z::AbstractVecOrMat{Float64}, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel,
   lik::Likelihood) = BKMR(x,z, y, mean, kernel, lik)

"""
    initialise_ll!(kmr::BKMR)

Initialise the log-likelihood of Kernel machine `kmr`.
"""
function initialise_ll!(kmr::BKMR)
    # log p(Y|v,θ)
    kmr.μ = mean(kmr.mean,kmr.z)
    Σ = cov(kmr.kernel, kmr.x, kmr.x, kmr.data)
    kmr.cK = PDMat(Σ + 1e-6*I)
    F = unwhiten(kmr.cK,kmr.v) + kmr.μ
    kmr.ll = sum(log_dens(kmr.lik,F,kmr.y)) #Log-likelihood
    kmr
end

"""
    update_cK!(kmr::BKMR)

Update the covariance matrix and its Cholesky decomposition of Kernel machine `kmr`.
"""
function update_cK!(kmr::BKMR)
    old_cK = kmr.cK
    Σbuffer = old_cK.mat
    cov!(Σbuffer, kmr.kernel, kmr.x, kmr.x, kmr.data)
    for i in 1:kmr.nobs
        Σbuffer[i,i] += 1e-6 # no logNoise for GPMC
    end
    chol_buffer = old_cK.chol.factors
    copyto!(chol_buffer, Σbuffer)
    chol = cholesky!(Symmetric(chol_buffer))
    kmr.cK = PDMat(Σbuffer, chol)
    kmr
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(kmr::BKMR; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern
        # only need to update the covariance matrix
        # if the covariance parameters have changed
        update_cK!(kmr)
    end
    kmr.μ = mean(kmr.mean,kmr.z)
    F = unwhiten(kmr.cK,kmr.v) + kmr.μ
    kmr.ll = sum(log_dens(kmr.lik,F,kmr.y)) #Log-likelihood
    kmr
end

function get_L_bar!(L_bar::AbstractMatrix, dl_df::AbstractVector, v::AbstractVector, cK::PDMat)
    fill!(L_bar, 0.0)
    BLAS.ger!(1.0, dl_df, v, L_bar)
    tril!(L_bar)
    # ToDo:
    # the following two steps allocates memory
    # and are fickle, reaching into the internal
    # implementation of the cholesky decomposition
    L = cK.chol.L.data
    tril!(L)
    #
    chol_unblocked_rev!(L, L_bar)
    return L_bar
end

struct FullCovMCMCPrecompute <: AbstractGradientPrecompute
    L_bar::Matrix{Float64}
    dl_df::Vector{Float64}
    f::Vector{Float64}
end
function FullCovMCMCPrecompute(nobs::Int)
    buffer1 = Matrix{Float64}(undef, nobs, nobs)
    buffer2 = Vector{Float64}(undef, nobs)
    buffer3 = Vector{Float64}(undef, nobs)
    return FullCovMCMCPrecompute(buffer1, buffer2, buffer3)
end
init_precompute(kmr::BKMR) = FullCovMCMCPrecompute(kmr.nobs)
    
function precompute!(precomp::FullCovMCMCPrecompute, kmr::GPBase) 
    f = unwhiten(kmr.cK, kmr.v)  + kmr.μ
    dl_df = dlog_dens_df(kmr.lik, f, kmr.y)
    precomp.dl_df[:] = dl_df
    precomp.f[:] = f
end
function dll_kern!(dll::AbstractVector, kmr::GPBase, precomp::FullCovMCMCPrecompute, covstrat::CovarianceStrategy)
    L_bar = precomp.L_bar
    get_L_bar!(L_bar, precomp.dl_df, kmr.v, kmr.cK)
    nobs = kmr.nobs
    @inbounds for i in 1:nobs
        L_bar[i,i] *= 2
    end
    # in GPMC, L_bar plays the role of ααinvcKI
    return dmll_kern!(dll, kmr.kernel, kmr.x, kmr.data, L_bar, covstrat)
end
function dll_mean!(dll::AbstractVector, kmr::GPBase, precomp::FullCovMCMCPrecompute)
    dmll_mean!(dll, kmr.mean, kmr.z, precomp.dl_df)
end

"""
     update_dll!(kmr::BKMR, ...)

Update the gradient of the log-likelihood of Kernel machine `kmr`.
"""
function update_dll!(kmr::BKMR, precomp::AbstractGradientPrecompute;
    process::Bool=true, # include gradient components for the process itself
    lik::Bool=true,  # include gradient components for the likelihood parameters
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )

    n_lik_params = num_params(kmr.lik)
    n_mean_params = num_params(kmr.mean)
    n_kern_params = num_params(kmr.kernel)

    kmr.dll = Array{Float64}(undef, process * kmr.nobs + lik * n_lik_params +
                            domean * n_mean_params + kern * n_kern_params)
    precompute!(precomp, kmr)

    i = 1
    if process
        mul!(view(kmr.dll, i:i+kmr.nobs-1), kmr.cK.chol.U, precomp.dl_df)
        i += kmr.nobs
    end
    if lik && n_lik_params > 0
        Lgrads = dlog_dens_dθ(kmr.lik, precomp.f, kmr.y)
        for j in 1:n_lik_params
            kmr.dll[i] = sum(Lgrads[:,j])
            i += 1
        end
    end
    # if domean && n_mean_params > 0
        # Mgrads = grad_stack(kmr.mean, kmr.x)
        # for j in 1:n_mean_params
            # kmr.dll[i] = dot(precomp.dl_df,Mgrads[:,j])
            # i += 1
        # end
    # end
    if domean && n_mean_params>0
        dll_m = @view(kmr.dll[i:i+n_mean_params-1])
        dll_mean!(dll_m, kmr, precomp)
        i += n_mean_params
    end
    if kern
        dll_k = @view(kmr.dll[i:end])
        dll_kern!(dll_k, kmr, precomp, kmr.covstrat)
    end

    kmr
end

function update_ll_and_dll!(kmr::BKMR, precomp::AbstractGradientPrecompute; kwargs...)
    update_ll!(kmr; kwargs...)
    update_dll!(kmr, precomp; kwargs...)
end


"""
    initialise_target!(kmr::BKMR)

Initialise the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Kernel machine `kmr`.
"""
function initialise_target!(kmr::BKMR)
    initialise_ll!(kmr)
    kmr.target = kmr.ll - (sum(abs2, kmr.v) + log2π * kmr.nobs) / 2 +
        prior_logpdf(kmr.lik) + prior_logpdf(kmr.mean) + prior_logpdf(kmr.kernel)
    kmr
end

"""
    update_target!(kmr::BKMR, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Kernel machine `kmr`.
"""
function update_target!(kmr::BKMR; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    update_ll!(kmr; process=process, lik=lik, domean=domean, kern=kern)
    kmr.target = kmr.ll - (sum(abs2, kmr.v) + log2π * kmr.nobs) / 2 +
        prior_logpdf(kmr.lik) + prior_logpdf(kmr.mean) + prior_logpdf(kmr.kernel)
    kmr
end

function update_dtarget!(kmr::BKMR, precomp::AbstractGradientPrecompute; kwargs...)
    update_dll!(kmr, precomp; kwargs...)
    kmr.dtarget = kmr.dll + prior_gradlogpdf(kmr; kwargs...)
    kmr
end

"""
    update_target_and_dtarget!(kmr::BKMR, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Kernel machine `kmr` and its derivative.
"""
function update_target_and_dtarget!(kmr::BKMR, precomp::AbstractGradientPrecompute; kwargs...)
    update_target!(kmr; kwargs...)
    update_dtarget!(kmr, precomp; kwargs...)
end

function update_target_and_dtarget!(kmr::BKMR; kwargs...)
    precomp = init_precompute(kmr)
    update_target_and_dtarget!(kmr, precomp; kwargs...)
end

predict_full(kmr::BKMR, xpred::AbstractMatrix, zpred::AbstractMatrix) = predictMVN(kmr,xpred, zpred, kmr.x, kmr.z, kmr.y, kmr.kernel, kmr.mean, kmr.v, kmr.covstrat, kmr.cK)
"""
    predict_y(kmr::BKMR, x::Union{Vector{Float64}, z::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Kernel machine `kmr` at specfic points which
are given as columns of matrixes `x` and `z`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_y(kmr::BKMR, x::AbstractMatrix, z::AbstractMatrix; full_cov::Bool=false)
    μ, σ2 = predict_f(kmr, x, z; full_cov=full_cov)
    return predict_obs(kmr.lik, μ, σ2)
end

""" Compute predictions using the standard multivariate normal 
    conditional distribution formulae.
"""
function predictMVN(kmr::GPBase,
                   xpred::AbstractMatrix,zpred::AbstractMatrix, 
                   xtrain::AbstractMatrix, ztrain::AbstractMatrix, ytrain::AbstractVector, 
                   kernel::Kernel, meanf::Mean, alpha::AbstractVector,
                   covstrat::CovarianceStrategy, Ktrain::AbstractPDMat)
    crossdata = KernelData(kernel, xtrain, xpred)
    priordata = KernelData(kernel, xpred, xpred)
    Kcross = cov(kernel, xtrain, xpred, crossdata)
    Kpred = cov(kernel, xpred, xpred, priordata)
    mz = mean(meanf, zpred)
    mu, Sigma_raw = predictMVN!(kmr,Kpred, Ktrain, Kcross, mz, alpha)
    return mu, Sigma_raw
end



function predictMVN!(kmr::BKMR,Kxx, Kff, Kfx, mz, αf)
    Lck = whiten!(Kff, Kfx)
    mu = mz + Lck' * αf
    subtract_Lck!(Kxx, Lck)
    return mu, Kxx
end



predict_full(kmr::BKMR, xpred::AbstractMatrix, zpred::AbstractMatrix) = 
  predictMVN(kmr,xpred, zpred, kmr.x, kmr.z, kmr.y, kmr.kernel, kmr.mean, kmr.alpha, kmr.covstrat, kmr.cK)
"""
    predict_full(kmr::GPE, x::Union{Vector{Float64},Matrix{Float64}, z::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Kernel machine `kmr` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""


"""
    predict_f(kmr::GPBase, X::Matrix{Float64}, Z::Matrix{Float64}[; full_cov::Bool = false])

Return posterior mean and variance of the Kernel machine `kmr` at specfic points which are
given as columns of matrixes `X` and `Z`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_f(kmr::GPBase, x::AbstractMatrix, z::AbstractMatrix; full_cov::Bool=false)
    size(x,1) == kmr.dim || throw(ArgumentError("Kernel machine object and input observations (X) do not have consistent dimensions"))
    size(z,1) == kmr.dimz || throw(ArgumentError("Kernel machine object and input observations (Z) do not have consistent dimensions"))
    if full_cov
        return predict_full(kmr, x, z)
    else
        ## Calculate prediction for each point independently
        μ = Array{eltype(x)}(undef, size(x,2))
        σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = predict_full(kmr, x[:,k:k], z[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(diag(sig)[1], 0.0)
        end
        return μ, σ2
    end
end


appendlikbounds!(lb, ub, kmr::BKMR, bounds) = appendbounds!(lb, ub, num_params(kmr.lik), bounds)

function num_params(kmr::BKMR; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n = length(kmr.v)
    lik && (n += num_params(kmr.lik))
    domean && (n += num_params(kmr.mean))
    kern && (n += num_params(kmr.kernel))
    n
end


function get_params(kmr::BKMR; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    append!(params, kmr.v)
    if lik  && num_params(kmr.lik)>0
        append!(params, get_params(kmr.lik))
    end
    if domean && num_params(kmr.mean)>0
        append!(params, get_params(kmr.mean))
    end
    if kern
        append!(params, get_params(kmr.kernel))
    end
    return params
end


function set_params!(kmr::BKMR, hyp::AbstractVector; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n_lik_params = num_params(kmr.lik)
    n_mean_params = num_params(kmr.mean)
    n_kern_params = num_params(kmr.kernel)

    i = 1
    if process
        kmr.v = hyp[1:kmr.nobs]
        i += kmr.nobs
    end
    if lik  && n_lik_params>0
        set_params!(kmr.lik, hyp[i:i+n_lik_params-1]);
        i += n_lik_params
    end
    if domean && n_mean_params>0
        set_params!(kmr.mean, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end
    if kern
        set_params!(kmr.kernel, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end

function prior_gradlogpdf(kmr::BKMR; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    if process
        grad = -kmr.v
    else
        grad = Float64[]
    end
    if lik
        append!(grad, prior_gradlogpdf(kmr.lik))
    end
    if domean
        append!(grad, prior_gradlogpdf(kmr.mean))
    end
    if kern
        append!(grad, prior_gradlogpdf(kmr.kernel))
    end
    return grad
end

#——————————————————————————————————
# Sample random draws from the BKMR
function Random.rand!(kmr::GPBase, x::AbstractMatrix, z::AbstractMatrix, A::DenseMatrix)
    nobs = size(x,2)
    n_sample = size(A,2)

    if kmr.nobs == 0
        # Prior mean and covariance
        μ = mean(kmr.mean, z);
        Σraw = cov(kmr.kernel, x, x);
        Σraw, chol = make_posdef!(Σraw)
        Σ = PDMat(Σraw, chol)
    else
        # Posterior mean and covariance
        μ, Σraw = predict_f(kmr, x, z; full_cov=true)
        Σraw, chol = make_posdef!(Σraw)
        Σ = PDMat(Σraw, chol)
    end
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobs, n_sample)))
end

Random.rand(kmr::GPBase, x::AbstractMatrix, z::AbstractMatrix, n::Int) = rand!(kmr, x,z, Array{Float64}(undef, size(x, 2), n))

# Sample from 1D GPBase
Random.rand(kmr::GPBase, x::AbstractVector, z::AbstractMatrix, n::Int) = rand(kmr, x', z, n)
Random.rand(kmr::GPBase, x::AbstractVector, z::AbstractVector, n::Int) = rand(kmr, x', z', n)

# Generate only one sample from the GPBase and returns a vector
Random.rand(kmr::GPBase, x::AbstractMatrix, z::AbstractMatrix) = vec(rand(kmr,x,z,1))
Random.rand(kmr::GPBase, x::AbstractMatrix, z::AbstractVector) = vec(rand(kmr,x,z',1))
Random.rand(kmr::GPBase, x::AbstractVector, z::AbstractMatrix) = vec(rand(kmr,x',z,1))
Random.rand(kmr::GPBase, x::AbstractVector, z::AbstractVector) = vec(rand(kmr,x',z',1))



function Base.show(io::IO, kmr::BKMR)
    println(io, "KMR Monte Carlo object:")
    println(io, "  N variables in h(x) = ", kmr.dim)
    println(io, "  N fixed effects = ", kmr.dimz)
    println(io, "  Number of observations = ", kmr.nobs)
    println(io, "  Mean function:")
    show(io, kmr.mean, 2)
    println(io, "\n  Kernel:")
    show(io, kmr.kernel, 2)
    println(io, "\n  Likelihood:")
    show(io, kmr.lik, 2)
    if kmr.nobs == 0
        println("\n  No observation data")
    else
        println(io, "\n  Input observations: gaussian process function = ")
        show(io, kmr.x)
        println(io, "\n  Input observations: fixed effects = ")
        show(io, kmr.z)
        print(io,"\n  Output observations = ")
        show(io, kmr.y)
        print(io,"\n  Log-posterior = ")
        show(io, round(kmr.target; digits = 3))
    end
end

Base.show(kmr::BKMR) = Base.show(Base.stdout, kmr)
