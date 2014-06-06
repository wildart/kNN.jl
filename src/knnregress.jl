# K-nearest neighbor regression
immutable kNNRegression{T <: Real}
	t::NearestNeighborTree
	y::Vector{T}
    weights::Symbol
    p::Float64
end


# Weight functions
###################

# Simple average
function uniform(ys, xds)
	return vec(sum(ys,2)./size(ys,2))
end

# Inverse distance weighted average
function idw_average(ys, ds, p::Float64=2.)
	idw = (1./ds).^p
	idw ./= sum(idw)
	return vec(sum(ys.*idw',2))
end

#####################

knnregression{T <: Real}(X::Matrix{T}, y::Vector{T}) = knnregression(X, y, NaiveNeighborTree{T})

function knnregression{T <: Real,
                       K <: NearestNeighborTree}(
                       	X::Matrix{T},
                       	y::Vector{T},
                       	::Type{K};
                       	metric::Metric = Euclidean(),
                        weights::Symbol = :uniform, #:distance
                        p::Float64 = 2.0)
	return kNNRegression(K(X, metric), y, weights, p)
end

function StatsBase.predict(model::kNNRegression, x::Vector, k::Int = 1)
    inds, dists = nearest(model.t, x, k)
    ys = model.y[inds]'
    if model.weights == :uniform
    	return uniform(ys, dists)
    else
    	return idw_average(ys, dists, model.p)
    end
end

function StatsBase.predict!{T <: Real}(ys::AbstractMatrix,
                                       model::kNNRegression,
                                       xs::AbstractMatrix{T},
                                       k::Integer = 1)
    n = length(xs)
    for i in 1:n
        ys[i] = predict(model, xs[:,i], k)
    end
end

function StatsBase.predict{T <: Real}(model::kNNRegression,
                                      xs::AbstractMatrix{T},
                                      k::Integer = 1)
    ys = Array(T, size(xs))
    predict!(ys, model, xs, k)
    return ys
end

function StatsBase.predict!{T <: Real}(ys::Vector,
                                       model::kNNRegression,
                                       xs::AbstractVector{T},
                                       k::Integer = 1)
    n = length(xs)
    for i in 1:n
        ys[i] = predict(model, [xs[i]], k)[1]
    end
end

function StatsBase.predict{T <: Real}(model::kNNRegression,
                                      xs::AbstractVector{T},
                                      k::Integer = 1)
    ys = Array(T, length(xs))
    predict!(ys, model, xs, k)
    return ys
end