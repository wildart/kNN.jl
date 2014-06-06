module TestKNNRegress
	using Base.Test
	using Distance
	using NearestNeighbors
	using kNN
	using StatsBase

	srand(100)
	X = sort(5 * rand(40))'
	T = 0:0.002:5
	y = vec(sin(X))
	y[1:5:end] .+= (0.5 .- rand(8))
	k = 5

	fit = knnregression(X, y) # NaiveNeighborTree & uniform
	yy1 = predict(fit, T, k)
	@assert length(yy1) == length(T)

	fit = knnregression(X, y, KDTree, metric=Euclidean(), weights=:distance)
	yy2 = predict(fit, T, k)
	@assert length(yy2) == length(T)
end
