using Base.Test
using Distance
using NearestNeighbors
using kNN
using StatsBase
using Winston

# Taken from http://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html
srand(100)
X = sort(5 * rand(1,40), 2)
T = 0:0.002:5
y = vec(sin(X))
y[1:5:end] .+= (0.5 .- rand(8))
k = 5

# kNN regressors
fit = knnregression(X, y) # same as below
#fit = knnregression(X, y, NaiveNeighborTree, metric=Euclidean(), weights=:uniform)
yy1 = predict(fit, T, k)

fit = knnregression(X, y, KDTree, metric=Euclidean(), weights=:distance)
yy2 = predict(fit, T, k)

# Kernel regressor
fit1 = kernelregression(vec(X), y)
yy3 = predict(fit1, T)

fit2 = kernelregression(vec(X), y, kernel = :gaussian)
yy4 = predict(fit2, T)

# Plot data
p = FramedPlot(aspect_ratio=1, xrange=(-0.25,5.25), yrange=(-1.25,1.25))
setattr(p.x2, label="Regression")
d = Points(X, y, kind="filled circle", size=0.3)
setattr(d, label="data")
c1 = Curve(T, yy1, color="red")
setattr(c1, label="knn-reg k=$(k) (uni)")
c2 = Curve(T, yy2, color="green")
setattr(c2, label="knn-reg k=$(k) (idw)")
c3 = Curve(T, yy3, color="blue")
setattr(c3, label="kernel: $(string(fit1.k))")
c4 = Curve(T, yy4, color="cyan")
setattr(c4, label="kernel: $(string(fit2.k))")
l = Legend(.1, .3, {c1,c2,c3,c4,d})
add(p, d, c1, c2, c3, c4, l)