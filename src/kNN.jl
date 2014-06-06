module kNN
    export knn, kernelregression, predict, knnregression

    using StatsBase
    using Distance
    using NearestNeighbors
    using SmoothingKernels

    include("majority_vote.jl")
    include("classifier.jl")
    include("regress.jl")
    include("knnregress.jl")
end
