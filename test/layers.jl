using Knet: relu, gpu


@testset "layer" begin
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}

    input = rand(1:10)
    output = rand(1:10)
    layer = AutoML.Layer(input, output)

    batchsize = map(x->2^x, rand(3:10))
    x = atype(randn(input, batchsize))
    y = atype(randn(output, batchsize))

    output = layer(x)
    @test typeof(output) <: atype

    error = layer(x, y)
    @test typeof(error) == typeof(float(rand(1:2)))
end

@testset "linearchain" begin
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}

    input = rand(1:10)
    hidden = rand(1:10)
    output = rand(1:10)

    layer = AutoML.Layer(input, hidden)
    layer2 = AutoML.Layer(hidden, output)
    chain = AutoML.LinearChain(layer, layer2)

    batchsize = map(x->2^x, rand(3:10))
    x = atype(randn(input, batchsize))
    y = atype(randn(output, batchsize))

    output = chain(x)
    @test typeof(output) <: atype

    error = chain(x, y)
    @test typeof(error) == typeof(float(rand(1:2)))
end

@testset "classificationchain" begin
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}

    input = rand(1:10)
    hidden = rand(1:10)
    output = rand(1:10)

    layer = AutoML.Layer(input, hidden)
    layer2 = AutoML.Layer(hidden, output)
    chain = AutoML.CategoricalChain(layer, layer2)

    batchsize = map(x->2^x, rand(3:10))
    numclass = rand(2:10)

    x = atype(randn(input, batchsize))
    y = rand([1:numclass;], 1, batchsize)

    output = chain(x)
    @test typeof(output) <: atype

    error = chain(x, y)
    @test typeof(error) == typeof(float(rand(1:2)))
end
