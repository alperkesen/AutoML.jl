using Knet: relu, gpu


@testset "layer" begin
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}

    input = rand(1:10)
    output = rand(1:10)
    layer = AutoML.Layer(input, output)

    examplesize = rand(20:30)
    x = atype(randn(input, examplesize))
    y = atype(randn(output, examplesize))

    output = layer(x)
    @test typeof(output) <: atype

    error = layer(x, y)
    @test typeof(error) == typeof(float(rand(1:2)))
end

