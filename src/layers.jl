using Knet: Param, relu, sumabs2, mat, conv4, pool, dropout, nll
using Statistics: mean

struct Layer; w; b; f; pdrop; end

Layer(i::Int, o::Int, scale=0.01, f=relu; pdrop=0.5, categorical=true) = Layer(
    Param(scale * randn(o, i)), Param(zeros(o)), f, pdrop)

(l::Layer)(x) = l.f.(l.w * dropout(x, l.pdrop) .+ l.b)

function (l::Layer)(x, y)
    loss = sumabs2(y - l(x)) / size(y,2)
end

struct Layer2; w; b; f; pdrop; end

Layer2(i::Int, o::Int, scale=0.01, f=relu; pdrop=0.5, categorical=true) = Layer2(
    Param(scale * randn(o, i)), Param(zeros(o)), f, pdrop)

(l::Layer2)(x) = l.f.(l.w * dropout(x, l.pdrop) .+ l.b)
(l::Layer2)(x, y) = nll(l(x), y)

struct Chain2
    layers
    Chain2(layers...) = new(layers)
end

(c::Chain2)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain2)(x, y) = nll(c(x), y)



struct Chain
    layers
    Chain(layers...; categorical=true) = new(layers)
end

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

function (c::Chain)(x, y)
    loss = sumabs2(y - c(x)) / size(y,2)
end

struct Conv; w; b; f; pdrop; end

(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x, c.pdrop)) .+ c.b))

Conv(w1::Int, w2::Int, cx::Int, cy::Int, f=relu; pdrop=0, scale=0.01) =
    Conv(Param(randn(w1, w2, cx, cy) * scale), Param(zeros(1, 1, cy, 1)), f, pdrop)

predict(model, x) = map(i->i[1], findmax(Array(model(x)),dims=1)[2])
accuracy(model, x, y) = mean(y .== predict(model, x))
accuracy(model, data) = mean(accuracy(model,x,y) for (x,y) in data)
