using Knet: KnetArray, RNN, Param, relu, sumabs2, mat, conv4, pool, dropout, nll, gpu
using Statistics: mean



struct Layer; w; b; f; pdrop; end

Layer(i::Int, o::Int, scale=0.01, f=relu; pdrop=0.5, categorical=true) = Layer(
    Param(scale * randn(o, i)), Param(zeros(o)), f, pdrop)

(l::Layer)(x) = l.f.(l.w * mat(dropout(x, l.pdrop)) .+ l.b)
(l::Layer)(x, y) = sumabs2(y - l(x)) / size(y,2)


struct Layer2; w; b; f; pdrop; end

Layer2(i::Int, o::Int, scale=0.01, f=relu; pdrop=0.5, categorical=true) = Layer2(
    Param(scale * randn(o, i)), Param(zeros(o)), f, pdrop)

(l::Layer2)(x) = l.f.(l.w * mat(dropout(x, l.pdrop)) .+ l.b)
(l::Layer2)(x, y) = nll(l(x), y)


struct Chain2
    layers
    Chain2(layers...) = new(layers)
end

(c::Chain2)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain2)(x, y) = nll(c(x), y)


struct Chain
    layers
    Chain(layers...) = new(layers)
end

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x, y) = loss = sumabs2(y - c(x)) / size(y,2)


struct Conv; w; b; f; pdrop; end

Conv(w1::Int, w2::Int, cx::Int, cy::Int, f=relu; pdrop=0, scale=0.01) =
    Conv(Param(randn(w1, w2, cx, cy) * scale),
         Param(zeros(1, 1, cy, 1)),
         f,
         pdrop)

(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x, c.pdrop)) .+ c.b))


struct RNNClassifier; input; rnn; output; b; pdrop; end

RNNClassifier(input::Int, embed::Int, hidden::Int, output::Int; pdrop=0, scale=0.01, atype=gpu()>=0 ? KnetArray{Float64} : Array{Float64}) =
    RNNClassifier(Param(atype(randn(embed, input) * scale)),
                  RNN(embed, hidden, rnnType=:gru, dataType=Float64),
                  Param(atype(randn(output, hidden) * scale)),
                  Param(atype(zeros(output))),
                  pdrop)

function (c::RNNClassifier)(input)
    embed = c.input[:, permutedims(hcat(input...))]
    embed = dropout(embed, c.pdrop)
    hiddenoutput = c.rnn(embed)
    hiddenoutput = dropout(hiddenoutput, c.pdrop)

    return c.output * hiddenoutput[:,:,end] .+ c.b
end

(c::RNNClassifier)(input,output) = nll(c(input), output)


predict(model, x) = map(i->i[1], findmax(Array(model(x)),dims=1)[2])
accuracy(model, x, y) = mean(y .== predict(model, x))
accuracy(model, data) = mean(accuracy(model,x,y) for (x,y) in data)
