using Knet: relu

function build(m::Model, dtrn::Data)
    chain = iscategorical(m) ? CategoricalChain : LinearChain
    inputsize = size(dtrn.x, 1)
    outputsize = iscategorical(m) ? length(unique(dtrn.y)) : size(dtrn.y, 1)

    hidden = m.params["hidden"]
    pdrop = m.params["pdrop"]

    layer = LinearLayer(inputsize, hidden, relu; pdrop=pdrop)
    layer2 = LinearLayer(hidden, outputsize, identity; pdrop=pdrop)

    m.model = chain(layer, layer2)
end
