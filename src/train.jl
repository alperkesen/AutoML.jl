function train(m::Model, dtrn::Data; epochs=1, showprogress=true,
               savemodel=false, optimize=false)
    optimize && hyperoptimization(m, dtrn)
    build(m, dtrn)
    dtrn = minibatch(dtrn.x, dtrn.y, m.params["batchsize"]; shuffle=true)

    showprogress ? progress!(adam(m.model, repeat(dtrn, epochs); lr=m.params["lr"])) :
        adam!(m.model, repeat(dtrn, epochs); lr=m.params["lr"])

    savemodel && savemodel(m)
    m, dtrn
end

function train(m::Model, traindata::Dict{String, Array{T,1} where T};
               epochs=1, showprogress=true, savemodel=false)
    dtrn = getbatches(m, traindata; batchsize=m.params["batchsize"])
    train(m, dtrn; epochs=epochs, showprogress=showprogress, savemodel=false)
end

function train(m::Model, trainpath::String; args...)
    m.datapath = dirname(trainpath)
    traindata = csv2data(trainpath)
    train(m, traindata; args...)
end

function partialtrain(m::Model, trainpath::String)
    m.datapath = dirname(trainpath)
    traindata = csv2data(trainpath)
    partialtrain(m, traindata)
end

function partialtrain(m::Model, traindata::Dict{String, Array{T,1} where T})
    dtrn = getbatches(m, traindata; batchsize=m.params["batchsize"])
    partialtrain(m, traindatao)
end

function partialtrain(m::Model, dtrn::Data)
    train(m, dtrn; epochs=1)
end

function crossvalidate(m::Model, dtrn::Data; k=5, epochs=1, showprogress=false)
    indices = sample(1:size(dtrn.x, 2), size(dtrn.x, 2), replace=false)
    dx, dy = dtrn.x[:, indices], dtrn.y[:, indices]
    xfolds, yfolds = kfolds(dx, k), kfolds(dy, k)
    trainloss, testloss = zeros(2)

    for i=1:k
        foldxtrn = hcat([xfolds[j] for j=1:k if j != i]...)
        foldytrn = hcat([yfolds[j] for j=1:k if j != i]...)

        foldxtst = xfolds[i]
        foldytst = yfolds[i]

        dftrn = minibatch(foldxtrn, foldytrn, m.params["batchsize"];
                          shuffle=true)
        dftst = minibatch(foldxtst, foldytst, m.params["batchsize"];
                          shuffle=true)

        showprogress ? progress!(adam(m.model, repeat(dftrn, epochs))) :
            adam!(m.model, repeat(dftrn, epochs))

        trainloss += sum([m.model(x, y) for (x, y) in dftrn])
        testloss += sum([m.model(x, y) for (x, y) in dftst])
    end

    trainloss /= k
    testloss /= k

    showprogress && println("Train loss: $trainloss")
    showprogress && println("Test loss: $testloss")

    return testloss
end
