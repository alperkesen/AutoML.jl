using Knet: Knet, AutoGrad, adam, progress!, minibatch, save, relu, gpu, KnetArray, load
using Statistics: mean
using DataFrames
using Plots

PARAMS = Dict("batchsize" => 32,
              "lensentence" => 20,
              "vocsize" => 30000)


mutable struct Model
    config::Config;
    name::String
    model
    savepath::String
    vocabulary
    params
    extractor
end

Model(config::Config; name="model", savedir=SAVEDIR, voc=nothing,
      params=PARAMS, extractor=Dict()) = Model(config, name, nothing, savedir,
                                               voc, params, extractor)
Model(inputs::Array{Tuple{String, String},1},
      outputs::Array{Tuple{String, String},1};
      name="model", savedir=SAVEDIR, voc=nothing, params=PARAMS,
      extractor=Dict()) = Model(Config(inputs, outputs), name=name,
                                savedir=SAVEDIR, voc=voc, params=params,
                                extractor=extractor)

getfeatures(m::Model; ftype="all") = getfeatures(m.config; ftype=ftype)
getfnames(m::Model; ftype="all") = getfnames(m.config; ftype=ftype)
getftypes(m::Model; ftype="all") = getftypes(m.config; ftype=ftype)

function iscategorical(m::Model)
    in(CATEGORY, getftypes(m; ftype="output")) ||
        in(BINARYCATEGORY, getftypes(m; ftype="output"))
end

isimagemodel(m::Model) = in(IMAGE, getftypes(m; ftype="input"))
istextmodel(m::Model) = in(TEXT, getftypes(m; ftype="input"))

function savemodel(m::Model, savepath=nothing)
    savepath = savepath == nothing ? joinpath(SAVEDIR, "$(m.name).jld2") : savepath
    save(savepath,
         "config", m.config,
         "model", m.model,
         "params", m.params,
         "vocabulary", m.vocabulary)
end

function loadmodel(m::Model, loadpath::String)
    m.config = load(loadpath, "config")
    m.model = load(loadpath, "model")
    m.params = load(loadpath, "params")
    m.vocabulary = load(loadpath, "vocabulary")
end

function loadmodel(loadpath::String)
    m = Model(Config())
    loadmodel(m, loadpath)
    m
end

function preprocess(m::Model, data)
    preprocessed = Dict()
    featurelist = getfeatures(m; ftype="all")
    commonfeatures = [(fname, ftype) for (fname, ftype) in featurelist
                      if in(fname, keys(data))]
    for (fname, ftype) in commonfeatures
        if ftype == STRING
            preprocessed[fname] = doc2ids(data[fname])
        elseif ftype == INT
            preprocessed[fname] = Int.(data[fname])
        elseif ftype == FLOAT
            if eltype(data[fname]) == String
                preprocessed[fname] = map(x->parse(Float64, x), data[fname])
            elseif eltype(data[fname]) == Int
                preprocessed[fname] = Float64.(data[fname])
            else
                preprocessed[fname] = data[fname]
            end
        elseif ftype == BINARYCATEGORY
            if eltype(data[fname]) != Int
                preprocessed[fname] = map(x->parse(Int64, x), data[fname]) .+ 1
            else
                preprocessed[fname] = data[fname] .+ 1
            end
        elseif ftype == CATEGORY
            preprocessed[fname] = doc2ids(data[fname])
        else
            preprocessed[fname] = data[fname]
        end
    end
    preprocessed
end

function preprocess2(m::Model, data)
    preprocessed = Dict()
    featurelist = getfeatures(m; ftype="all")
    commonfeatures = [(fname, ftype) for (fname, ftype) in featurelist
                      if in(fname, keys(data))]

    if istextmodel(m) && !haskey(m.extractor, "bert")
        m.extractor["bert"] = PretrainedBert()
        vocpath = joinpath(DATADIR, "bert", "bert-base-uncased-vocab.txt")
        m.vocabulary = initializevocab(vocpath)
    end

    if isimagemodel(m) && !haskey(m.extractor, "resnet")
        m.extractor["resnet"] = ResNet()
    end

    for (fname, ftype) in commonfeatures
        if ftype == IMAGE
            resnet = m.extractor["resnet"]
            println("Resnet...")

            preprocessed[fname] = [resnet(joinpath(DATADIR, "cifar_100", path))
                                   for path in data[fname]]
        elseif ftype == TEXT
            bert = m.extractor["bert"]
            docs = read_and_process(data[fname], m.vocabulary)
            inputids, masks, segmentids = preprocessbert(docs)

            println("Bert...")
            preprocessed[fname] = [mean(bert.bert(
                inputids[k],
                segmentids[k];
                attention_mask=masks[k])[:, :, end], dims=2)
                       for k in 1:length(inputids)]
        else
            preprocessed[fname] = data[fname]
        end
    end
    preprocessed
end

function preparedata(m::Model, traindata; output=true)
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}
    trn = preprocess2(m, traindata)

    xtrn = Dict(fname => trn[fname] for fname in getfnames(m; ftype="input"))

    xtrn = vcat([atype(hcat(value...)) for (fname, value) in xtrn]...)

    if output
        ytrn = Dict(fname => trn[fname]
                    for fname in getfnames(m; ftype="output"))
        ytrn = vcat([atype(hcat(value...)) for (fname, value) in ytrn]...)
        ytrn = iscategorical(m) ? Array{Int64}(ytrn) : ytrn

        return xtrn, ytrn
    end

    return xtrn
end

function build(m::Model, inputsize, outputsize)
    chain = iscategorical(m) ? CategoricalChain : LinearChain
    hiddensize = 200

    layer = LinearLayer(inputsize, hiddensize, 0.01, relu; pdrop=0)
    layer2 = LinearLayer(hiddensize, outputsize, 0.01, identity; pdrop=0)
    layer3 = LinearLayer(inputsize, outputsize, 0.01, identity; pdrop=0)
    m.model = chain(layer, layer2)
end

function train(m::Model, traindata::Dict{String, Array{T,1} where T};
               epochs=1, cv=false)
    dtrn = getbatches(m, traindata; batchsize=m.params["batchsize"])

    inputsize = size(dtrn.x, 1)
    outputsize = iscategorical(m) ? length(unique(dtrn.y)) : size(dtrn.y, 1)
    build(m, inputsize, outputsize)

    progress!(adam(m.model, repeat(dtrn, epochs)))
    savemodel(m)
    m, dtrn
end

function train(m::Model, trainpath::String; args...)
    traindata = csv2data(trainpath)
    train(m, traindata; args...)
end

function partialtrain(m::Model, trainpath::String)
    traindata = csv2data(trainpath)
    train(m, traindata; epochs=1)
end

function partialtrain(m::Model, traindata::Dict{String, Array{T,1} where T})
    train(m, traindata; epochs=1)
end

function predictdata(m::Model, example)
    data = Dict(fname => [value] for (fname, value) in example)
    data = preprocess(m, data)
    x = preparedata(m, data; output=false)

    iscategorical(m) ? predict(m.model, x) : m.model(x)
end

function crossvalidate(m::Model, x, y; k=5, epochs=1)
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}
    xfolds, yfolds = kfolds(x, k), kfolds(y, k)
    trainacc, testacc = zeros(2)

    for i=1:k
        foldxtrn = hcat([xfolds[j] for j=1:k if j != i]...)
        foldytrn = hcat([yfolds[j] for j=1:k if j != i]...)

        foldxtst = xfolds[i]
        foldytst = yfolds[i]

        foldxtrn = atype(foldxtrn)
        foldxtst = atype(foldxtst)
 
        dtrn = minibatch(foldxtrn, foldytrn, m.params["batchsize"];
                         shuffle=true)
        dtst = minibatch(foldxtst, foldytst, m.params["batchsize"];
                         shuffle=true)

        progress!(adam(m.model, repeat(dtrn, epochs)))

        trainacc += accuracy(m.model, dtrn)
        testacc += accuracy(m.model, dtst)
    end

    trainacc /= k
    testacc /= k

    println("Train acc:")
    println(trainacc)

    println("Test acc:")
    println(testacc)

    return m
end

function getbatches(m::Model, traindata::Dict{String, Array{T,1} where T};
                    n=1000, batchsize=32)
    batches = []
    trn = preprocess(m, traindata)
    numexamples = length(iterate(values(trn))[1])

    for i=1:n:numexamples
        j = (i+n-1) < numexamples ? i+n-1 : numexamples
        dict = Dict(fname => values[i:j] for (fname, values) in trn)
        @time x,y = preparedata(m, dict)
        d = minibatch(x,y,batchsize;shuffle=false)
        push!(batches, d)
    end

    dx = [d.x for d in batches]
    dy = [d.y for d in batches]

    dfinal = minibatch(hcat(dx...), hcat(dy...), batchsize; shuffle=true)
end

function getbatches(m::Model, df::DataFrames.DataFrame; args...)
    traindata = csv2data(df)
    getbatches(m, traindata; args...)
end
