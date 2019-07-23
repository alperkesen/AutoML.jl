using Knet: adam, progress!, minibatch, save, relu, gpu, KnetArray
using Statistics: mean
using Plots

PARAMS = Dict("batchsize" => 32,
              "lensentence" => 20)


mutable struct Model
    config::Config;
    model
    savepath::String
    vocabulary
    params
end

Model(config::Config; savedir=SAVEDIR, voc=nothing, params=PARAMS) = Model(
    config, nothing, savedir, voc, params)
Model(inputs::Array{Tuple{String, String},1},
      outputs::Array{Tuple{String, String},1};
      savedir=SAVEDIR, voc=nothing, params=PARAMS) = Model(
          Config(inputs, outputs),
          savedir=SAVEDIR, voc=voc, params=params)

getfeatures(m::Model; ftype="all") = getfeatures(m.config; ftype=ftype)
getfnames(m::Model; ftype="all") = getfnames(m.config; ftype=ftype)
getftypes(m::Model; ftype="all") = getftypes(m.config; ftype=ftype)

function iscategorical(m::Model)
    in("Category", getftypes(m; ftype="output")) ||
        in("Binary Category", getftypes(m; ftype="output"))
end

isimagemodel(m::Model) = in("Image", getftypes(m; ftype="input"))
istextmodel(m::Model) = in("Text", getftypes(m; ftype="input"))


function preprocess(m::Model, data; changevoc=false)
    preprocessed = Dict()
    featurelist = getfeatures(m; ftype="all")
    commonfeatures = [(fname, ftype) for (fname, ftype) in featurelist
                      if in(fname, keys(data))]

    for (fname, ftype) in commonfeatures
        if ftype == "String"
            preprocessed[fname] = doc2ids(data[fname])
        elseif ftype == "Int"
            preprocessed[fname] = Int.(data[fname])
        elseif ftype == "Float"
            if eltype(data[fname]) == String
                preprocessed[fname] = map(x->parse(Float64, x), data[fname])
            elseif eltype(data[fname]) == Int
                preprocessed[fname] = Float64.(data[fname])
            else
                preprocessed[fname] = data[fname]
            end
        elseif ftype == "Binary Category"
            if eltype(data[fname]) != Int
                preprocessed[fname] = map(x->parse(Int64, x), data[fname]) .+ 1
            else
                preprocessed[fname] = data[fname] .+ 1
            end
        elseif ftype == "Category"
            preprocessed[fname] = doc2ids(data[fname])
        elseif ftype == "Image"
            preprocessed[fname] = [Float64.(readimage(
                imagepath; dirpath="cifar_100")) for imagepath in data[fname]]
        elseif ftype == "Text"
            ids, voc = preprocesstext(data[fname]; voc=m.vocabulary,
                                      changevoc=changevoc,
                                      sentencelen=m.params["lensentence"])
            preprocessed[fname] = ids
            m.vocabulary = voc
        else
            preprocessed[fname] = data[fname]
        end
    end
    preprocessed
end

function preparedata(m::Model, traindata; output=true, changevoc=false)
    trn = preprocess(m, traindata; changevoc=changevoc)

    xtrn = Dict(fname => trn[fname] for fname in getfnames(m; ftype="input"))

    if isimagemodel(m)
        catdim = length(size(first(values(xtrn))[1])) + 1
        xtrn = Array(cat(collect(values(xtrn))[1]..., dims=catdim))
    elseif istextmodel(m)
        xtrn = AutoML.slicematrix(collect(hcat(values(xtrn)...)))
    else
        catdim = length(size(first(values(xtrn)))) + 1
        xtrn = Array(cat(values(xtrn)..., dims=catdim)')
        xtrn = float(xtrn)
    end

    if output
        ytrn = Dict(fname => trn[fname]
                    for fname in getfnames(m; ftype="output"))
        ytrn = slicematrix(hcat(values(ytrn)...))

        return xtrn, ytrn
    end

    return xtrn
end

function train(m::Model, traindata::Dict{String, Array{T,1} where T};
               epochs=1, cv=false)
    atype = gpu() >= 0 ? KnetArray : Array
    xtrn, ytrn = preparedata(m, traindata; changevoc=true)

    inputsize = size(xtrn, 1)
    outputsize = size(ytrn, 1)

    if !iscategorical(m)
        xtrn, ytrn = atype(xtrn), atype(ytrn)
        m.model = buildlinearestimator(inputsize, outputsize)
    else
        outputsize = length(unique(ytrn))

        if isimagemodel(m)
            m.model = buildimageclassification(inputsize, outputsize)
        elseif istextmodel(m)
            fdict = getfdict(m.config, ftype="input")
            numtexts = fdict["Text"]

            if numtexts == 1
                m.model = buildsentimentanalysis(outputsize; pdrop=0.5)
            else
                m.model = buildquestionmatching(outputsize; pdrop=0.5)
            end
        else
            println("Building classification model...")
            m.model = buildclassificationmodel(inputsize, outputsize; pdrop=0)
        end
    end

    if cv
        println("Cross validation")
        m = crossvalidate(m, xtrn, ytrn; k=10, epochs=epochs)
        xtrn, ytrn = atype(xtrn), ytrn
        dtrn = minibatch(xtrn, ytrn, m.params["batchsize"]; shuffle=true)

        return m, dtrn
    else
        dtrn = minibatch(xtrn, ytrn, m.params["batchsize"]; shuffle=true)
        progress!(adam(m.model, repeat(dtrn, epochs)))
        save(joinpath(SAVEDIR, "model.jld2"), "model", m.model)
    end
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
