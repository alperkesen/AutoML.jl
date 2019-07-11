import CSV
using DataFrames
using Knet: Param, nll, relu, adam, progress!, minibatch, mat, sumabs2, save

DATATYPES = ["String",
             "Int",
             "Float",
             "Binary",
             "Date",
             "Timestamp",
             "Binary category",
             "Category",
             "Image",
             "Text",
             "Array"
             ]

struct Config;
    inputs::Array{Tuple{String, String}, 1};
    outputs::Array{Tuple{String, String},1};
end

function getfeatures(c::Config; ftype="all")
    if ftype == "all"
        features = vcat(c.inputs, c.outputs)
    elseif ftype == "input"
        features = c.inputs
    elseif ftype == "output"
        features = c.outputs
    end
end

mutable struct Model
    config::Config;
    model
    savepath::String
end

Model(config::Config; savedir=SAVEDIR) = Model(config, Chain(), savedir)
Model(inputs::Array{Tuple{String, String},1},
      outputs::Array{Tuple{String, String},1};
      savedir=SAVEDIR) = Model(Config(inputs, outputs), savedir=SAVEDIR)

getfeatures(m::Model; ftype="all") = getfeatures(m.config; ftype=ftype)

function csv2data(csvpath::String)
    df = CSV.read(csvpath)
    fnames = names(df)
    data = Dict(String(fname) => Array(df[fname])
                for fname in fnames)
end

function csv2data(df::DataFrames.DataFrame)
    fnames = names(df)
    data = Dict(String(fname) => Array(df[fname])
                for fname in fnames)
end

function preprocess(data, features)
    preprocessed = Dict()

    for (fname, ftype) in features
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
            preprocessed[fname] = data[fname] .+ 1
        elseif ftype == "Category"
            preprocessed[fname] = doc2ids(data[fname])
        elseif ftype == "Image"
            preprocessed[fname] = [Float64.(readimage(imagepath; dirpath="cifar_100"))
                                   for imagepath in data[fname]]
        elseif ftype == "Text"
            preprocessed[fname] = preprocesstext(data[fname])
        else
            preprocessed[fname] = data[fname]
        end
    end
    preprocessed
end

function train(m::Model, traindata; epochs=1)
    flist = getfeatures(m; ftype="all")
    outflist = getfeatures(m; ftype="output")
    inpflist = getfeatures(m; ftype="input")

    inpftypes = [ftype for (fname, ftype) in inpflist]
    outftypes = [ftype for (fname, ftype) in outflist]
    
    categorical = in("Category", outftypes) || in("Binary Category", outftypes)
    imagemodel = in("Image", inpftypes)
    sequencemodel = in("Text", inpftypes)

    trn = preprocess(traindata, flist)

    xtrn = Dict(fname => trn[fname] for (fname, ftype) in inpflist)
    ytrn = Dict(fname => trn[fname] for (fname, ftype) in outflist)

    if imagemodel
        catdim = length(size(first(values(xtrn))[1])) + 1
        xtrn = Array(cat(collect(values(xtrn))[1]..., dims=catdim))
    elseif sequencemodel
        xtrn = collect(values(xtrn))[1]
    else
        catdim = length(size(first(values(xtrn)))) + 1
        xtrn = Array(cat(values(xtrn)..., dims=catdim)')
    end

    ytrn = slicematrix(hcat(values(ytrn)...))
    dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)

    inputsize = size(xtrn, 1)
    outputsize = size(ytrn, 1)
    hiddensize = 20

    if !categorical
        l1 = Layer(inputsize, hiddensize, 0.01, relu; pdrop=0)
        l2 = Layer(hiddensize, outputsize, 0.01, identity; pdrop=0)
        m.model = Chain(l1, l2)
    elseif imagemodel && categorical
        conv1 = Conv(5, 5, 3, 20)
        l1 = Layer2(3920, 100, 0.01, identity; pdrop=0)
        m.model = Chain2(conv1, l1)
    elseif sequencemodel && categorical
        voclen = 30000
        embeddim = 100
        hiddendim = 20
        outputdim = 2
        m.model = RNNClassifier(voclen, embeddim, hiddendim, outputdim;
                                pdrop=0.5, scale=0.01)
    else
        outputsize = length(unique(ytrn))
        l1 = Layer(inputsize, hiddensize, 0.01, relu; pdrop=0)
        l2 = Layer(hiddensize, outputsize, 0.01, identity; pdrop=0)
        m.model = Chain2(l1, l2)
    end

    progress!(adam(m.model, repeat(dtrn, epochs)))
    save(joinpath(SAVEDIR, "model.jld2"), "model", m.model)
    m, dtrn
end
