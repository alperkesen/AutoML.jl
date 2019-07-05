import CSV
using DataFrames
using Knet: Param, nll, relu, adam, progress!, minibatch, mat, sumabs2

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
    weights::Chain;
    savepath::String
end

Model(config::Config; savedir=SAVEDIR) = Model(config, Chain(), savedir)
Model(inputs::Array{Tuple{String, String},1},
      outputs::Array{Tuple{String, String},1};
      savedir=SAVEDIR) = Model(Config(inputs, outputs), savedir=SAVEDIR)

getfeatures(m::Model; ftype="all") = getfeatures(m.config; ftype=ftype)

function csv2data(csvfile)
    df = CSV.read(csvfile)
    fnames = names(df)
    data = Dict(String(fname) => Array(df[fname])
                for fname in fnames)
end

function doc2ids(data::Array{String,1})
    dict = Dict{String, Int}()
    pid = 1
    
    for x in data
        if !haskey(dict, x)
            dict[x] = pid
            pid += 1
        end
    end
    ids = [dict[x] for x in data]
end

function preprocess(data, features)
    preprocessed = Dict()

    for (fname, ftype) in features
        if ftype == "String"
            preprocessed[fname] = doc2ids(data[fname])
        elseif ftype == "Int"
            preprocessed[fname] = Int.(data[fname])
        elseif ftype == "Float"
            if eltype(data[fname]) != Float64
                preprocessed[fname] = map(x->parse(Float64, x), data[fname])   
            end
        elseif ftype == "Binary"
            preprocessed[fname] = data[fname] + 1
        elseif ftype == "Category"
            preprocessed[fname] = doc2ids(data[fname])
        else
            preprocessed[fname] = data[fname]
        end
    end
    preprocessed
end

function train(m::Model, traindata, epochs=1)
    flist = getfeatures(m; ftype="all")
    outflist = getfeatures(m; ftype="output")
    inpflist = getfeatures(m; ftype="input")

    trn = preprocess(traindata, flist)

    xtrn = Dict(fname => trn[fname] for (fname, ftype) in inpflist)
    ytrn = Dict(fname => trn[fname] for (fname, ftype) in outflist)

    xtrn = hcat(slicematrix(hcat(values(xtrn)...))...)
    ytrn = slicematrix(hcat(values(ytrn)...))

    outtypes = [ftype for (fname, ftype) in outflist]
    categorical = in("Category", outtypes) || in("Binary Category", outtypes)
    
    if !categorical
        inputsize = size(xtrn, 1)
        outputsize = size(ytrn, 1)
        hiddensize = 20

        l1 = Layer(inputsize, hiddensize, 0.01, relu; pdrop=0)
        l2 = Layer(hiddensize, outputsize, 0.01, identity; pdrop=0)
        m.weights = Chain(l1, l2)
    
        dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)
        progress!(adam(m.weights, repeat(dtrn, epochs)))
    else
        inputsize = size(xtrn, 1)
        outputsize = size(ytrn, 1)
        hiddensize = 20
        dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)
        return xtrn, ytrn, dtrn, m
    end
end

