using Knet: Knet, AutoGrad, Data, adam, adam!, progress!, minibatch, save, relu,
    gpu, KnetArray, load, sigm, goldensection, hyperband
using Statistics: mean
using DataFrames
using Plots
using Dates

PARAMS = Dict("batchsize" => 32,
              "pdrop" => 0.1,
              "lr" => 0.01,
              "hidden" => 100,
              "optimizer" => adam!
              )


mutable struct Model
    config::Config;
    name::String
    model
    savepath::String
    datapath::String
    vocabulary
    params
    extractor
    fdict
end

Model(config::Config; name="model", savedir=SAVEDIR, datadir="",
      voc=nothing, params=PARAMS, extractor=Dict(), fdict=Dict()) = Model(
          config, name, nothing, savedir, datadir, voc, params, extractor,
          fdict)
Model(inputs::Array{Tuple{String, String},1},
      outputs::Array{Tuple{String, String},1};
      name="model", savedir=SAVEDIR, datadir="", voc=nothing, params=PARAMS,
      extractor=Dict(), fdict=Dict()) = Model(Config(inputs, outputs), name=name,
                                savedir=SAVEDIR, datadir=datadir,
                                voc=voc, params=params,
                                extractor=extractor, fdict=fdict)

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

function predictdata(m::Model, example)
    data = Dict(fname => [value] for (fname, value) in example)
    data = preprocess(m, data)
    x = preparedata(m, data; output=false)

    iscategorical(m) ? predict(m.model, x) : m.model(x)
end

