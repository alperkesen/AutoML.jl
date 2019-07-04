import CSV: read
using DelimitedFiles
using Knet: Param, nll

struct Feature; fname::String; ftype::String; end

Feature(t::Tuple{String, String}) = Feature(t...)

struct Config; input::Array{Feature,1}; output::Array{Feature,1}; end


Config(inputs::Array{Tuple{String, String},1},
       outputs::Array{Tuple{String, String},1}) = Config(
           [Feature(inp) for inp in inputs],
           [Feature(out) for out in outputs])

function features(c::Config, ftype="all")
    if ftype == "all"
        features = vcat(c.input, c.output)
    elseif ftype="input"
        features = c.input
    elseif ftype="output"
        features = c.output
    end
end

struct Model; config::Config; weights::Array{Any}; end

struct Layer; w; b; f; end

Layer(i::Int, o::Int, scale=0.01, f=relu) = Layer(
    Param(scale * randn(o, i)), Param(zeros(o)), f)

(m::Layer)(x) = m.f.(m.w * x .+ m.b)
(m::Layer)(x, y) = nll(m(x), y)

Model(config::Config) = Model(config, Array{Any,1}())
Model(inputs::Array{Tuple{String, String},1}, outputs::Array{Tuple{String, String},1}) = Model(
    Config(inputs, outputs))

features(m::Model, ftype="all") = features(m.config, ftype=ftype)

function save(m::Model)
    println("Saving model...")
    open(joinpath(SAVEDIR, "weights.txt"), "w") do io
        for w in m.weights
            writedlm(io, w)
        end
    end
end

function doc2ids(data::Array{String,1})
    dict = Dict{String, Int}
    pid = 1
    
    for x in data
        if !haskey(dict, x)
            dict[x] = pid
            pid += 1
        end
    end
    ids = [dict[x] for x in data]
end

function preprocess(data)
    preprocessed = copy(data)
    features = keys(data)

    for (fname, ftype) in features
        if ftype == String
            preprocessed[f[fname]] = docs2ids(data[f[fname]])
        end
    end
    preprocessed
end

function train(m::Model, train_data::Dict{String, Array{Any, 1}}, epochs=None)
    trn = preprocess(trn)
    trn = hcat(values(trn)...)
    
end
    


