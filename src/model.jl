import CSV
using DelimitedFiles
using Knet: Param, nll, relu, adam, progress!, minibatch, mat, sumabs2

PROCESSTYPES = ["String"]

struct Feature; fname::String; ftype::String; end

Feature(t::Tuple{String, String}) = Feature(t...)

struct Config; input::Array{Feature,1}; output::Array{Feature,1}; end

Config(inputs::Array{Tuple{String, String},1},
       outputs::Array{Tuple{String, String},1}) = Config(
           [Feature(inp) for inp in inputs],
           [Feature(out) for out in outputs])

function features(c::Config; ftype="all")
    if ftype == "all"
        features = vcat(c.input, c.output)
    elseif ftype == "input"
        features = c.input
    elseif ftype == "output"
        features = c.output
    end
end

struct Model; config::Config; weights::Array{Any}; end

struct Layer; w; b; f; end

Layer(i::Int, o::Int, scale=0.01, f=relu) = Layer(
    Param(scale * randn(o, i)), Param(zeros(o)), f)

(m::Layer)(x) = m.f.(m.w * mat(hcat(x...)) .+ m.b)
(m::Layer)(x, y) = sumabs2(y - m(x)) / size(y,2)

accuracy(model, x, y) = mean(y' .== map(i->i[1], findmax(Array(model(x)),dims=1)[2]))

accuracy(model, data) = mean(accuracy(model,x,y) for (x,y) in data)

Model(config::Config) = Model(config, Array{Any,1}())
Model(inputs::Array{Tuple{String, String},1}, outputs::Array{Tuple{String, String},1}) = Model(
    Config(inputs, outputs))

features(m::Model; ftype="all") = features(m.config; ftype=ftype)

function save(m::Model)
    println("Saving model...")
    open(joinpath(SAVEDIR, "weights.txt"), "w") do io
        for w in m.weights
            writedlm(io, w)
        end
    end
end

function slicematrix(A::AbstractMatrix{T}) where T
    m, n = size(A)
    B = Vector{T}[Vector{T}(undef, n) for _ in 1:m]
    for i in 1:m
        B[i] .= A[i, :]
    end

    if n == 1
        return permutedims(A[:, 1])
    end

    return B
end

function csv2data(csvfile)
    df = CSV.read(csvfile)
    fnames = String.(names(df))
    data = Dict(fname => Array(eval(Meta.parse("df.$fname")))
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

function preprocess(data, features; processtypes=PROCESSTYPES)
    preprocessed = copy(data)

    for feature in features
        if in(feature.ftype, processtypes)
            preprocessed[feature.fname] = doc2ids(data[feature.fname])
        elseif feature.ftype == "Int"
            preprocessed[feature.fname] = Int.(data[feature.fname])
        elseif feature.ftype == "Float"
            if eltype(data[feature.fname]) != Float64
                preprocessed[feature.fname] = map(x->parse(Float64, x), data[feature.fname])   
            end
        end
    end
    preprocessed
end

function train(m::Model, traindata, epochs=1)
    flist = features(m; ftype="all")
    outflist = features(m; ftype="output")
    inpflist = features(m; ftype="input")

    trn = preprocess(traindata, flist)
    xtrn = Dict(f.fname => trn[f.fname] for f in inpflist)
    ytrn = Dict(f.fname => trn[f.fname] for f in outflist)

    xtrn = slicematrix(hcat(values(xtrn)...))
    ytrn = slicematrix(hcat(values(ytrn)...))

    inpsize = size(xtrn[1], 1)
    outsize = size(ytrn, 1)

    l = Layer(inpsize, outsize, 0.01, identity)

    dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)
    xtrn, ytrn, dtrn, l
end
