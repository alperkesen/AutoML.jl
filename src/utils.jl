using CSV
using DataFrames
using Images: load
using StatsBase: sample

STOPWORDS = ["this", "is", "a", "an", "the",
             '.', '?', '!', ';']


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

function readimage(imagepath; dirpath="")
    image = load(joinpath(DATADIR, dirpath, imagepath))
    image3d(image)
end

function image3d(image)
    imagergb = [float([pixel.r, pixel.g, pixel.b]) for pixel in image]
    image3d = reshape(hcat(imagergb...)', (size(image, 1), size(image, 2),
                                           length(image[1])))
end

function readtext(filename, dlm=",")
    data = split.(readlines(filename), dlm)
    features = data[1]
    rows = data[2:end]
    numfeatures = length(features)

    fdict = Dict{String, Array{T,1} where T}(
        String(features[i]) => [String.(row[i]) for row in rows]
        for i in 1:numfeatures)
end

function tokenize(text)
    punctuations = ['.', '?', '!', ';', '(', ')']
    text = replace(text, "<br />" => "")

    for p in punctuations
        text = replace(text, "$p" => " $p")
    end

    tokenized = split(text)
end

function skipwords(x; stopwords=STOPWORDS)
    removed = [[word for word in doc if !in(word, stopwords)]
               for doc in x]
end

function doc2ids(data::Array{String,1}; dict=nothing)
    dict = dict == nothing ? Dict{String, Int}() : dict
    pid = length(dict) + 1
    
    for x in data
        if !haskey(dict, x)
            dict[x] = pid
            pid += 1
        end
    end
    ids = [dict[x] for x in data]
    return ids, dict
end

function frequencies(x)
    freqs = Dict{String, Integer}()

    for doc in x
        for word in doc
            freqs[word] = 1 + get(freqs, word, 0)
        end
    end
    freqs
end

function vocabulary(freqs; voc=nothing, threshold=1)
    vocabulary = voc == nothing ? Dict{String, Integer}() : voc
    vocabulary["<pad>"] = 1
    vocabulary["<unk>"] = 2
    pid = length(vocabulary) + 1

    frequency_order = sort([(freqs[word], word)
                            for word in keys(freqs)], rev=true)

    for (f, w) in frequency_order
        if f >= threshold && !in(w, keys(vocabulary))
            vocabulary[w] = pid
            pid += 1
        end
    end

    vocabulary
end

maxsentence(x) = maximum(length(doc) for doc in x)

function addpadding!(x; pos="post", lenpadding=nothing)
    max_length = lenpadding != nothing ? lenpadding : maxsentence(x)

    for doc in x
        while length(doc) < max_length
            pos == "post" ? push!(doc, 1) : pushfirst!(doc, 1)
        end
    end
    x
end

function truncate!(x, th=300; pos="post")
    for doc in x
        while length(doc) > th
            pos == "post" ? pop!(doc) : popfirst!(doc)
        end
    end
    x
end

function doc2ids(x, voc)
    x = [map(lowercase, doc) for doc in x]
    ids = [map(x -> haskey(voc, x) ? voc[x] : voc["<unk>"], doc)
           for doc in x]
end

function preprocesstext(x; padding="pre", freqthreshold=10, lensentence=50,
                        voc=nothing, changevoc=false)
    x = map(doc->map(lowercase, doc), tokenize.(x))

    if voc == nothing || changevoc
        freqs = frequencies(skipwords(x))
        voc = vocabulary(freqs; threshold=freqthreshold, voc=voc)
    end

    ids = addpadding!(doc2ids(x, voc), pos=padding;
                      lenpadding=lensentence)
    ids = truncate!(ids, lensentence, pos=padding)
    ids, voc
end

function csv2data(csvpath::String)
    df = CSV.read(csvpath)
    df = cleandata(df)
    fnames = names(df)
    data = Dict{String, Array{T,1} where T}(String(fname) => Array(replace(
        df[!, fname], missing => "?")) for fname in fnames)
end

function csv2data(df::DataFrames.DataFrame)
    fnames = names(df)
    data = Dict{String, Array{T,1} where T}(String(fname) => Array(replace(
        df[!, fname], missing => "?")) for fname in fnames)
end

function cleandata(df::DataFrames.DataFrame)
    df = dropmissing(df)
end

function splitdata(df::DataFrames.DataFrame, outputcolumn=nothing; trainprop=0.8)
    categorical = outputcolumn != nothing

    if !categorical
        examplesize = size(df, 1)
        trainsize = Int(round(examplesize * trainprop))

        trnindices = sample(1:examplesize, trainsize, replace=false)
        tstindices = [i for i=1:examplesize if !in(i, trnindices)]

        trn, tst = df[trnindices, :], df[tstindices, :]
    else
        classes = unique(outputcolumn)
        divided = [df[outputcolumn .== c, :] for c in classes]
        trainframes, testframes = [], []
        for rows in divided
            examplesize = size(rows, 1)
            trainsize = Int(round(examplesize * trainprop))

            trnindices = sample(1:examplesize, trainsize, replace=false)
            tstindices = [i for i=1:examplesize if !in(i, trnindices)]

            push!(trainframes, rows[trnindices, :])
            push!(testframes, rows[tstindices, :])
        end
        trn = vcat(trainframes...)
        tst = vcat(testframes...)
    end
    trn, tst
end

function splitdata(data::Dict{String, Array{T, 1} where T},
                   outputcolumn=nothing; trainprop=0.8)
    categorical = outputcolumn != nothing

    if !categorical
        examplesize = length(iterate(values(data))[1])
        trainsize = Int(round(examplesize * trainprop))

        trnindices = sample(1:examplesize, trainsize, replace=false)
        tstindices = [i for i=1:examplesize if !in(i, trnindices)]

        trn = Dict(fname => value[trnindices] for (fname, value) in data)
        tst = Dict(fname => value[tstindices] for (fname, value) in data)
    else
        classes = unique(outputcolumn)
        examplesize = length(iterate(values(data))[1])
        classindices = [[i for i=1:examplesize if outputcolumn[i] == c]
                        for c in classes]
        train, test = [], []

        for indices in classindices
            examplesize = length(indices)
            trainsize = Int(round(examplesize * trainprop))

            trnindices = sample(indices, trainsize, replace=false)
            tstindices = [x for x in indices if !in(x, trnindices)]

            newtrn = Dict(fname => value[trnindices] for (fname, value) in data)
            newtst = Dict(fname => value[tstindices] for (fname, value) in data)

            push!(train, newtrn)
            push!(test, newtst)
        end
        trn = Dict{String, Array{T, 1} where T}(fname => vcat(
            [dict[fname] for dict in train]...) for fname in keys(data))
        tst = Dict{String, Array{T, 1} where T}(fname => vcat(
            [dict[fname] for dict in test]...) for fname in keys(data))
    end
    trn, tst
end

function splitdata(d::Data; trainprop=0.8)
    categorical = eltype(d.y) == Int

    if !categorical
        examplesize = size(d.x, 2)
        trainsize = Int(round(examplesize * trainprop))

        trnindices = sample(1:examplesize, trainsize, replace=false)
        tstindices = [i for i in 1:examplesize if !in(i, trnindices)]

        xtrn = dtrn.x[:, trnindices]
        ytrn = dtrn.y[:, trnindices]
        dtrn = minibatch(xtrn, ytrn, d.batchsize; shuffle=true)

        xtst = dtst.x[:, tstindices]
        ytst = dtst.y[:, tstindices]
        dtst = minibatch(xtst, ytst, d.batchsize; shuffle=true)

        return dtrn, dtst
    else
        examplesize = size(d.x, 2)
        classes = unique(d.y)
        classindices = [[i for i=1:examplesize if d.y[i] == c]
                        for c in classes]
        train, test = [], []

        for indices in classindices
            examplesize = length(indices)
            trainsize = Int(round(examplesize * trainprop))

            trnindices = sample(indices, trainsize, replace=false)
            tstindices = [i for i in indices if !in(i, trnindices)]

            xtrn = d.x[:, trnindices]
            ytrn = d.y[:, trnindices]

            xtst = d.x[:, tstindices]
            ytst = d.y[:, tstindices]

            push!(train, (xtrn, ytrn))
            push!(test, (xtst, ytst))
        end

        xtrn = hcat([x[1] for x in train]...)
        ytrn = hcat([x[2] for x in train]...)
        dtrn = minibatch(xtrn, ytrn, d.batchsize; shuffle=true)

        xtst = hcat([x[1] for x in test]...)
        ytst = hcat([x[2] for x in test]...)
        dtst = minibatch(xtst, ytst, d.batchsize; shuffle=true)

        return dtrn, dtst
    end
end

function kfolds(x, k::Int)
    n = size(x, 2)
    s = n / k
    folds = [x[:, round(Int64, (i-1)*s)+1:min(n,round(Int64, i*s))] for i=1:k]
end

function onehot(i, dims)
    v = zeros(dims)
    v[i] = 1

    return v
end

function onehotencode(data)
    dims = length(unique(data))
    ids = doc2ids(data)
    vectors = [onehot(i,  dims) for i in ids]
end

function frequentvalue(array)
    counts = [(count(x->x==i, array), i) for i in unique(array)]
    mostfrequent = maximum(counts)[2]
end
