using CSV
using DataFrames
using Images: load
using StatsBase: sample

STOPWORDS = ["this", "is", "a", "an", "the",
             '.', '?', '!', ';']


function house_rentals()
    df = CSV.read(joinpath(DATADIR, "home_rentals", "home_rentals.csv"))
end

function splice_junction()
    df = CSV.read(joinpath(DATADIR, "splice_junction_gene_sequences",
                           "splice_junction_gene_sequences.csv"))
end

function cifar_100()
    df = CSV.read(joinpath(DATADIR, "cifar_100", "train.csv"))
end

function imdb_movie_review()
    trn = CSV.read(joinpath(DATADIR, "imdb_movie_review", "train.tsv"))
    tst = CSV.read(joinpath(DATADIR, "imdb_movie_review", "test.tsv"))
    trn, tst
end

function quora_questions()
    trn = CSV.read(joinpath(DATADIR, "quora_questions", "train.csv"))
    trn = dropmissing(trn)

    tst = CSV.read(joinpath(DATADIR, "quora_questions", "test.csv"))
    tst = dropmissing(tst)
    trn, tst
end

function default_of_credit()
    trn = CSV.read(joinpath(DATADIR, "default_of_credit", "train.csv"))
    tst = CSV.read(joinpath(DATADIR, "default_of_credit", "test.csv"))
    trn, tst
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

function readimage(imagepath; dirpath="")
    image = load(joinpath(DATADIR, dirpath, imagepath))
    image3d(image)
end

function image3d(image)
    imagergb = [float([pixel.r, pixel.g, pixel.b]) for pixel in image]
    image3d = reshape(hcat(imagergb...)', (size(image, 1), size(image, 2),
                                           length(image[1])))
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
    pid = length(voc) + 1

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
        freqs = frequencies(
            skipwords(x))
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
    data = Dict{String, Array{T,1} where T}(String(fname) => Array(df[fname])
                for fname in fnames)
end

function csv2data(df::DataFrames.DataFrame)
    fnames = names(df)
    data = Dict{String, Array{T,1} where T}(String(fname) => Array(df[fname])
                for fname in fnames)
end

function cleandata(df::DataFrames.DataFrame)
    df = dropmissing(df)
end

function splitdata(df::DataFrames.DataFrame; trainprop=0.8)
    examplesize = size(df, 1)
    trainsize = Int(round(examplesize * trainprop))
    trnindices = sample(1:examplesize, trainsize, replace=false)
    tstindices = [i for i=1:examplesize if !in(i, trnindices)]
    trn, tst = df[trnindices, :], df[tstindices, :]
end

function kfolds(x, k::Int)
    n = size(x, 2)
    s = n / k
    folds = [x[:, round(Int64, (i-1)*s)+1:min(n,round(Int64, i*s))] for i=1:k]
end
