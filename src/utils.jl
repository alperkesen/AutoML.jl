import Random
using CSV
using DataFrames
using Images: load

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
    image3d = reshape(hcat(imagergb...)', (size(image, 1), size(image, 2), length(image[1])))
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

function vocabulary(freqs; defaultvoc=nothing, threshold=1)
    if defaultvoc == nothing
        vocabulary = Dict{String, Integer}()
        vocabulary["<pad>"] = 1
        vocabulary["<unk>"] = 2
        pid = 3
    else
        vocabulary = defaultvoc
        pid = length(defaultvoc) + 1
    end

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

function maxsentence(x)
    maximum(length(doc) for doc in x)
end

function addpadding!(x; pos="post", paddinglen=nothing)
    max_length = maxsentence(x)

    if paddinglen != nothing
        max_length = paddinglen
    end

    for doc in x
        while length(doc) < max_length
            if pos == "post"
                push!(doc, 1)
            else
                pushfirst!(doc, 1)
            end
        end
    end
    x
end

function truncate!(x, th=300; pos="post")
    for doc in x
        while length(doc) > th
            if pos == "post"
                pop!(doc)
            else
                popfirst!(doc)
            end
        end
    end
    x
end

function doc2ids(x, voc)
    x = [map(lowercase, doc) for doc in x]
    ids = [map(x -> haskey(voc, x) ? voc[x] : voc["<unk>"], doc)
           for doc in x]
end

function preprocesstext(xtrn; padding="pre", freqthreshold=10, sentencelen=50,
                        defaultvoc=nothing)
    xtrn = tokenize.(xtrn)
    freqs = frequencies(
        skipwords(map(doc->map(lowercase, doc), xtrn)))
    voc = vocabulary(freqs; threshold=freqthreshold, defaultvoc=defaultvoc)

    xtrn_ids = addpadding!(doc2ids(xtrn, voc), pos=padding; paddinglen=sentencelen)
    xtrn_ids = truncate!(xtrn_ids, sentencelen, pos=padding)
    xtrn_ids, voc
end

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

function preprocess(data, features; defaultvoc=nothing)
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
            if eltype(data[fname]) != Int
                preprocessed[fname] = map(x->parse(Int64, x), data[fname]) .+ 1
            else
                preprocessed[fname] = data[fname] .+ 1
            end
        elseif ftype == "Category"
            preprocessed[fname] = doc2ids(data[fname])
        elseif ftype == "Image"
            preprocessed[fname] = [Float64.(readimage(imagepath; dirpath="cifar_100"))
                                   for imagepath in data[fname]]
        elseif ftype == "Text"
            ids, voc = preprocesstext(data[fname]; defaultvoc=defaultvoc)
            preprocessed[fname] = ids
            defaultvoc = voc
        else
            preprocessed[fname] = data[fname]
        end
    end
    preprocessed
end

function splitdata(df::DataFrames.DataFrame; trainprop=0.8)
    examplesize = size(df, 1)
    trainsize = Int(round(examplesize * trainprop))
    indices = Random.shuffle(Random.seed!(0), Vector(1:examplesize))
    trn, tst = df[indices[1:trainsize], :], df[indices[trainsize+1:end], :]
end
