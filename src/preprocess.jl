function process_string(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == STRING]

    for fname in features
        data[fname] = fill_string(data[fname])
        data[fname] = string.(data[fname])
        dict = haskey(m.fdict, fname) ? m.fdict[fname] : nothing
        ids, dict = doc2ids(data[fname]; dict=dict)
        m.fdict[fname] = dict
        data[fname] = ids
    end
end

function process_int(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == INT]

    for fname in features
        data[fname] = fill_int(data[fname])
    end
end

function process_float(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == FLOAT]

    for fname in features
        if eltype(data[fname]) == String
            data[fname] = map(x->parse(Float64, x), data[fname])
        elseif eltype(data[fname]) == Int
            data[fname] = Float64.(data[fname])
        else
            data[fname]
        end
    end
end

function process_bin_category(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == BINARYCATEGORY]

    for fname in features
        data[fname] = fill_bin_category(data[fname])
        data[fname] = string.(data[fname])
        dict = haskey(m.fdict, fname) ? m.fdict[fname] : nothing
        ids, dict = doc2ids(data[fname]; dict=dict)
        m.fdict[fname] = dict
        data[fname] = ids
    end
end

function process_category(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == CATEGORY]

    for fname in features
        data[fname] = fill_category(data[fname])
        data[fname] = string.(data[fname])
        dict = haskey(m.fdict, fname) ? m.fdict[fname] : nothing
        ids, dict = doc2ids(data[fname]; dict=dict)
        m.fdict[fname] = dict
        data[fname] = ids
    end
end

function process_date(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == DATE]

    for fname in features
        data[fname] = fill_date(data[fname])
    end

    alldates = [[Date(date) for date in data[fname]] for fname in features]

    for i in 1:length(alldates)
        for j in i+1:length(alldates)
            fname = "date" * string(i) * string(j)
            println(fname)
            data[fname] = [(alldates[i][k] - alldates[j][k]).value for k in 1:length(alldates[i])]
        end
    end

    for fname in features
        dates = [Date(date) for date in data[fname]]
        data[fname] = [[Dates.year(date),
                        Dates.month(date),
                        Dates.day(date),
                        Dates.dayofweek(date)] for date in dates]
    end
end

function process_timestamp(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == TIMESTAMP]

    for fname in features
        data[fname] = fill_timestamp(data[fname])
        timestamps = [DateTime(ts) for ts in data[fname]]
        data[fname] = [[Dates.year(ts),
                        Dates.month(ts),
                        Dates.day(ts),
                        Dates.hour(ts),
                        Dates.minute(ts),
                        Dates.second(ts)] for ts in timestamps]
    end
end

function process_text(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == TEXT]

    if istextmodel(m) && !haskey(m.extractor, "bert")
        m.extractor["bert"] = PretrainedBert()
        vocpath = joinpath(DATADIR, "bert", "bert-base-uncased-vocab.txt")
        m.vocabulary = initializevocab(vocpath)
    end

    bert = m.extractor["bert"]
    
    println("Bert...")

    for fname in features
        data = fill_text(data, fname)
        data = remove_invalid(data, fname)
        docs = read_and_process(data[fname], m.vocabulary)
        inputids, masks, segmentids = preprocessbert(docs)
        
        data[fname] = [mean(bert.bert(
            inputids[k],
            segmentids[k];
            attention_mask=masks[k])[:, :, end], dims=2)
                       for k in 1:length(inputids)]
    end
end

function process_image(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == IMAGE]

    if isimagemodel(m) && !haskey(m.extractor, "resnet")
        m.extractor["resnet"] = ResNet()
    end

    resnet = m.extractor["resnet"]

    for fname in features
        data = fill_image(data, fname)
        println("Resnet...")
        data[fname] = [resnet(joinpath(m.datapath, path)) for path in data[fname]]
    end
end

function preprocess(m::Model, data::Dict{String, Array{T,1} where T})
    preprocessed = copy(data)
    preprocessed = Dict{String, Array{T,1} where T}(
        fname => value for (fname, value) in preprocessed
        if in(fname, getfnames(m, ftype="all")))

    process_string(m, preprocessed)
    process_int(m, preprocessed)
    process_float(m, preprocessed)
    process_bin_category(m, preprocessed)
    process_category(m, preprocessed)
    process_date(m, preprocessed)

    preprocessed
end

function preprocess2(m::Model, data::Dict{String, Array{T,1} where T})
    preprocessed = copy(data)

    istextmodel(m) && process_text(m, preprocessed)
    isimagemodel(m) && process_image(m, preprocessed)

    preprocessed
end

function preparedata(m::Model, traindata; output=true)
    atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}
    trn = preprocess2(m, traindata)

    xtrn = Dict(fname => value for (fname, value) in trn
                if !in(fname, getfnames(m; ftype="output")))
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

function getbatches(m::Model, traindata::Dict{String, Array{T,1} where T};
                    n=1000, batchsize=32, showtime=true)
    batches = []
    trn = preprocess(m, traindata)
    numexamples = length(iterate(values(trn))[1])

    for i=1:n:numexamples
        j = (i+n-1) < numexamples ? i+n-1 : numexamples
        dict = typeof(traindata)(fname => values[i:j] for (fname, values) in trn)
        x,y = showtime ? @time(preparedata(m, dict)) : preparedata(m, dict)
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

function getbatches(m::Model, trainpath::String; args...)
    traindata = csv2data(trainpath)
    getbatches(m, traindata; args...)
end
