function process_string(m::Model, data)
    features = [fname for (fname, ftype) in getfeatures(m; ftype="all")
                if in(fname, keys(data)) && ftype == STRING]

    for fname in features
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
        if eltype(data[fname]) != Int
            data[fname] = Int.(data[fname])
        end
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
        dates = [Date(date) for date in data[fname] if date != "?"]
        data[fname] = [[Dates.year(date) / 3000.0,
                        Dates.month(date) / 12.0,
                        Dates.day(date) / 31.0,
                        Dates.dayofweek(date) / 7.0] for date in dates]
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
    println("Resnet...")

    for fname in features
        data[fname] = [resnet(joinpath(m.datapath, path)) for path in data[fname]]
    end
end

