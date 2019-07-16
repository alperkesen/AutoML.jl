using Knet: adam, progress!, minibatch, save, relu

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
getfnames(m::Model; ftype="all") = getfnames(m.config; ftype=ftype)
getftypes(m::Model; ftype="all") = getftypes(m.config; ftype=ftype)

function iscategorical(m::Model)
    in("Category", getftypes(m; ftype="output")) ||
        in("Binary Category", getftypes(m; ftype="output"))
end

isimagemodel(m::Model) = in("Image", getftypes(m; ftype="input"))
istextmodel(m::Model) = in("Text", getftypes(m; ftype="input"))


function preparedata(m::Model, traindata)
    featurelist = getfeatures(m; ftype="all")
    trn = preprocess(traindata, featurelist)
    
    xtrn = Dict(fname => trn[fname] for fname in getfnames(m; ftype="input"))
    ytrn = Dict(fname => trn[fname] for fname in getfnames(m; ftype="output"))

    if isimagemodel(m)
        catdim = length(size(first(values(xtrn))[1])) + 1
        xtrn = Array(cat(collect(values(xtrn))[1]..., dims=catdim))
    elseif istextmodel(m)
        xtrn = AutoML.slicematrix(collect(hcat(values(xtrn)...)))
    else
        catdim = length(size(first(values(xtrn)))) + 1
        xtrn = Array(cat(values(xtrn)..., dims=catdim)')
        # trn = AutoML.slicematrix(collect(hcat(values(xtrn)...)))
        xtrn = float(xtrn)
    end

    ytrn = slicematrix(hcat(values(ytrn)...))
    xtrn, ytrn
end

function train(m::Model, traindata; epochs=1, batchsize=32, shuffle=true)
    xtrn, ytrn = preparedata(m, traindata)
    dtrn = minibatch(xtrn, ytrn, batchsize; shuffle=shuffle)

    inputsize = size(xtrn, 1)
    outputsize = size(ytrn, 1)

    if !iscategorical(m)
        m.model = buildlinearestimator(inputsize, outputsize)
    else
        outputsize = length(unique(ytrn))

        if isimagemodel(m)
            m.model = buildimageclassification(inputsize, outputsize)
        elseif istextmodel(m)
            fdict = getfdict(m.config, ftype="input")
            numtexts = fdict["Text"]

            if numtexts == 1
                m.model = buildsentimentanalysis(outputsize)
            else
                m.model = buildquestionmatching(outputsize)
            end
        else
            m.model = buildclassificationmodel(inputsize, outputsize; pdrop=0.5)
        end
    end

    progress!(adam(m.model, repeat(dtrn, epochs)))
    save(joinpath(SAVEDIR, "model.jld2"), "model", m.model)
    m, dtrn
end

