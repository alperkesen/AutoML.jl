using Knet: Knet, AutoGrad

module AutoML

using ArgParse

export
    Knet,
    AutoGrad,
    Model,
    Config,
    savemodel,
    loadmodel,
    preprocess,
    preprocess2,
    preparedata,
    build,
    hyperoptimization,
    train,
    partialtrain,
    predictdata,
    crossvalidate,
    getbatches,
    csv2data,
    readtext,
    splitdata,
    house_rentals,
    splice_junction,
    cifar100,
    imdb_movie_review,
    quora_questions,
    default_of_credit,
    prediction_of_return,
    train_house_rentals,
    train_gene_sequences,
    train_cifar_100,
    train_imdb,
    train_quora,
    train_default_of_credit,
    train_prediction_of_return


const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, "..", "models"))
const DATADIR = abspath(joinpath(DIR, "..", "data"))

include("config.jl")
include("layers.jl")
include("data.jl")
include("utils.jl")
include("model.jl")
include("build.jl")
include("clean.jl")
include("preprocess.jl")
include("hyperopt.jl")
include("train.jl")
include("examples.jl")
include("resnet.jl")
include("bert.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Automated Machine Learning Tool"

    @add_arg_table s begin
        ("--config"; default=nothing; help="Yaml file for config")
        ("--train"; default=nothing; help="Training dataset file.")
        ("--model_name"; default="model"; help="Model name")
        ("--model_path"; default=nothing; help="Pretrained model path")
        ("--predict"; default=nothing; help="Test dataset file.")
        ("--epochs"; default="1"; help="Number of epochs")
        ("--optimize"; default="true"; help="Hyperparameter optimization")
    end

    isa(args, AbstractString) && (args=split(args))
    println(s.description)
    o = parse_args(args, s; as_symbols=true)

    istrain = o[:train] != nothing
    ispredict = o[:predict] != nothing

    if istrain
        o[:config] == nothing && throw("Config file is missing")

        yamlpath = isabspath(o[:config]) ? o[:config] :
            joinpath(dirname(DIR), o[:config])
        trainpath = isabspath(o[:train]) ? o[:train] :
            joinpath(dirname(DIR), o[:train])
        epochs = parse(Int, o[:epochs])
        optimize = parse(Bool, o[:optimize])
        modelname = o[:model_name]

        inputfeatures, outputfeatures = readyaml(yamlpath)
        model = Model(inputfeatures, outputfeatures; name=modelname)
        model, dtrn = train(model, trainpath;
                            epochs=epochs,
                            showprogress=false,
                            issave=true,
                            optimize=optimize)

        println("Error: ", model.model(dtrn.x, dtrn.y))
        println("Accuracy: ", accuracy(model.model, dtrn))
bo
        return
    elseif ispredict
        o[:model_path] == nothing && throw("Model path is missing")

        modelpath = isabspath(o[:model_path]) ? o[:model_path] :
            joinpath(dirname(DIR), o[:model_path])
        testpath = isabspath(o[:predict]) ? o[:predict] :
            joinpath(dirname(DIR), o[:predict])

        model = loadmodel(modelpath)
        dtst = getbatches(model, testpath; showtime=false)
        println("Error: ", model.model(dtst.x, dtst.y))
        println("Accuracy: ", accuracy(model.model, dtst))
    else
        throw("Use --train or --predict parameters to use package")
    end
end

basename(PROGRAM_FILE) == "AutoML.jl" && main(ARGS)

end
