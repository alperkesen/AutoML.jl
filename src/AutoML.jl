module AutoML

export
    Knet,
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

end
