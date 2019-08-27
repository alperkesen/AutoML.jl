# Train/test dataset paths

const HOMERENTALS = joinpath(DATADIR, "home_rentals", "home_rentals.csv")

const GENESEQUENCES = joinpath(DATADIR, "splice_junction_gene_sequences",
                               "splice_junction_gene_sequences.csv")

const CIFAR100TRAIN = joinpath(DATADIR, "cifar_100", "train.csv")
const CIFAR100TEST = joinpath(DATADIR, "cifar_100", "test.csv")

const IMDBTRAIN = joinpath(DATADIR, "imdb_movie_review", "train.tsv")
const IMDBTEST = joinpath(DATADIR, "imdb_movie_review", "test.tsv")

const QUORATRAIN = joinpath(DATADIR, "quora_questions", "train.csv")
const QUORATEST = joinpath(DATADIR, "quora_questions", "test.csv")

const DOCTRAIN = joinpath(DATADIR, "default_of_credit", "train.csv")
const DOCTEST = joinpath(DATADIR, "default_of_credit", "test.csv")

const PORTRAIN = joinpath(DATADIR, "prediction_of_return", "train.txt")
const PORTEST = joinpath(DATADIR, "prediction_of_return", "test.txt")


function house_rentals()
    df = CSV.read(HOMERENTALS)
end

function splice_junction()
    df = CSV.read(GENESEQUENCES)
end

function cifar100()
    trn = CSV.read(CIFAR100TRAIN)
    tst = CSV.read(CIFAR100TEST)
    trn, tst
end

function imdb_movie_review()
    trn = CSV.read(IMDBTRAIN)
    tst = CSV.read(IMDBTEST)
    trn, tst
end

function quora_questions()
    trn = CSV.read(QUORATRAIN)
    tst = CSV.read(QUORATEST)
    trn, tst
end

function default_of_credit()
    trn = CSV.read(DOCTRAIN)
    tst = CSV.read(DOCTEST)
    trn, tst
end

function prediction_of_return()
    trn = readtext(PORTRAIN)
    tst = readtext(PORTEST)
    trn, tst
end
