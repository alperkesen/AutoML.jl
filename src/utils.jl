using CSV
using Images: load

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
    df = CSV.read(joinpath(DATADIR, "imdb_movie_review", "train.tsv"))
end

function quora_questions()
    df = CSV.read(joinpath(DATADIR, "quora_questions", "train.csv"))
end

function default_of_credit()
    df = CSV.read(joinpath(DATADIR, "default_of_credit", "train.csv"))
end
