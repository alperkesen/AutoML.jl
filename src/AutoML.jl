module AutoML

const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, "..", "models"))
const DATADIR = abspath(joinpath(DIR, "..", "data"))

include("layers.jl")
include("utils.jl")
include("model.jl")
include("examples.jl")

end
