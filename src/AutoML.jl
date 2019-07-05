module AutoML

const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, "..", "models"))

include("layers.jl")
include("model.jl")
include("utils.jl")

end
