module AutoML

const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, "..", "models"))
const DATADIR = abspath(joinpath(DIR, "..", "data"))

include("config.jl")
include("layers.jl")
include("utils.jl")
include("build.jl")
include("model.jl")
include("examples.jl")
include("resnet.jl")
include("bert.jl")

end
