const STRING = "String"
const INT = "Int"
const FLOAT = "Float"
const BINARY = "Binary"
const DATE = "Date"
const TIMESTAMP = "Timestamp"
const BINARYCATEGORY = "Binary Category"
const CATEGORY = "Category"
const IMAGE = "Image"
const TEXT = "Text"
const ARRAY = "Array"
const DATATYPES = [STRING, INT, FLOAT, BINARY, DATE, TIMESTAMP,
                   BINARYCATEGORY, CATEGORY, IMAGE, TEXT, ARRAY]

struct Config;
    inputs::Array{Tuple{String, String}, 1};
    outputs::Array{Tuple{String, String},1};
end

function getfeatures(c::Config; ftype="all")
    if ftype == "all"
        features = vcat(c.inputs, c.outputs)
    elseif ftype == "input"
        features = c.inputs
    elseif ftype == "output"
        features = c.outputs
    end
end

function getftypes(c::Config; ftype="all")
    ftypes = [ftype for (fname, ftype) in getfeatures(c; ftype=ftype)]
end

function getfnames(c::Config; ftype="all")
    ftypes = [fname for (fname, ftype) in getfeatures(c; ftype=ftype)]
end

function getfdict(c::Config; ftype="all")
    ftypes = getftypes(c::Config; ftype=ftype)
    Dict(ftype => count(i->i==ftype, ftypes) for ftype in unique(ftypes))
end
                   
