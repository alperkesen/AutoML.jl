DATATYPES = ["String",
             "Int",
             "Float",
             "Binary",
             "Date",
             "Timestamp",
             "Binary category",
             "Category",
             "Image",
             "Text",
             "Array"
             ]

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
                   
