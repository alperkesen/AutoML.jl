function fill_string(data)
    nonempty = [x for x in data if x != "?"]
    data = [x == "?" ? sample(nonempty) : x for x in data]
end

function fill_int(data)
    nonmissing = [x for x in data if x != "?"]
    ints = [typeof(x) == String ? Int64(parse(Float64, x)) : Int64(x)
            for x in nonmissing]
    meanvalue = Int64(round(mean(ints)))

    data = [x == "?" ? meanvalue : x for x in data]
    data = [typeof(x) == String ? Int64(parse(Float64, x)) : Int64(x)
            for x in data]
end

function fill_bin_category(data)
    nonempty = [x for x in data if x != "?"]
    data = [x == "?" ? sample(nonempty) : x for x in data]
end

function fill_category(data)
    nonempty = [x for x in data if x != "?"]
    data = [x == "?" ? sample(nonempty) : x for x in data]
end

function fill_date(data)
    dates = [Date(x) for x in data if x != "?"]
    data = [x == "?" ? sample(dates) : x for x in data]
end

function fill_timestamp(data)
    timestamps = [DateTime(x) for x in data if x != "?"]
    data = [x == "?" ? sample(timestamps) : x for x in data]
end

function fill_text(data)
    data = replace(data, "?" => "")
end
