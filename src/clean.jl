function fill_string(data, value=""; method="mostfrequent")
    nonempty = [x for x in data if x != "?"]

    if method == "sampling"
        data = [x == "?" ? sample(nonempty) : x for x in data]
    elseif method == "constant"
        data = [x == "?" ? value : x for x in data]
    elseif method == "mostfrequent"
        mostfrequent = frequentvalue(nonempty)
        data = [x == "?" ? mostfrequent : x for x in data]
    else
        throw("Undefined filling strategy")
    end
end

function fill_int(data, value=0; method="mean")
    nonmissing = [x for x in data if x != "?"]
    ints = [typeof(x) == String ? Int64(parse(Float64, x)) : Int64(x)
            for x in nonmissing]

    if method == "sampling"
        data = [x == "?" ? sample(ints) : x for x in data]
    elseif method == "mean"
        meanvalue = Int64(round(mean(ints)))
        data = [x == "?" ? meanvalue : x for x in data]
    elseif method == "constant"
        data = [x == "?" ? value : x for x in data]
    elseif method == "mostfrequent"
        mostfrequent = frequentvalue(nonempty)
        data = [x == "?" ? mostfrequent : x for x in data]
    else
        throw("Undefined filling strategy")
    end

    data = [typeof(x) == String ? Int64(parse(Float64, x)) : Int64(x)
            for x in data]
end

function fill_bin_category(data, value=""; method="mostfrequent")
    nonempty = [x for x in data if x != "?"]

    if method == "sampling"
        data = [x == "?" ? sample(nonempty) : x for x in data]
    elseif method == "constant"
        data = [x == "?" ? value : x for x in data]
    elseif method == "mostfrequent"
        mostfrequent = frequentvalue(nonempty)
        data = [x == "?" ? mostfrequent : x for x in data]
    else
        throw("Undefined filling strategy")
    end
end

function fill_category(data, value=""; method="mostfrequent")
    nonempty = [x for x in data if x != "?"]

    if method == "sampling"
        data = [x == "?" ? sample(nonempty) : x for x in data]
    elseif method == "constant"
        data = [x == "?" ? value : x for x in data]
    elseif method == "mostfrequent"
        mostfrequent = frequentvalue(nonempty)
        data = [x == "?" ? mostfrequent : x for x in data]
    else
        throw("Undefined filling strategy")
    end
end

function fill_date(data, value=""; method="mostfrequent")
    dates = [Date(x) for x in data if x != "?"]

    if method == "sampling"
       data = [x == "?" ? sample(dates) : x for x in data]
    elseif method == "constant"
        data = [x == "?" ? value : x for x in data]
    elseif method == "mostfrequent"
        mostfrequent = frequentvalue(dates)
        data = [x == "?" ? mostfrequent : x for x in data]
    else
        throw("Undefined filling strategy")
    end
end

function fill_timestamp(data, value=""; method="mostfrequent")
    timestamps = [DateTime(x) for x in data if x != "?"]

    if method == "sampling"
       data = [x == "?" ? sample(timestamps) : x for x in data]
    elseif method == "constant"
        data = [x == "?" ? value : x for x in data]
    elseif method == "mostfrequent"
        mostfrequent = frequentvalue(timestamps)
        data = [x == "?" ? mostfrequent : x for x in data]
    else
        throw("Undefined filling strategy")
    end
end

function fill_text(data, value=""; method="constant")
    if method == "constant"
        data = replace(data, "?" => "")
    else
        throw("Undefined filling strategy")
    end
end
