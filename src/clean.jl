function fill_string(data)
    nonempty = [x for x in data if x != "?"]
    data = [x == "?" ? sample(nonempty) : x for x in data]
end


function fill_int(data)
    nonmissing = [x for x in data if x != "?"]
    ints = [parse(Int64, string(x)) for x in nonmissing]
    meanvalue = Int64(round(mean(ints)))

    data = [x == "?" ? meanvalue : x for x in data]
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
    minday = minimum(dates)
    diffs = [(day - minday).value for day in dates]
    meanvalue = Int64(round(mean(diffs)))

    data = [x == "?" ? string(minday + Dates.Day(meanvalue)) : x
            for x in data]
end
