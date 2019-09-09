function goldensectionopt(m::Model, dtrn::Data; showloss=true, cv=false)
    neval = 0
    dt, dv = splitdata(dtrn)

    function f(x)
        neval += 1
        lr, hidden, pdrop, batchsize = xform(x)

        if hidden < 10000 && 0 <= pdrop <= 1 && 0 < batchsize < 2048
            m.params["lr"] = lr
            m.params["hidden"] = hidden
            m.params["pdrop"] = pdrop
            m.params["batchsize"] = batchsize

            if cv
                loss = crossvalidate(m, dtrn; showprogress=true)
            else
                train(m, dt; showprogress=false, epochs=1, savemodel=false)
                loss = m.model(dv.x, dv.y)
            end
        else
            loss = NaN
        end

        config = xform(x)
        showloss && println("Loss: $loss, Config: $config")

        return loss
    end

    function xform(x)
        lr, hidden, pdrop, batchsize = exp.(x) .* [0.001, 100.0, 0.5, 64]
        hidden = ceil(Int, hidden)
        batchsize = ceil(Int, batchsize)
        (lr, hidden, pdrop, batchsize)
    end

    (f0, x0) = goldensection(f, 4)
    lr, hidden, pdrop, batchsize = xform(x0)

    m.params["lr"] = lr
    m.params["hidden"] = ceil(Int, hidden)
    m.params["pdrop"] = pdrop
    m.params["batchsize"] = ceil(Int, batchsize)
end

function hyperbandopt(m::Model, dtrn::Data; showloss=true, cv=false)
    best = (Inf,)
    neval = 0
    dt, dv = splitdata(dtrn)

    function getloss(config, epochs)
        neval += 1
        lr, hidden, pdrop, batchsize = config
        epochs = round(Int, epochs)

        m.params["lr"] = lr
        m.params["hidden"] = hidden
        m.params["pdrop"] = pdrop
        m.params["batchsize"] = batchsize

        if cv
            loss = crossvalidate(m, dtrn; showprogress=true)
        else
            train(m, dt; showprogress=false, epochs=epochs, savemodel=false)
            loss = m.model(dv.x, dv.y)
        end

        showloss && println("Loss: $loss, Config: $config")

        if loss < best[1]
            best = (loss, config, epochs)
        end

        return loss
    end

    function getconfig()
        lr = 0.001 ^ rand() * 10 / 3
        hidden = 50 + floor(Int, 1000 ^ rand())
        pdrop = 0.005 ^ rand() / 2
        batchsize = 2 ^ rand(0:10)

        return (lr, hidden, pdrop, batchsize)
    end

    hyperband(getconfig, getloss)
    lr, hidden, pdrop, batchsize = best[2]

    m.params["lr"] = lr
    m.params["hidden"] = hidden
    m.params["pdrop"] = pdrop
    m.params["batchsize"] = batchsize
end

function hyperoptimization(m::Model, dtrn::Data; method="goldensection",
                           showloss=true, cv=false)
    if method == "goldensection"
        goldensectionopt(m, dtrn; showloss=showloss, cv=cv)
    elseif method == "hyperband"
        hyperbandopt(m, dtrn; showloss=showloss, cv=cv)
    else
        throw("Invalid optimization method")
    end
end
