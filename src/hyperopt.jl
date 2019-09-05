function hyperoptimization(m::Model, dtrn::Data; showloss=true, cv=false)
    neval = 0

    function f(x)
        neval += 1
        lr, hidden, pdrop, batchsize = xform(x)

        if hidden < 10000 && 0 <= pdrop <= 1 && 0 < batchsize < 512
            m.params["lr"] = lr
            m.params["hidden"] = hidden
            m.params["pdrop"] = pdrop
            m.params["batchsize"] = batchsize

            if cv
                loss = crossvalidate(m, dtrn; showprogress=true)
            else
                dt, dv = splitdata(dtrn)
                train(m, dt; showprogress=false, epochs=1, savemodel=false)
                loss = sum([m.model(x,y) for (x,y) in dv])
            end
        else
            loss = NaN
        end

        showloss && println("Loss: $loss")

        return loss
    end

    function xform(x)
        lr, hidden, pdrop, batchsize = exp.(x) .* [0.01, 100.0, 0.1, 32]
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
