import Random
using Knet: gpu, KnetArray

function train_house_rentals(; epochs=1)
    hr = house_rentals()
    trn, tst = splitdata(hr; trainprop=0.8)

    hrtrn = csv2data(trn)
    hrtst = csv2data(tst)

    hrinputs = [("neighborhood", "String"),
                 ("number_of_bathrooms", "Int"),
                 ("location", "String"),
                 ("days_on_market", "Int"),
                 ("initial_price", "Float"),
                 ("number_of_rooms", "Int"),
                 ("sqft", "Float")]
    hroutputs = [("rental_price", "Float")]

    model = Model(hrinputs, hroutputs; name="houserentals")
    model, dtrn = train(model, hrtrn; epochs=epochs)
    dtst = getbatches(model, hrtst)

    println("Train error:")
    println(model.model(dtrn.x, dtrn.y))

    println("Test error:")
    println(model.model(dtst.x, dtst.y))

    model, dtrn, dtst
end

function train_gene_sequences(; epochs=1)
    gs = splice_junction()
    trn, tst = splitdata(gs, gs.Class; trainprop=0.8)

    gstrn = csv2data(trn)
    gstst = csv2data(tst)

    gsinputs = [("attribute_$i", "Category") for i in 1:60]
    gsoutputs = [("Class", "Category")]

    model = Model(gsinputs, gsoutputs; name="genesequences")
    model, dtrn = train(model, gstrn; epochs=epochs)
    dtst = getbatches(model, gstst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    return model, dtrn, dtst
end

function train_cifar_100(; epochs=1, datapath=nothing)
    cifarinputs = [("image_path", "Image")]
    cifaroutputs = [("class", "Category")]

    model = Model(cifarinputs, cifaroutputs; name="cifar100")

    if datapath == nothing
        println("Preprocessing from scratch...")
        model, dtrn = train(model, CIFAR100TRAIN; epochs=epochs)
        dtst = getbatches(model, CIFAR100TEST)
    else
        datapath = joinpath(SAVEDIR, datapath)
        !isfile(datapath) ? throw("Datapath is invalid") : println("Loading data...")
        dtrn, dtst = Knet.load(datapath, "dtrn", "dtst")
        model, dtrn = train(model, dtrn; epochs=epochs)
    end
  
    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_imdb(; epochs=1, datapath=nothing)
    imdbinputs = [("review", "Text")]
    imdboutputs = [("sentiment", "Category")]

    model = Model(imdbinputs, imdboutputs; name="imdbreviews")

    if datapath == nothing
        println("Preprocessing from scratch...")
        model, dtrn = train(model, IMDBTRAIN; epochs=epochs)
        dtst = getbatches(model, IMDBTEST)
    else
        datapath = joinpath(SAVEDIR, datapath)
        !isfile(datapath) ? throw("Datapath is invalid") : println("Loading data...")
        dtrn, dtst = Knet.load(datapath, "dtrn", "dtst")
        model, dtrn = train(model, dtrn; epochs=epochs)
    end

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_quora(; epochs=1)
    trn, tst = quora_questions()

    quoratrn = csv2data(trn)
    quoratst = csv2data(tst[2:end, :])

    quorainputs = [("question1", "Text"), ("question2", "Text")]
    quoraoutputs = [("is_duplicate", "Binary Category")]

    model = Model(quorainputs, quoraoutputs; name="quora")
    model, dtrn = train(model, quoratrn; epochs=epochs)
    dtst = getbatches(model, quoratst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_default_of_credit(; epochs=1)
    trn, tst = default_of_credit()

    credittrn = csv2data(trn)
    credittst = csv2data(tst)
    
    creditinputs = [(x[1], "Int") for x in credittrn
                     if x != "default.payment.next.month"]
    creditoutputs = [("default.payment.next.month", "Binary Category")]

    model = Model(creditinputs, creditoutputs; name="defaultofcredit")
    model, dtrn = train(model, credittrn; epochs=epochs)
    dtst = getbatches(model, credittst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_prediction_of_return(; epochs=1)
    portrn, portst = prediction_of_return()

    porinputs = [("customerID", INT),
                 ("creationDate", DATE),
                 ("manufacturerID", INT),
                 ("price", FLOAT),
                 ("deliveryDate", DATE),
                 ("salutation", CATEGORY),
                 ("dateOfBirth", DATE),
                 ("state", CATEGORY),
                 ("itemID", INT),
                 ("orderDate", DATE),
                 ("size", CATEGORY),
                 ("color", CATEGORY),
                 ("orderItemID", INT)]
    poroutputs = [("returnShipment", BINARYCATEGORY)]

    model = Model(porinputs, poroutputs; name="predictofreturn")
    model, dtrn = train(model, portrn; epochs=epochs)
    dtst = getbatches(model, portst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_sensor(; epochs=1)
    sd = sensor()
    trn, tst = splitdata(sd; trainprop=0.8)

    sdtrn = csv2data(trn)
    sdtst = csv2data(tst)

    sdinputs = [("sensor 1", INT),
                ("sensor 2", INT),
                ("sensor 3", INT),
                ("sensor4", INT)]
    sdoutputs = [("output", CATEGORY)]

    model = Model(sdinputs, sdoutputs; name="sensordata")
    model, dtrn = train(model, sdtrn; epochs=epochs)
    dtst = getbatches(model, sdtst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_spam(; epochs=1)
    spam = spam()
    trn, tst = splitdata(spam; trainprop=0.8)

    spamtrn = csv2data(trn)
    spamtst = csv2data(tst)

    spaminputs = [("v2", TEXT)]
    spamoutputs = [("v1", BINARYCATEGORY)]

    model = Model(spaminputs, spamoutputs; name="spam")
    model, dtrn = train(model, spamtrn; epochs=epochs)
    dtst = getbatches(model, spamtst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end
