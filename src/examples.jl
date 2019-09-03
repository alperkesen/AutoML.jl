import AutoML
import Random
using Knet: gpu, KnetArray

function train_house_rentals(; epochs=1)
    hr = AutoML.house_rentals()
    trn, tst = splitdata(hr; trainprop=0.8)

    hrtrn = AutoML.csv2data(trn)
    hrtst = AutoML.csv2data(tst)

    hrinputs = [("neighborhood", "String"),
                 ("number_of_bathrooms", "Int"),
                 ("location", "String"),
                 ("days_on_market", "Int"),
                 ("initial_price", "Float"),
                 ("number_of_rooms", "Int"),
                 ("sqft", "Float")]
    hroutputs = [("rental_price", "Float")]

    model = AutoML.Model(hrinputs, hroutputs; name="houserentals")
    model, dtrn = AutoML.train(model, hrtrn; epochs=epochs)
    dtst = AutoML.getbatches(model, hrtst)

    println("Train error:")
    println(model.model(dtrn.x, dtrn.y))

    println("Test error:")
    println(model.model(dtst.x, dtst.y))

    model, dtrn, dtst
end

function train_gene_sequences(; epochs=1)
    gs = AutoML.splice_junction()
    trn, tst = splitdata(gs, gs.Class; trainprop=0.8)

    gstrn = AutoML.csv2data(trn)
    gstst = AutoML.csv2data(tst)

    gsinputs = [("attribute_$i", "Category") for i in 1:60]
    gsoutputs = [("Class", "Category")]

    model = AutoML.Model(gsinputs, gsoutputs; name="genesequences")
    model, dtrn = AutoML.train(model, gstrn; epochs=epochs)
    dtst = AutoML.getbatches(model, gstst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    return model, dtrn, dtst
end

function train_cifar_100(; epochs=1, datapath=nothing)
    cifarinputs = [("image_path", "Image")]
    cifaroutputs = [("class", "Category")]

    model = AutoML.Model(cifarinputs, cifaroutputs; name="cifar100")

    if datapath == nothing
        println("Preprocessing from scratch...")
        model, dtrn = AutoML.train(model, AutoML.CIFAR100TRAIN; epochs=epochs)
        dtst = AutoML.getbatches(model, AutoML.CIFAR100TEST)
    else
        datapath = joinpath(AutoML.SAVEDIR, datapath)
        !isfile(datapath) ? throw("Datapath is invalid") : println("Loading data...")
        dtrn, dtst = Knet.load(datapath, "dtrn", "dtst")
        model, dtrn = AutoML.train(model, dtrn; epochs=epochs)
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

    model = AutoML.Model(imdbinputs, imdboutputs; name="imdbreviews")

    if datapath == nothing
        println("Preprocessing from scratch...")
        model, dtrn = AutoML.train(model, AutoML.IMDBTRAIN; epochs=epochs)
        dtst = AutoML.getbatches(model, AutoML.IMDBTEST)
    else
        datapath = joinpath(AutoML.SAVEDIR, datapath)
        !isfile(datapath) ? throw("Datapath is invalid") : println("Loading data...")
        dtrn, dtst = Knet.load(datapath, "dtrn", "dtst")
        model, dtrn = AutoML.train(model, dtrn; epochs=epochs)
    end

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_quora(; epochs=1)
    trn, tst = AutoML.quora_questions()

    quoratrn = AutoML.csv2data(trn)
    quoratst = AutoML.csv2data(tst[2:end, :])

    quorainputs = [("question1", "Text"), ("question2", "Text")]
    quoraoutputs = [("is_duplicate", "Binary Category")]

    model = AutoML.Model(quorainputs, quoraoutputs; name="quora")
    model, dtrn = AutoML.train(model, quoratrn; epochs=epochs)
    dtst = AutoML.getbatches(model, quoratst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_default_of_credit(; epochs=1)
    trn, tst = AutoML.default_of_credit()

    credittrn = AutoML.csv2data(trn)
    credittst = AutoML.csv2data(tst)
    
    creditinputs = [(x[1], "Int") for x in credittrn
                     if x != "default.payment.next.month"]
    creditoutputs = [("default.payment.next.month", "Binary Category")]

    model = AutoML.Model(creditinputs, creditoutputs; name="defaultofcredit")
    model, dtrn = AutoML.train(model, credittrn; epochs=epochs)
    dtst = AutoML.getbatches(model, credittst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_prediction_of_return(; epochs=1)
    portrn, portst = AutoML.prediction_of_return()

    porinputs = [("customerID", AutoML.INT),
                 ("creationDate", AutoML.DATE),
                 ("manufacturerID", AutoML.INT),
                 ("price", AutoML.FLOAT),
                 ("deliveryDate", AutoML.DATE),
                 ("salutation", AutoML.CATEGORY),
                 ("dateOfBirth", AutoML.DATE),
                 ("state", AutoML.CATEGORY),
                 ("itemID", AutoML.INT),
                 ("orderDate", AutoML.DATE),
                 ("size", AutoML.CATEGORY),
                 ("color", AutoML.CATEGORY),
                 ("orderItemID", AutoML.INT)]
    poroutputs = [("returnShipment", AutoML.BINARYCATEGORY)]

    model = AutoML.Model(porinputs, poroutputs; name="predictofreturn")
    model, dtrn = AutoML.train(model, portrn; epochs=epochs)
    dtst = AutoML.getbatches(model, portst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_sensor(; epochs=1)
    sd = AutoML.sensor()
    trn, tst = splitdata(sd; trainprop=0.8)

    sdtrn = AutoML.csv2data(trn)
    sdtst = AutoML.csv2data(tst)

    sdinputs = [("sensor 1", AutoML.INT),
                ("sensor 2", AutoML.INT),
                ("sensor 3", AutoML.INT),
                ("sensor4", AutoML.INT)]
    sdoutputs = [("output", AutoML.CATEGORY)]

    model = AutoML.Model(sdinputs, sdoutputs; name="sensordata")
    model, dtrn = AutoML.train(model, sdtrn; epochs=epochs)
    dtst = AutoML.getbatches(model, sdtst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_spam(; epochs=1)
    spam = AutoML.spam()
    trn, tst = splitdata(spam; trainprop=0.8)

    spamtrn = AutoML.csv2data(trn)
    spamtst = AutoML.csv2data(tst)

    spaminputs = [("v2", AutoML.TEXT)]
    spamoutputs = [("v1", AutoML.BINARYCATEGORY)]

    model = AutoML.Model(spaminputs, spamoutputs; name="spam")
    model, dtrn = AutoML.train(model, spamtrn; epochs=epochs)
    dtst = AutoML.getbatches(model, spamtst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end
