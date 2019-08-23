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

function train_cifar_100(; smallset=true, epochs=1)
    trn, tst = AutoML.cifar100()
    
    if smallset
        selected = Random.shuffle(Random.seed!(0), Vector(1:10000))[1024:1600]
        trn = trn[selected, :]
        tst = tst[selected, :]
    end

    cifartrn = AutoML.csv2data(trn)
    cifartst = AutoML.csv2data(tst)

    cifarinputs = [("image_path", "Image")]
    cifaroutputs = [("class", "Category")]

    model = AutoML.Model(cifarinputs, cifaroutputs; name="cifar100")
    model, dtrn = AutoML.train(model, cifartrn; epochs=epochs)
    dtst = AutoML.getbatches(model, cifartst)
  
    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_imdb(; epochs=1)
    trn, tst = AutoML.imdb_movie_review()

    imdbtrn = AutoML.csv2data(trn)
    imdbtst = AutoML.csv2data(tst)

    imdbinputs = [("review", "Text")]
    imdboutputs = [("sentiment", "Category")]

    model = AutoML.Model(imdbinputs, imdboutputs; name="imdbreviews")
    model, dtrn = AutoML.train(model, imdbtrn; epochs=epochs)
    dtst = AutoML.getbatches(model, imdbtst)

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
