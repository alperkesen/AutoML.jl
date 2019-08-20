import AutoML
import Random
using Knet: gpu, KnetArray

function train_house_rentals(; epochs=1)
    house_rentals = AutoML.house_rentals()
    trn, tst = splitdata(house_rentals; trainprop=0.8)

    house_rentals_trn = AutoML.csv2data(trn)
    house_rentals_tst = AutoML.csv2data(tst)

    house_rentals_inputs = [("neighborhood", "String"),
                            ("number_of_bathrooms", "Int"),
                            ("location", "String"),
                            ("days_on_market", "Int"),
                            ("initial_price", "Float"),
                            ("number_of_rooms", "Int"),
                            ("sqft", "Float")]
    house_rentals_outputs = [("rental_price", "Float")]

    model = AutoML.Model(house_rentals_inputs, house_rentals_outputs;
                         name="houserentals")
    model, dtrn = AutoML.train(model, house_rentals_trn; epochs=epochs)
    dtst = AutoML.getbatches(model, house_rentals_tst)

    println("Train error:")
    println(model.model(dtrn.x, dtrn.y))

    println("Test error:")
    println(model.model(dtst.x, dtst.y))

    model, dtrn, dtst
end

function train_gene_sequences(; epochs=1)
    gene_sequences = AutoML.splice_junction()
    trn, tst = splitdata(gene_sequences; trainprop=0.8)

    gen_trn = AutoML.csv2data(trn)
    gen_tst = AutoML.csv2data(tst)

    gene_sequences_inputs = [("attribute_$i", "Category") for i in 1:60]
    gene_sequences_outputs = [("Class", "Category")]

    model = AutoML.Model(gene_sequences_inputs, gene_sequences_outputs;
                         name="genesequences")
    model, dtrn = AutoML.train(model, gen_trn; epochs=epochs, cv=false)
    dtst = AutoML.getbatches(model, gen_tst)

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

    cifar_100_trn = AutoML.csv2data(trn)
    cifar_100_tst = AutoML.csv2data(tst)

    cifar_100_inputs = [("image_path", "Image")]
    cifar_100_outputs = [("class", "Category")]

    model = AutoML.Model(cifar_100_inputs, cifar_100_outputs; name="cifar100")
    model, dtrn = AutoML.train(model, cifar_100_trn; epochs=epochs)
    dtst = AutoML.getbatches(model, cifar_100_tst)
  
    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_imdb(; epochs=1)
    imdb_train, imdb_test = AutoML.imdb_movie_review()

    imdb_trn = AutoML.csv2data(imdb_train)
    imdb_tst = AutoML.csv2data(imdb_test)

    imdb_inputs = [("review", "Text")]
    imdb_outputs = [("sentiment", "Category")]

    model = AutoML.Model(imdb_inputs, imdb_outputs; name="imdbreviews")
    model, dtrn = AutoML.train(model, imdb_trn; epochs=epochs)
    dtst = AutoML.getbatches(model, imdb_tst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_quora(; epochs=1)
    quora_train, quora_test = AutoML.quora_questions()

    quora_trn = AutoML.csv2data(quora_train)
    quora_tst = AutoML.csv2data(quora_test[2:end, :])

    quora_inputs = [("question1", "Text"), ("question2", "Text")]
    quora_outputs = [("is_duplicate", "Binary Category")]

    model = AutoML.Model(quora_inputs, quora_outputs; name="quora")
    model, dtrn = AutoML.train(model, quora_trn; epochs=epochs)
    dtst = AutoML.getbatches(model, quora_tst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end

function train_default_of_credit(; epochs=1)
    credit_train, credit_test = AutoML.default_of_credit()

    credit_trn = AutoML.csv2data(credit_train)
    credit_tst = AutoML.csv2data(credit_test)
    
    credit_inputs = [(x[1], "Int") for x in credit_trn
                     if x != "default.payment.next.month"]
    credit_outputs = [("default.payment.next.month", "Binary Category")]

    model = AutoML.Model(credit_inputs, credit_outputs; name="defaultofcredit")
    model, dtrn = AutoML.train(model, credit_trn; epochs=epochs)
    dtst = AutoML.getbatches(model, credit_tst)

    println("Train accuracy:")
    println(accuracy(model.model, dtrn))

    println("Test accuracy:")
    println(accuracy(model.model, dtst))

    model, dtrn, dtst
end
