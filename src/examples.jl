import AutoML
import Random

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

    model = AutoML.Model(house_rentals_inputs, house_rentals_outputs)
    result = AutoML.train(model, house_rentals_trn; epochs=epochs)

    xtrn, ytrn = preparedata(model, house_rentals_trn)
    xtst, ytst = preparedata(model, house_rentals_tst)

    println("Train error:")
    println(model.model(xtrn, ytrn))

    println("Test error:")
    println(model.model(xtst, ytst))

    model, xtrn, ytrn, xtst, ytst
end

function train_gene_sequences(; epochs=1)
    gene_sequences = AutoML.splice_junction()
    trn, tst = splitdata(gene_sequences; trainprop=0.8)

    gen_trn = AutoML.csv2data(trn)
    gen_tst = AutoML.csv2data(tst)

    gene_sequences_inputs = [("attribute_$i", "Category") for i in 1:60]
    gene_sequences_outputs = [("Class", "Category")]

    model = AutoML.Model(gene_sequences_inputs, gene_sequences_outputs)
    result = AutoML.train(model, gen_trn; epochs=epochs)

    xtrn, ytrn = preparedata(model, gen_trn)
    dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)

    xtst, ytst = preparedata(model, gen_tst)
    dtst = minibatch(xtst, ytst, 32; shuffle=true)

    model, dtrn, dtst
end

function train_cifar_100(; smallset=true, epochs=1)
    cifar_100 = AutoML.cifar_100()
    
    if smallset
        selected = Random.shuffle(Random.seed!(0), Vector(1:50000))[1024:1600]
        cifar_100 = cifar_100[selected, :]
    end

    cifar_100_data = AutoML.csv2data(cifar_100)
    cifar_100_inputs = [("image_path", "Image")]
    cifar_100_outputs = [("class", "Category")]
    model = AutoML.Model(cifar_100_inputs, cifar_100_outputs)
    result = AutoML.train(model, cifar_100_data; epochs=epochs)
end

function train_imdb(; epochs=1)
    imdb_train, imdb_test = AutoML.imdb_movie_review()

    imdb_trn = AutoML.csv2data(imdb_train)
    imdb_tst = AutoML.csv2data(imdb_test)

    imdb_inputs = [("review", "Text")]
    imdb_outputs = [("sentiment", "Category")]

    model = AutoML.Model(imdb_inputs, imdb_outputs)
    result = AutoML.train(model, imdb_trn; epochs=epochs)

    xtrn, ytrn = preparedata(model, imdb_trn)
    dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)

    xtst, ytst = preparedata(model, imdb_tst)
    dtst = minibatch(xtst, ytst, 32; shuffle=true)

    model, dtrn, dtst
end

function train_quora(; epochs=1)
    quora_train, quora_test = AutoML.quora_questions()
    quora_trn = AutoML.csv2data(quora_train)
    quora_tst = AutoML.csv2data(quora_test[2:end, :])

    quora_inputs = [("question1", "Text"), ("question2", "Text")]
    quora_outputs = [("is_duplicate", "Binary Category")]

    model = AutoML.Model(quora_inputs, quora_outputs)
    result = AutoML.train(model, quora_trn; epochs=epochs)

    xtrn, ytrn = preparedata(model, quora_trn)
    dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)

    xtst, ytst = preparedata(model, quora_tst)
    dtst = minibatch(xtst, ytst, 32; shuffle=true)

    model, dtrn, dtst
end

function train_default_of_credit(; epochs=1)
    credit_train, credit_test = AutoML.default_of_credit()
    credit_trn = AutoML.csv2data(credit_train)
    credit_tst = AutoML.csv2data(credit_test)
    
    credit_inputs = [(x[1], "Int") for x in credit_trn
                     if x != "default.payment.next.month"]
    credit_outputs = [("default.payment.next.month", "Binary Category")]
    model = AutoML.Model(credit_inputs, credit_outputs)
    result = AutoML.train(model, credit_trn; epochs=epochs)

    xtrn, ytrn = preparedata(model, credit_trn)
    dtrn = minibatch(xtrn, ytrn, 32; shuffle=true)

    xtst, ytst = preparedata(model, credit_tst)
    dtst = minibatch(xtst, ytst, 32; shuffle=true)

    model, dtrn, dtst
end
