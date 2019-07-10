import AutoML
import Random

function train_house_rentals()
    house_rentals = AutoML.house_rentals()
    house_rentals_data = AutoML.csv2data(house_rentals)
    house_rentals_inputs = [("neighborhood", "String"),
                            ("number_of_bathrooms", "Int"),
                            ("location", "String"),
                            ("days_on_market", "Int"),
                            ("initial_price", "Float"),
                            ("number_of_rooms", "Int"),
                            ("sqft", "Float")]
    house_rentals_outputs = [("rental_price", "Float")]
    model = AutoML.Model(house_rentals_inputs, house_rentals_outputs)
    result = AutoML.train(model, house_rentals_data; epochs=100)
end

function train_gene_sequences()
    gene_sequences = AutoML.splice_junction()
    gene_sequences_data = AutoML.csv2data(gene_sequences)
    gene_sequences_inputs = [("attribute_$i", "Category") for i in 1:60]
    gene_sequences_outputs = [("Class", "Category")]
    model = AutoML.Model(gene_sequences_inputs, gene_sequences_outputs)
    result = AutoML.train(model, gene_sequences_data; epochs=100)
end

function train_cifar_100(; smallset=true)
    cifar_100 = AutoML.cifar_100()
    
    if smallset
        selected = Random.shuffle(Random.seed!(0), Vector(1:50000))[1024:1600]
        cifar_100 = cifar_100[selected, :]
    end

    cifar_100_data = AutoML.csv2data(cifar_100)
    cifar_100_inputs = [("image_path", "Image")]
    cifar_100_outputs = [("class", "Category")]
    model = AutoML.Model(cifar_100_inputs, cifar_100_outputs)
    result = AutoML.train(model, cifar_100_data; epochs=100)
end

function train_imdb()
    imdb = AutoML.imdb_movie_review()
    imdb_data = AutoML.csv2data(imdb)
    imdb_inputs = [("review", "Text")]
    imdb_outputs = [("sentiment", "Category")]
    model = AutoML.Model(imdb_inputs, imdb_outputs)
    result = AutoML.train(model, imdb_data; epochs=1)
end

function train_quora()
    quora = AutoML.quora_questions()
    quora_data = AutoML.csv2data(quora)
    quora_inputs = [("question1", "Text"), ("question2", "Text")]
    quora_outputs = [("is_duplicate", "Binary Category")]
    model = AutoML.Model(quora_inputs, quora_outputs)
    result = AutoML.train(model, quora_data; epochs=100)
end

function train_default_of_credit()
    credit = AutoML.default_of_credit()
    credit_data = AutoML.csv2data(credit)
    credit_inputs = [(x[1], "Int") for x in credit_data
                     if x != "default.payment.next.month"]
    credit_outputs = [("default.payment.next.month", "Binary Category")]
    model = AutoML.Model(credit_inputs, credit_outputs)
    result = AutoML.train(model, credit_data; epochs=100)
end

