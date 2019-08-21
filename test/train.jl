using Knet: relu, gpu

@testset "houserentals" begin
    hr = AutoML.house_rentals()
    trn, tst = AutoML.splitdata(hr; trainprop=0.8)
    epochs = 100

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

    trnerr = model.model(dtrn.x, dtrn.y)
    tsterr = model.model(dtst.x, dtst.y)
    threshold = 1000

    @test abs(trnerr - tsterr) < threshold
end
