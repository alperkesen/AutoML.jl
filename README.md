# AutoML

AutoML package implemented in Julia using Knet.

## Getting Started

First, you have to install AutoML package. Run julia from the command line, set the environment for the package and install AutoML package using the repository address.

```julia
(v1.1) pkg> activate env
(env) pkg> add https://github.com/alperkesen/AutoML.jl
```

## How to use

AutoML package provides a trained model using a few lines of code. In order to initialize a model, names and types of input features and output features should be given to constructor. Then, we can train our model using path to csv file and initialized model.

### Home Rental Example

#### Initializing the model

Home rentals dataset has seven input features:
- neighborhood
- number of bathrooms
- location
- days on market
- initial price
- number of rooms
- sqft

It has one output column:
- rental price

Lists containing tuples of feature names and types should be defined for input and output columns. Using those two lists, we can initialize our model before training it.

```julia
julia> hrinputs = [("neighborhood", AutoML.CATEGORY),
                   ("number_of_bathrooms", AutoML.INT),
                   ("location", AutoML.CATEGORY),
                   ("days_on_market", AutoML.INT),
                   ("initial_price", AutoML.FLOAT),
                   ("number_of_rooms", AutoML.INT),
                   ("sqft", AutoML.FLOAT)]
julia> hroutputs = [("rental_price", AutoML.FLOAT)]
julia> model = AutoML.Model(hrinputs, hroutputs; name="homerentals")
```

#### Training the model

After initializing the model, we can use give our model and path of csv file to train it. Default number of epochs is 1, an arbitrary number can be given as a parameter to epochs argument. Train method returns trained model and batc

```julia
julia> model, dtrn = AutoML.train(model, "data/home_rentals/home_rentals.csv"; epochs=50)
```

#### Calculating accuracy/loss

We can evaluate the performance of our model using accuracy/loss.

```julia
julia> err = model.model(dtrn.x, dtrn.y)
julia> println("Train error: $err")

julia> acc = AutoML.accuracy(m.model, dtrn)
julia> println("Train accuracy: $acc")
```
