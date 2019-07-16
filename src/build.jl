using Knet: relu


function buildlinearestimator(inputsize, outputsize; hiddensize=20,
                              f=relu, pdrop=0, scale=0.01)
    l1 = Layer(inputsize, hiddensize, scale, f; pdrop=pdrop)
    l2 = Layer(hiddensize, outputsize, scale, identity; pdrop=pdrop)
    Chain(l1, l2)
end

function buildclassificationmodel(inputsize, outputsize; hiddensize=50,
                                  f=relu, pdrop=0, scale=0.01)
    l1 = Layer(inputsize, hiddensize, 0.01, relu; pdrop=pdrop)
    l2 = Layer(hiddensize, outputsize, 0.01, identity; pdrop=pdrop)
    Chain2(l1, l2)
end

function buildimageclassification(inputsize, outputsize; cx=3, cy=20, wx=5, wy=5,
                                  f=relu, pdrop=0, scale=0.01)
    conv1 = Conv(wx, wy, cx, cy)
    l1 = Layer2(3920, outputsize, scale, identity; pdrop=pdrop)
    Chain2(conv1, l1)
end

function buildsentimentanalysis(outputsize; voclen=30000, embedsize=100,
                                hiddensize=20, pdrop=0, scale=0.01)
    OneLayerBiRNN(voclen, embedsize, hiddensize, outputsize;
                  pdrop=pdrop, scale=scale)
end

function buildquestionmatching(outputsize; voclen=30000, embedsize=100,
                               hiddensize=20, pdrop=0, scale=0.01)
    TwoTextsClassifier(voclen, embedsize, hiddensize, outputsize;
                       pdrop=pdrop, scale=scale, rnnType=:lstm)
end

function buildgenesequence(outputsize; voclen=10, embedsize=4,
                           hiddensize1=60, hiddensize2=30, pdrop=0,
                           scale=0.01)
    TwoLayerBiRNN(voclen, embedsize, hiddensize1, hiddensize2, outputsize;
                  pdrop=pdrop, scale=scale, rnnType=:lstm, bidirectional=true)
end
