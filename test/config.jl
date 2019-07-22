import AutoML
using Test

@testset "config" begin
    inputs = [("feature1", "String"),
              ("feature2", "Text"),
              ("feature3", "Int"),
              ("feature4", "Image"),
              ("feature5", "Binary Category")]
    outputs = [("label1", "Category"),
               ("label2", "Float")]
    @test typeof(AutoML.Config(inputs, outputs)) == AutoML.Config
end
