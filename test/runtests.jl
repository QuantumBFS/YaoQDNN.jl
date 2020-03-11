using YaoQDNN
using Yao
using Test

@testset "layer.jl" begin
    ql_1 = QNNL{Float64}(hardware_efficient, 8, 8, 20, "XYZ")
    ql_2 = QNNL{Float64}(hardware_efficient, 6, 6, rand(12), "YZ", rand(12))

    x = rand(8)

    YaoQDNN.forward(ql_1, x)
    YaoQDNN.back_propagation(ql_1, x)
end
