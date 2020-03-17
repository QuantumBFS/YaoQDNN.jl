using YaoQDNN
using Yao
using YaoAD
using Flux
using Test

using Zygote

@testset "layer.jl" begin
    ql_1 = QNNL{Float64}(hardware_efficient, 8, 12, 20, "YZ")
    ql_2 = QNNL{Float64}(hardware_efficient, 4, 16, 10, "X")
    m = Chain(ql_1, ql_2)
    loss(x, y) = Flux.mse(m(x), y)
    x = ones(12)
    y = ones(4)

    gs = gradient(()->loss(x,y), params(ql_2))

    _, back = Zygote.pullback(Flux.mse, m(x), y)
    m̄ = back(1)[1]

    @test all(gs[ql_2.params] .≈ YaoQDNN.forwardAD(ql_2, ql_1(x), m̄)[1][1])
    @test all(gs[ql_2.bias] .≈ YaoQDNN.forwardAD(ql_2, ql_1(x), m̄)[1][2])

    # Float32
    ql_1 = QNNL{Float32}(hardware_efficient, 8, 12, 20, "YZ")
    ql_2 = QNNL{Float32}(hardware_efficient, 4, 16, 10, "X")
    m = Chain(ql_1, ql_2)
    loss(x, y) = Flux.mse(m(x), y)
    x = ones(Float32, 12)
    y = ones(Float32, 4)

    _, back = Zygote.pullback(Flux.mse, m(x), y)
    m̄ = back(1)[1]

    gs = gradient(()->loss(x,y), params(ql_2))
    @test all(gs[ql_2.params] .≈ YaoQDNN.forwardAD(ql_2, ql_1(x), m̄)[1][1])
    @test all(gs[ql_2.bias] .≈ YaoQDNN.forwardAD(ql_2, ql_1(x), m̄)[1][2])
end
