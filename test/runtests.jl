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
    loss(m, x, y) = Flux.mse(m(x), y)
    opt = ADAM()
    x = ones(12)
    y = ones(4)

    _, back = Zygote.pullback(Flux.mse, m(x), y)
    m̄ = back(1)[1]

    gs = gradient((m)->loss(m,x,y), m)
    @test all(gs[1][1][2] .≈ YaoQDNN.forwardAD(ql_2, ql_1(x), m̄)[1])

    # Float32
    ql_1 = QNNL{Float32}(hardware_efficient, 8, 12, 20, "YZ")
    ql_2 = QNNL{Float32}(hardware_efficient, 4, 16, 10, "X")
    m = Chain(ql_1, ql_2)
    loss(m, x, y) = Flux.mse(m(x), y)
    opt = ADAM()
    x = ones(Float32, 12)
    y = ones(Float32, 4)

    _, back = Zygote.pullback(Flux.mse, m(x), y)
    m̄ = back(1)[1]

    gs = gradient((m)->loss(m,x,y), m)
    @test all(gs[1][1][2] .≈ YaoQDNN.forwardAD(ql_2, ql_1(x), m̄)[1])
end
