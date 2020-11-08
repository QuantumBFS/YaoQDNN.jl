using YaoQDNN, Yao
using Flux

ql_1 = QNNL{Float64}(hardware_efficient, 8, 64, 136, "XYZ")
ql_2 = QNNL{Float64}(hardware_efficient, 6, 24, 84, "YZ")

out_proj = [[1.0+0im 0; 0 0], [0 0; 0 1.0+0im]]
out_hami = [put(4, 1 => matblock(out_proj[i])) for i = 1:2]
ql_3 = QNNL{Float64}(hardware_efficient, 4, 12, 32, out_hami; no_bias = true)

m = Chain(ql_1, ql_2, ql_3)

# m_faithful = Chain(
#     QNNL_faithful(ql_1, 100000),
#     QNNL_faithful(ql_2, 100000),
#     QNNL_faithful(ql_3, 100000),
# )

using MLDatasets
using Images

# load full training set
rawdata_x, rawdata_y = MNIST.traindata();
# load full test set
# test_x,  test_y  = MNIST.testdata();
data_x = Vector{Float64}[]
data_y = Vector{Float64}[]
data_ind = Int[]
for i = 1:60000
    if rawdata_y[i] < 2
        push!(data_x, Array{Float64,1}(imresize(rawdata_x[:, :, i], 8, 8)[:]))
        y = zeros(2)
        y[rawdata_y[i]+1] = 1
        push!(data_y, y)
        push!(data_ind, i)
    end
end

using Flux.Optimise: update!, ADAM
using FileIO
# opt = ADAM(0.01)

function train(qm, data_x, data_y, iter::Integer, nbatch::Integer, opt, folder_name)
    function loss(data_x, data_y)
        n = length(data_x)
        return sum(2 * Flux.mse(qm(data_x[i]), data_y[i]) for i = 1:n) / n
    end

    ps = params(qm)
    l = loss(data_x, data_y)
    save("$(folder_name)/loss_0.jld", "l", l)
    save("$(folder_name)/qm_0.jld", "qm", qm)
    println("At first, loss = $(l)")
    for i = 1:iter
        println("Iteration $(i):")
        println("Computing gradient...")
        batch_ids = rand(1:length(data_x), nbatch)
        gs = gradient(
            () -> loss(getindex(data_x, batch_ids), getindex(data_y, batch_ids)),
            ps,
        )
        println("Updating and computing loss...")
        for p in ps
            update!(opt, p, gs[p])
        end
        l = loss(data_x, data_y)
        save("$(folder_name)/loss_$(i).jld", "l", l)
        save("$(folder_name)/qm_$(i).jld", "qm", qm)
        println("loss = $(l)")
    end
end

# train(m, data_x, data_y, 200, 240, ADAM(0.01), "data_ideal")

m0 = load("data_ideal/qm_0.jld", "qm")
ql0_1 = m0[1]
ql0_2 = m0[2]
ql0_3 = m0[3]

nshots = 1000
m_faithful = Chain(
    QNNL_faithful(ql0_1, nshots),
    QNNL_faithful(ql0_2, nshots),
    QNNL_faithful(ql0_3, nshots),
)
train(m_faithful, data_x, data_y, 200, 240, ADAM(0.01), "data_faithful_$(nshots)")
