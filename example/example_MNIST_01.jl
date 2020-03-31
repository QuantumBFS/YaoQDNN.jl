using YaoQDNN
using Yao
using Flux

ql_1 = QNNL{Float64}(hardware_efficient, 8, 64, 136, "XYZ")
ql_2 = QNNL{Float64}(hardware_efficient, 6, 24, 84, "YZ")
M = [zeros(ComplexF64, 2, 2) for i = 1:2]
for i = 1:2
    M[i][i,i] = 1
end
Hami_3 = [chain(4, put(4, 1=>matblock(M[i]))) for i = 1:2]
ql_3 = QNNL{Float64}(hardware_efficient, 4, 12, 32, Hami_3)

m = Chain(ql_1, ql_2, ql_3)
loss(data_x, data_y) = sum([Flux.mse(m(data_x[i]), data_y[i])*2 for i in 1:length(data_x)]) / length(data_x)

opt = ADAM()
x = [rand(64) for i = 1:3]
y = [rand(2) for i = 1:3]

gs = gradient(()->loss(x,y), params(m))
gs.grads

using MLDatasets
using Images

# load full training set
rawdata_x, rawdata_y = MNIST.traindata();
# load full test set
rawtest_x, rawtest_y  = MNIST.testdata();

data_x = Vector{Float64}[]
data_y = Vector{Float64}[]
data_ind = Int[]
for i = 1:60000
    if rawdata_y[i] < 2
        push!(data_x, Array{Float64,1}(imresize(rawdata_x[:,:,i], 8, 8)[:]))
        y = zeros(2)
        y[rawdata_y[i]+1] = 1
        push!(data_y, y)
        push!(data_ind, i)
    end
end

test_x = Vector{Float64}[]
test_y = Vector{Float64}[]
test_ind = Int[]
for i = 1:10000
    if rawtest_y[i] < 2
        push!(test_x, Array{Float64,1}(imresize(rawtest_x[:,:,i], 8, 8)[:]))
        y = zeros(2)
        y[rawtest_y[i]+1] = 1
        push!(test_y, y)
        push!(test_ind, i)
    end
end

train_data = Flux.Data.DataLoader(data_x, data_y, batchsize=240)

opt = ADAM()

function evalcb()
    l = loss(data_x, data_y)
    push!(history_l, l)
    l_test = loss(test_x, test_y)
    push!(history_l_test, l_test)

    println("loss = $l, test loss = $l_test")
end
history_l = []
history_l_test = []

Flux.Optimise.train!(loss, params(m), train_data, opt, cb = evalcb)

using Plots
plot(1:length(history_l), [history_l history_l_test])
