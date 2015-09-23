using Mocha

backend = CPUBackend()
init(backend)

include("mnist-layers.jl")

train_net = Net("train-net", backend, [train_dl, non_data_layers...])
test_net = Net("test-net", backend, [test_dl, non_data_layers...])

using JLD
base_dir = "snapshots_sun"
jdl_file = jldopen(joinpath(base_dir, "snapshot-150000.jld"))
load_network(jdl_file, train_net)
close(jdl_file)

forward(train_net)
backward(train_net)

forward(test_net)
backward(test_net)

data = Array(Float32, 28, 28, 100)
copy!(data, get_layer_state(train_net, "train-data").blobs[1])
recon = Array(Float32, 28, 28, 100)
copy!(recon, get_layer_state(train_net, "x_mean").blobs[1])
open("train_datarecon", "w") do f
    serialize(f, (data, recon))
end

data2 = Array(Float32, 28, 28, 100)
copy!(data2, get_layer_state(test_net, "test-data").blobs[1])
recon2 = Array(Float32, 28, 28, 100)
copy!(recon2, get_layer_state(test_net, "x_mean").blobs[1])
open("test_datarecon", "w") do f
    serialize(f, (data2, recon2))
end

