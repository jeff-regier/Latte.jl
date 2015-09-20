using Mocha

backend = CPUBackend()
init(backend)

include("mnist-layers.jl")

train_net = Net("train-net", backend, [train_dl, non_data_layers...])
test_net = Net("test-net", backend, [test_dl, non_data_layers...])

using JLD
base_dir = "."
jdl_file = jldopen(joinpath(base_dir, "snapshot-140000.jld"))
load_network(jdl_file, train_net)
close(jdl_file)

forward(train_net)
backward(train_net)
