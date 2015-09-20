using Mocha


x_dim = 28^2
z_dim = 2
hidden_dim = 128
batch_size = 100

backend = GPUBackend()
init(backend)


train_dl = HDF5DataLayer(name="train-data",
        source=ENV["DATASETS"]"/mnist/train.txt", 
        batch_size=batch_size)
test_dl = HDF5DataLayer(name="test-data",
        source=ENV["DATASETS"]"/mnist/test.txt",
        batch_size=batch_size)

enc_layers = [
    InnerProductLayer(name="enc1",
        output_dim=hidden_dim,
        neuron=Neurons.ReLU(),
        bottoms=[:data], tops=[:enc1]),
    SplitLayer(name="enc1_split",
        bottoms=[:enc1], tops=[:enc1a, :enc1b]),
    InnerProductLayer(name="z_mean",
        output_dim=z_dim,
        neuron=Neurons.Identity(),
        bottoms=[:enc1a], tops=[:z_mean]),
    InnerProductLayer(name="z_sd",
        output_dim=z_dim, 
        bias_init=ConstantInitializer(-1.),
        weight_init=ConstantInitializer(0.),
        neuron=Neurons.Exponential(),
        bottoms=[:enc1b], tops=[:z_sd])]

sampling_layers = [
    SplitLayer(name="z_mean_split",
        bottoms=[:z_mean], tops=[:z_mean1, :z_mean2]),
    SplitLayer(name="z_sd_split",
        bottoms=[:z_sd], tops=[:z_sd1, :z_sd2]),

    RandomNormalLayer(output_dims=[z_dim],
        batch_sizes=[batch_size],
        tops=[:noise01]),
    ElementWiseLayer(operation=ElementWiseFunctors.Multiply(),
        bottoms=[:noise01, :z_sd1],
        tops=[:noise0sd]),
    ElementWiseLayer(name="z",
        operation=ElementWiseFunctors.Add(),
        bottoms=[:z_mean1, :noise0sd],
        tops=[:z]),

    GaussianKLLossLayer(name="encoder-loss",
        bottoms=[:z_mean2, :z_sd2])]

dec_layers = [
    InnerProductLayer(name="dec1", output_dim=hidden_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:z], tops=[:dec1]),
    SplitLayer(name="dec1_split",
        bottoms=[:dec1], tops=[:dec1a, :dec1b]),
    InnerProductLayer(name="x_mean", output_dim=x_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:dec1a], tops=[:x_mean]),
    InnerProductLayer(name="x_sd_0", output_dim=x_dim,
        neuron=Neurons.Exponential(),
        bottoms=[:dec1b], tops=[:x_sd_0]),
    PowerLayer(name="x_sd",
        shift=1e-3,
        bottoms=[:x_sd_0],
        tops=[:x_sd])]

expected_recon_loss = GaussianReconLossLayer(bottoms=[:x_mean, :x_sd, :data])


non_data_layers = [
    enc_layers...,
    sampling_layers...,
    dec_layers...,
    expected_recon_loss]

train_net = Net("train-net", backend, [train_dl, non_data_layers...])
test_net = Net("test-net", backend, [test_dl, non_data_layers...])

adam_instance = Adam()
params = make_solver_parameters(adam_instance;
    max_iter=1_200_000,
    regu_coef=1e-4,
    lr_policy=LRPolicy.Fixed(1e-3))
solver = Solver(adam_instance, params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=10000)
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=10000)

base_dir = "snapshots_sat"
setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=30000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=30000)

solve(solver, train_net)

destroy(train_net)
destroy(test_net)


shutdown(backend)

1

#=
function vblob(blob)
    ret = Array(Float32, blob.shape)
    copy!(ret, blob)
    ret
end
=#

#=
using JLD
jdl_file = jldopen(joinpath(base_dir, "snapshot-075000.jld"))
load_network(jdl_file, train_net)
close(jdl_file)

forward(train_net)
backward(train_net)
=#

