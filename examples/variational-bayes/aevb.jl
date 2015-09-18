using Mocha


x_dim = 28^2
z_dim = 2
wide_z_dim = 10
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
        neuron=Neurons.Tanh(),
        bottoms=[:enc1a], tops=[:z_mean]),
    InnerProductLayer(name="z_sd",
        output_dim=z_dim, 
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
    ElementWiseLayer(operation=ElementWiseFunctors.Add(),
        bottoms=[:z_mean1, :noise0sd],
        tops=[:z]),

    GaussianKLLossLayer(name="encoder-loss",
        weight=1.,
        bottoms=[:z_mean2, :z_sd2])]

dec_layers = [
    InnerProductLayer(name="dec",
        output_dim=x_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:z], tops=[:dec]),
    InnerProductLayer(name="x_mean",
        output_dim=x_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:dec], tops=[:x_mean]),
    SquareLossLayer(weight=1e-0, bottoms=[:x_mean, :data])]

wide_layers = [
    InnerProductLayer(name="z_mean_wide",
        output_dim=wide_z_dim,
        neuron=Neurons.Tanh(),
        bottoms=[:enc1], tops=[:z_mean_wide]),
    IdentityLayer(bottoms=[:z_mean_wide], tops=[:z_wide]),
    InnerProductLayer(name="dec_wide",
        output_dim=x_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:z_wide], tops=[:dec])]

function solve_net(non_data_layers, max_iter)
    train_net = Net("train-net", backend, [train_dl, non_data_layers...])
    test_net = Net("test-net", backend, [test_dl, non_data_layers...])

    adam_instance = Adam()
    params = make_solver_parameters(adam_instance;
        max_iter=max_iter,
        regu_coef=1e-4,
        lr_policy=LRPolicy.Fixed(1e-3))
    solver = Solver(adam_instance, params)
    add_coffee_break(solver, TrainingSummary(), every_n_iter=batch_size)
    add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=500)
    solve(solver, train_net)
end

pretrain1_layers = [
    enc_layers[1],
    wide_layers...,
    dec_layers[2:3]...]
solve_net(pretrain1_layers, 10_000)

pretrain2_layers = [
    enc_layers[[1,3]]...,
    IdentityLayer(bottoms=[:enc1], tops=[:enc1a]),
    IdentityLayer(bottoms=[:z_mean], tops=[:z]),
    dec_layers...]
solve_net(pretrain2_layers, 20_000)

pretrain3_layers = [
    enc_layers...,
    IdentityLayer(bottoms=[:z_mean], tops=[:z_mean1]),
    IdentityLayer(bottoms=[:z_sd], tops=[:z_sd1]),
    sampling_layers[3:5],
    dec_layers...]
solve_net(pretrain3_layers, 20_000)

pretrain4_layers = [
    enc_layers...,
    sampling_layers...,
    dec_layers...]
solve_net(pretrain4_layers, 20_000)

#=
recon_layers = [
    TiedInnerProductLayer(name="recon_enc_hidden",
               tied_param_key="enc_hidden",
               neuron=Neurons.ReLU(),
               bottoms=[:l1_out], tops=[:recon]),
    TiedInnerProductLayer(name="recon_z_mean",
               tied_param_key="z_mean",
               neuron=Neurons.Tanh(),
               bottoms=[:l2_out], tops=[:recon]),
    TiedInnerProductLayer(name="recon_dec_hidden",
               tied_param_key="dec_hidden",
               neuron=Neurons.ReLU(),
               bottoms=[:l3_out], tops=[:recon]),
    TiedInnerProductLayer(name="recon_x_mean",
               tied_param_key="x_mean",
               neuron=Neurons.Sigmoid(),
               bottoms=[:l4_out], tops=[:recon])
]


function pretrain(layer_id::Int64)
    to_sym = symbol("l$(layer_id)_in")
    from_sym = layer_id > 1 ? symbol("l$(layer_id-1)_in") : :data

    noise = [
        RandomNormalLayer(tops=[:noise],
            output_dims=[28,28,1],
            batch_sizes=[batch_size]),
        PowerLayer(scale=0.5,
            tops=[:scaled_noise],
            bottoms=[:noise]),
        ElementWiseLayer(operation=ElementWiseFunctors.Add(),
            tops=[to_sym],
            bottoms=[from_sym, :scaled_noise])]

    rename_test = IdentityLayer(bottoms=[:data], tops=[to_sym])

    enc = original_layers[layer_id]
    dec = recon_layers[layer_id]
    loss = SquareLossLayer(bottoms=[:recon, to_sym])
    both = [enc, dec, loss]

    pretrain_net = Net("pretrain", backend, [train_dl, noise, both...])
    pretest_net = Net("pretest", backend, [test_dl, rename_test, both...])

    adam_instance = Adam()
    params = make_solver_parameters(adam_instance;
        max_iter=500_000,
        regu_coef=1e-4,
        lr_policy=LRPolicy.Fixed(1e-3))
    solver = Solver(adam_instance, params)
    add_coffee_break(solver, TrainingSummary(), every_n_iter=batch_size)
    add_coffee_break(solver, ValidationPerformance(pretest_net), every_n_iter=1000)
    solve(solver, pretrain_net)
end
=#


#=
base_dir = "snapshots_tinyencloss_z50_sqloss"
    setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)
=#

#=
    InnerProductLayer(name="dec1", output_dim=hidden_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:z], tops=[:dec1]),
    SplitLayer(name="dec1_split",
        bottoms=[:dec1], tops=[:dec1a, :dec1b]),
    InnerProductLayer(name="x_mean", output_dim=x_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:dec1a], tops=[:x_mean]),
    InnerProductLayer(name="x_sd", output_dim=x_dim,
        neuron=Neurons.Exponential(),
        bottoms=[:dec1b], tops=[:x_sd]),
    DecoderLossLayer(bottoms=[:x_mean, :x_sd, :data], weight=1.)
=#


#=
using JLD
jdl_file = jldopen(joinpath(base_dir, "snapshot-075000.jld"))
load_network(jdl_file, train_net)
close(jdl_file)

forward(train_net)
backward(train_net)
=#

#=
Profile.init(int(1e8), 0.001)
@profile solve(solver, train_net)
open("profile.txt", "w") do out
  Profile.print(out)
end
=#

destroy(train_net)
destroy(test_net)
shutdown(backend)
