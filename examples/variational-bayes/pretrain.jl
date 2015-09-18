using Mocha


backend = GPUBackend()
init(backend)

x_dim = 28^2
z_dim = 10
hidden_dim = 128

batch_size = 100


train_dl = HDF5DataLayer(name="train-data",
               source=ENV["DATASETS"]"/mnist/train.txt",
               batch_size=batch_size)
test_dl = HDF5DataLayer(name="test-data", 
               source=ENV["DATASETS"]"/mnist/test.txt",
               batch_size=batch_size)

original_layers = [
    InnerProductLayer(name="enc_hidden",
               output_dim=hidden_dim,
               param_key="enc_hidden",
               neuron=Neurons.ReLU(),
               bottoms=[:l0_out], tops=[:l1_out]),
    InnerProductLayer(name="z_mean",
               output_dim=z_dim,
               param_key="z_mean",
               neuron=Neurons.Tanh(),
               bottoms=[:l1_out], tops=[:l2_out]),
    InnerProductLayer(name="dec_hidden",
               output_dim=hidden_dim,
               param_key="dec_hidden",
               neuron=Neurons.ReLU(),
               bottoms=[:l2_out], tops=[:l3_out]),
    InnerProductLayer(name="x_mean",
               output_dim=x_dim,
               param_key="x_mean",
               neuron=Neurons.Sigmoid(),
               bottoms=[:l3_out], tops=[:l4_out])
]


function solve_net(middle_layers, max_iter)
    rename_layer = IdentityLayer(bottoms=[:data], tops=[:l0_out])
    loss_layer = SquareLossLayer(bottoms=[:l4_out, :data])

    train_net = Net("train_net", backend, [
        train_dl,
        rename_layer,
        middle_layers...,
        loss_layer])

    test_net = Net("test_net", backend, [
        test_dl,
        rename_layer,
        middle_layers,
        loss_layer])

    adam_instance = Adam()
    params = make_solver_parameters(adam_instance;
        max_iter=max_iter,
        regu_coef=1e-4,
        lr_policy=LRPolicy.Fixed(1e-3))
    solver = Solver(adam_instance, params)
    add_coffee_break(solver, TrainingSummary(), every_n_iter=batch_size)
    add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

    solve(solver, train_net)
end


solve_net(original_layers, 20_000)

###############

z_dim_2 = 2

new_layers = [
    original_layers[1],
    InnerProductLayer(name="z_mean_2",
        output_dim=z_dim_2,
        neuron=Neurons.Tanh(),
        bottoms=[:l1_out], tops=[:l2_out]),
    InnerProductLayer(name="dec_hidden_2",
            output_dim=hidden_dim,
        neuron=Neurons.ReLU(),
        bottoms=[:l2_out], tops=[:l3_out]),
    original_layers[4]
]

solve_net(new_layers, 200_000)



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
base_dir = "snapshots"
setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)

solve(solver, pretrain_net)

pretrain_net = Net("pretrain_enc1_trivial", backend,
    [train_dl, train_noise_layers..., coder_layers...])
=#
