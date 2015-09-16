using Mocha


backend = CPUBackend()
init(backend)

x_dim = 28^2

batch_size=100

train_dl = HDF5DataLayer(name="train-data",
               source=ENV["DATASETS"]"/mnist/train.txt",
               batch_size=batch_size)
train_noise_layers = [
    RandomNormalLayer(tops=[:noise],
        output_dims=[28,28,1],
        batch_sizes=[batch_size],
        eltype=Float64),
    PowerLayer(scale=0.02,
        tops=[:noise2],
        bottoms=[:noise]),
    ElementWiseLayer(operation=ElementWiseFunctors.Add(),
        tops=[:corrupted],
        bottoms=[:data, :noise2])]
coder_layers = [
    InnerProductLayer(name="enc1", output_dim=1200,
                param_key="enc1",
                neuron=Neurons.Sigmoid(),
                bottoms=[:corrupted], tops=[:enc1]),
    TiedInnerProductLayer(name="recon",
        tied_param_key="enc1",
        bottoms=[:enc1],
        tops=[:recon]),
    SquareLossLayer(bottoms=[:recon, :data])]
rename_layer = IdentityLayer(bottoms=[:data], tops=[:corrupted])


pretrain_net = Net("pretrain_enc1_trivial", backend,
    [train_dl, train_noise_layers..., coder_layers...])
init(pretrain_net)


base_dir = "snapshots_pretrain_enc1_02noise"
adam_instance = Adam()
params = make_solver_parameters(adam_instance;
    max_iter=500_000,
    regu_coef=1e0,
    lr_policy=LRPolicy.Fixed(1e-4))
solver = Solver(adam_instance, params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=batch_size)


test_txt = ENV["DATASETS"]"/mnist/test.txt"
test_dl = HDF5DataLayer(name="test-data", source=test_txt, batch_size=batch_size)
test_net = Net("mnist-test-aevb", backend, [test_dl, rename_layer, coder_layers...])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)


setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)


solve(solver, pretrain_net)

