blas_set_num_threads(1)

using Mocha


# fix the random seed to make results reproducable
srand(12345678)

x_dim = 28^2
z_dim = 50

#TODO: use L1 regularizer for these fully connected layers
#TODO: use dropout layers (?)
#TODO: tie enc/dec weights?

backend = CPUBackend()
init(backend)

non_data_layers = [
    InnerProductLayer(name="enc1", output_dim=1200,
                param_key="enc1",
                neuron=Neurons.Sigmoid(),
                bottoms=[:data], tops=[:enc1]),
    SplitLayer(name="enc1_split",
               bottoms=[:enc1], tops=[:enc1a, :enc1b]),
    InnerProductLayer(name="z_mean", output_dim=z_dim,
                      neuron=Neurons.Identity(),
                      bottoms=[:enc1a], tops=[:z_mean]),
    SplitLayer(name="z_mean_split",
               bottoms=[:z_mean], tops=[:z_mean1, :z_mean2]),
    InnerProductLayer(name="z_sd", output_dim=z_dim, 
                      neuron=Neurons.ExpReLU(),
                      bottoms=[:enc1b], tops=[:z_sd]),
    SplitLayer(name="z_sd_split",
               bottoms=[:z_sd], tops=[:z_sd1, :z_sd2]),
    GaussianNoiseLayer(name="z", output_dim=z_dim,
                       bottoms=[:z_mean1, :z_sd1], tops=[:z]),
    EncoderLossLayer(name="encoder-loss",
                     weight=1.,
                     bottoms=[:z_mean2, :z_sd2]),

#=
    InnerProductLayer(name="dec1", output_dim=1200,
                      neuron=Neurons.Sigmoid(),
                      bottoms=[:z], tops=[:dec1]),
    SplitLayer(name="dec1_split",
               bottoms=[:dec1], tops=[:dec1a, :dec1b]),
    InnerProductLayer(name="x_mean", output_dim=x_dim,
                      neuron=Neurons.Sigmoid(),
                      bottoms=[:dec1a], tops=[:x_mean]),
    InnerProductLayer(name="x_sd", output_dim=x_dim,
                      neuron=Neurons.PSigmoid(),
                      bottoms=[:dec1b], tops=[:x_sd]),
    DecoderLossLayer(bottoms=[:x_mean, :x_sd, :data], weight=1.)
=#
    InnerProductLayer(name="dec", output_dim=x_dim,
                      neuron=Neurons.Sigmoid(),
                      bottoms=[:z], tops=[:dec]),
    SquareLossLayer(weight=1e-0, bottoms=[:dec, :data])
]


train_dl = HDF5DataLayer(name="train-data",
               source=ENV["DATASETS"]"/mnist/train.txt", 
               batch_size=100)
train_net = Net("mnist-train-aevb", backend, [train_dl, non_data_layers...])
init(train_net)


base_dir = "snapshots_tinyencloss_z50_sqloss"
adam_instance = Adam()
params = make_solver_parameters(adam_instance;
    max_iter=500_000,
    regu_coef=1e0,
#    load_from="snapshots_adam_reg1_defaulteps",
    lr_policy=LRPolicy.Fixed(1e-4))
solver = Solver(adam_instance, params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

test_txt = ENV["DATASETS"]"/mnist/test.txt"
test_dl = HDF5DataLayer(name="test-data", source=test_txt, batch_size=100)
test_net = Net("mnist-test-aevb", backend, [test_dl, non_data_layers...])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=500)

setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)

solve(solver, train_net)


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
