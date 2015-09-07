#ENV["MOCHA_USE_NATIVE_EXT"] = "false"
#ENV["OMP_NUM_THREADS"] = 3
#blas_set_num_threads(3)

using Mocha


# fix the random seed to make results reproducable
srand(12345678)

z_dim = 2

#TODO: use L1 regularizer for these fully connected layers
#TODO: use dropout layers (?)
#TODO: tie enc/dec weights?

non_data_layers = [
    InnerProductLayer(name="enc1", output_dim=1200,
                neuron=Neurons.Sigmoid(),
                bottoms=[:data], tops=[:enc1]),
    SplitLayer(name="enc1_split",
               bottoms=[:enc1], tops=[:enc1a, :enc1b]),
    InnerProductLayer(name="z_mean", output_dim=z_dim, 
                      neuron=Neurons.Identity(), #TODO: Inverse Normal CDF?
                      bottoms=[:enc1a], tops=[:z_mean]),
    InnerProductLayer(name="z_sd", output_dim=z_dim, 
                      neuron=Neurons.Exp(),
                      bottoms=[:enc1b], tops=[:z_sd]),
    GaussianNoiseLayer(name="z", output_dim=z_dim,
                       bottoms=[:z_mean, :z_sd], tops=[:z]),
    #EncoderLossLayer(bottoms=[:z_mean, :z_sd]),

    InnerProductLayer(name="dec1", output_dim=1200,
                      neuron=Neurons.Sigmoid(),
                      bottoms=[:z], tops=[:dec1]),
    InnerProductLayer(name="recon", output_dim=(28^2),
                      neuron=Neurons.Sigmoid(),
                      bottoms=[:dec1], tops=[:recon]),

    #DecoderLossLayer(bottoms=[:data, :recon]),
    SquareLossLayer(bottoms=[:recon, :data])
]


backend = CPUBackend()
init(backend)

training_data_layer = HDF5DataLayer(name="train-data",
                source=ENV["DATASETS"]"/mnist/train.txt", 
                batch_size=100)
train_net = Net("MNIST-aevb", backend, [training_data_layer, non_data_layers...])
init(train_net)


forward(train_net)


base_dir = "snapshots_aevb_profile"
# we let the learning rate decrease by 0.998 in each epoch (=600 batches of size 100)
# and let the momentum increase linearly from 0.5 to 0.9 over 500 epochs
# which is equivalent to an increase step of 0.0008
# training is done for 2000 epochs
params = SolverParameters(max_iter=5000*10, regu_coef=0.0,
                          mom_policy=MomPolicy.Linear(0.5, 0.0008, 600, 0.9),
                          lr_policy=LRPolicy.Step(0.1, 0.998, 600),
                          load_from=base_dir)
solver = SGD(params)


setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=5000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=5000)

# show performance on test data every 600 iterations (one epoch)
test_txt = ENV["DATASETS"]"/mnist/test.txt"
data_layer_test = HDF5DataLayer(name="test-data", source=test_txt, batch_size=100)
test_net = Net("MNIST-test", backend, [data_layer_test, non_data_layers...])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=600)

solve(solver, train_net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, train_net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(train_net)
destroy(test_net)
shutdown(backend)
