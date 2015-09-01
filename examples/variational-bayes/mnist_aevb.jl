ENV["MOCHA_USE_NATIVE_EXT"] = "false"
ENV["OMP_NUM_THREADS"] = 4
#blas_set_num_threads(4)

using Mocha


# fix the random seed to make results reproducable
srand(12345678)

train_txt = ENV["DATASETS"]"/mnist/train.txt"
data_layer  = HDF5DataLayer(name="train-data", source=train_txt, batch_size=100)

z_dim = 2

#TODO: use L1 regularizer for these fully connected layers
#TODO: use dropout layers (?)
#TODO: tie enc/dec weights?

enc1_layer     = InnerProductLayer(name="enc1", output_dim=1200,
                            neuron=Neurons.Sigmoid(),
                            weight_init = GaussianInitializer(std=0.01),
                            bottoms=[:data], tops=[:enc1])
z_mean_layer   = InnerProductLayer(name="z_mean", output_dim=z_dim, 
                            neuron=Neurons.Sigmoid(),
                            weight_init = GaussianInitializer(std=0.01),
                            bottoms=[:enc1], tops=[:z_mean])
#=
z_sd_layer     = InnerProductLayer(name="z_sd", output_dim=z_dim, 
                            neuron=Neurons.Exp(),
                            weight_init = GaussianInitializer(std=0.01),
                            weight_cons = L2Cons(4.5),
                            bottoms=[:enc2], tops=[:z_sd])
z_sample_layer = GaussianNoiseLayer(name="z_sample", output_dim=z_dim,
                            bottoms=[:mean, :sd], tops=[:z_sample])
=#
dec1_layer     = InnerProductLayer(name="dec1", output_dim=1200,
                            neuron=Neurons.Sigmoid(),
                            weight_init = GaussianInitializer(std=0.01),
                            bottoms=[:z_mean], tops=[:dec1])
dec2_layer     = InnerProductLayer(name="dec2", output_dim=(28^2),
                            neuron=Neurons.Sigmoid(),
                            weight_init = GaussianInitializer(std=0.01),
                            bottoms=[:dec1], tops=[:dec2])
recon_loss_layer = SquareLossLayer(bottoms=[:dec2, :data])


backend = CPUBackend()
init(backend)

# put training net together, note that the correct ordering will automatically be established by the constructor
non_data_layers = [enc1_layer, z_mean_layer, dec1_layer, dec2_layer, recon_loss_layer]
net = Net("MNIST-aevb", backend, [data_layer, non_data_layers...])

base_dir = "snapshots_aevb"
# we let the learning rate decrease by 0.998 in each epoch (=600 batches of size 100)
# and let the momentum increase linearly from 0.5 to 0.9 over 500 epochs
# which is equivalent to an increase step of 0.0008
# training is done for 2000 epochs
params = SolverParameters(max_iter=600*2000, regu_coef=0.0,
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

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
