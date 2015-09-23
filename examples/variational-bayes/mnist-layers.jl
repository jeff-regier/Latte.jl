x_dim = 28^2
z_dim = 2
hidden_dim = 128
batch_size = 100

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
    InnerProductLayer(name="x_mean", output_dim=x_dim,
        neuron=Neurons.Sigmoid(),
        bottoms=[:dec1], tops=[:x_mean])]

expected_recon_loss = BernoulliReconLossLayer(bottoms=[:x_mean, :data])


non_data_layers = [
    enc_layers...,
    sampling_layers...,
    dec_layers...,
    expected_recon_loss]
