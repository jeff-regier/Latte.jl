using Mocha

backend = GPUBackend()
init(backend)

include("mnist-layers.jl")

train_net = Net("train-net", backend, [train_dl, non_data_layers...])
test_net = Net("test-net", backend, [test_dl, non_data_layers...])

adam_instance = Adam()
params = make_solver_parameters(adam_instance;
    max_iter=500_000,
    regu_coef=1e-4,
    lr_policy=LRPolicy.Fixed(1e-3))
solver = Solver(adam_instance, params)
add_coffee_break(solver, TrainingSummary(), every_n_iter=10000)
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=10000)

base_dir = "snapshots_tues"
setup_coffee_lounge(solver, save_into="$base_dir/statistics.jld", every_n_iter=10000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=10000)

solve(solver, train_net)

destroy(train_net)
destroy(test_net)


shutdown(backend)

1
