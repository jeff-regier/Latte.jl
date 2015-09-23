using Distributions


function test_bernoulli_recon_loss_layer(backend::Backend, T, eps)
    println("-- Testing BernoulliReconLossLayer on $(typeof(backend)){$T}...")

    ############################################################
    # Prepare Data for Testing
    ############################################################
    dims = (3, 4, 5)
    tensor_dim = length(dims)
    D, N = prod(dims[1:end-1]), dims[end]
    ND = N * D
    println("        > $dims")

    ############################################################
    # Setup
    ############################################################
    layer = BernoulliReconLossLayer(bottoms=[:p, :x])

    p = min(1 - 1e-4, max(1e-4, rand(D, N)))
    x = round(rand(dims), 0)

    inputs = Blob[
        make_blob(backend, convert(Array{T}, p)),
        make_blob(backend, convert(Array{T}, x))]

    diffs = Blob[
        make_blob(backend, rand(T, size(inputs[1]))),  # gets overwritten
        NullBlob()]  # the data is fixed

    state = setup(backend, layer, inputs, diffs)

    ############################################################
    # Forward Propagation
    ############################################################
    forward(backend, state, inputs)

    loss = 0.
    for i in 1:ND
        Xi_given_zi = Bernoulli(p[i])  # a random var
        loss -= logpdf(Xi_given_zi, x[i])
    end
    loss /= N
    @test loss >= 0

    info("expected $loss; got $(state.loss)")
    @test -eps < loss - state.loss < eps

    ############################################################
    # Backward Propagation
    ############################################################
    backward(backend, state, inputs, diffs)

    input1 = Array(T, size(inputs[1]))
    copy!(input1, inputs[1])
    diff1 = Array(T, size(diffs[1]))
    copy!(diff1, diffs[1])

    for k in 1:D, j in 1:N
        x0 = input1[k, j]
        function f(x::FloatingPoint)
            input1[k, j] = x
            copy!(inputs[1], input1)
            forward(backend, state, inputs)
            input1[k, j] = x0
            copy!(inputs[1], input1)
            state.loss
        end
        test_deriv(f, x0, diff1[k, j])
    end

    ############################################################
    # Shutdown
    ############################################################
    shutdown(backend, state)
end


function test_bernoulli_hyperparameter(backend::Backend)
    # setup
    D = 3
    p = min(1 - 1e-4, max(1e-4, rand(D, 1)))
    s = rand(D, 1)

    ############################################################
    # Forward Propagation
    ############################################################

    # get claimed
    inputs = Blob[make_blob(backend, p), make_blob(backend, s)]
    layer = BernoulliReconLossLayer(bottoms=[:p, :x])
    diffs = Blob[
        make_blob(backend, rand(D, 1)),
        NullBlob()]
    state = setup(backend, layer, inputs, diffs)
    forward(backend, state, inputs)

    # get expected
    losses = Float64[]
    num_x_samples = 100_000
    X_given_z = [Bernoulli(p[i]) for i in 1:D]
    for k in 1:num_x_samples
        x_sample = round(rand(D) .< s, 0)
        loss = 0.
        for i in 1:D
            loss -= logpdf(X_given_z[i], x_sample[i])
        end
        push!(losses, loss)
    end
    ci_radius = 5std(losses) / sqrt(num_x_samples)

    # do test
    println("$(state.loss) == $(mean(losses)) +/- $ci_radius")
    @test mean(losses) - ci_radius <= state.loss <= mean(losses) + ci_radius

    ############################################################
    # Backward Propagation
    ############################################################
    backward(backend, state, inputs, diffs)

    input1 = Array(Float64, size(inputs[1]))
    copy!(input1, inputs[1])
    diff1 = Array(Float64, size(diffs[1]))
    copy!(diff1, diffs[1])

    for k in 1:D, j in 1:1
        x0 = input1[k, j]
        function f(x::FloatingPoint)
            input1[k, j] = x
            copy!(inputs[1], input1)
            forward(backend, state, inputs)
            input1[k, j] = x0
            copy!(inputs[1], input1)
            state.loss
        end
        test_deriv(f, x0, diff1[k, j])
    end

    ############################################################
    # Shutdown
    ############################################################
    shutdown(backend, state)
end


function test_bernoulli_recon_loss_layer(backend::Backend)
    test_bernoulli_recon_loss_layer(backend, Float64, 1e-8)
    test_bernoulli_hyperparameter(backend)
end

if test_cpu
    test_bernoulli_recon_loss_layer(backend_cpu)
end
if test_gpu
    test_bernoulli_recon_loss_layer(backend_gpu)
end

