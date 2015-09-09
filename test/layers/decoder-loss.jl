using Distributions


function test_decoder_loss_layer(backend::Backend, T, eps)
    println("-- Testing DecoderLossLayer on $(typeof(backend)){$T}...")

    ############################################################
    # Prepare Data for Testing
    ############################################################
    tensor_dim = 3
    dims = (3, 4, 5)
    p, n = prod(dims[1:end-1]), dims[end]
    println("        > $dims")

    ############################################################
    # Setup
    ############################################################
    layer = DecoderLossLayer(bottoms=[:mu, :x])

    mu = max(1e-2, rand(p, n))  # sigmoid output is in [0, 1]
    x = max(1e-2, rand(dims))  # normalized pixel values are in [0, 1]

    inputs = Blob[
        make_blob(backend, convert(Array{T}, mu)),
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
    for i in 1:n
        Xi_given_zi = MvNormal(mu[:, i], sqrt(mu[:, i]))  # a random var
        xi = slicedim(x, tensor_dim, i)[:]  # an instance
        loss -= logpdf(Xi_given_zi, xi)
    end
    loss /= n
    @test loss >= 0

    @test -eps < loss - state.loss < eps

    ############################################################
    # Backward Propagation
    ############################################################
    for j in 1:p, i in 1:n
        x0 = inputs[1].data[j, i]
        backward(backend, state, inputs, diffs)
        dfdx = diffs[1].data[j, i]
        function f(x::FloatingPoint)
            inputs[1].data[j, i] = x
            forward(backend, state, inputs)
            inputs[1].data[j, i] = x0
            state.loss
        end
        test_deriv(f, x0, dfdx)
    end

    ############################################################
    # Shutdown
    ############################################################
    shutdown(backend, state)
end

function test_decoder_loss_layer(backend::Backend)
#    test_decoder_loss_layer(backend, Float32, 1e-3)
    test_decoder_loss_layer(backend, Float64, 1e-8)
end

if test_cpu
    test_decoder_loss_layer(backend_cpu)
end
if test_gpu
    test_decoder_loss_layer(backend_gpu)
end

