function test_encoder_loss_layer(backend::Backend, T, eps)
    println("-- Testing EncoderLossLayer on $(typeof(backend)){$T}...")

    ############################################################
    # Prepare Data for Testing
    ############################################################
    tensor_dim = 2
    dims = tuple((abs(rand(Int, tensor_dim)) % 6 + 6)...)
    println("        > $dims")

    ############################################################
    # Setup
    ############################################################
    layer = EncoderLossLayer(bottoms=[:z_mean, :z_sd])

    mu = 5rand(T, dims) - 2
    sigma = 2rand(T, dims) + 0.01

    inputs = Blob[
        make_blob(backend, mu),    # mean blob
        make_blob(backend, sigma)]    # sd blob

    diffs = Blob[
        make_blob(backend, rand(T, dims) + 4), # mean diff
        make_blob(backend, 4rand(T, dims) - 7)]    # sd diff

    state = setup(backend, layer, inputs, diffs)

    forward(backend, state, inputs)

    loss = 0.
    for i in 1:dims[end]
        for j in 1:dims[1]
            loss -= 1 + 2log(sigma[j, i]) - mu[j, i]^2 - sigma[j, i]^2
        end
    end
    loss /= dims[end]
    @test loss >= 0  # it's 2 * a KL divergence

    @test -eps < loss - state.loss < eps

    backward(backend, state, inputs, diffs)
    grad_mu = 2mu / dims[end]
    grad_sigma = (2sigma - 2./sigma) ./ dims[end]

    got_diff_mu = similar(grad_mu)
    copy!(got_diff_mu, diffs[1])
    @test all(-eps .< grad_mu - got_diff_mu .< eps)

    got_diff_sigma = similar(grad_sigma)
    copy!(got_diff_sigma, diffs[2])
    @test all(-eps .< grad_sigma - got_diff_sigma .< eps)

    shutdown(backend, state)
end

function test_encoder_loss_layer(backend::Backend)
    test_encoder_loss_layer(backend, Float32, 1e-2)
    test_encoder_loss_layer(backend, Float64, 1e-8)
end

function verify_encoder_loss_derivs(backend::Backend)
    input_dim = abs(rand(Int) % 5) + 1
    batch_size = abs(rand(Int) % 5) + 1
    inputs = Blob[make_blob(backend, randn(input_dim, batch_size)),
                  make_blob(backend, rand(input_dim, batch_size) + 1e-3)]
    diffs = Blob[make_blob(backend, 100randn(input_dim, batch_size)),
                  make_blob(backend, 100randn(input_dim, batch_size))]
    layer = EncoderLossLayer(bottoms=[:mean_dummy, :sd_dummy])
    state = setup(backend, layer, inputs, diffs)

    for i in 1:2, j in 1:input_dim, k in 1:batch_size
        x0 = inputs[i].data[j, k]
        backward(backend, state, inputs, diffs)
        dfdx = diffs[i].data[j, k]
        function f(x::FloatingPoint)
            inputs[i].data[j, k] = x
            forward(backend, state, inputs)
            inputs[i].data[j, k] = x0
            state.loss
        end
        test_deriv(f, x0, dfdx)
    end
end

if test_cpu
    test_encoder_loss_layer(backend_cpu)
    verify_encoder_loss_derivs(backend_cpu)
end
if test_gpu
    test_encoder_loss_layer(backend_gpu)
    verify_encoder_loss_derivs(backend_gpu)
end
