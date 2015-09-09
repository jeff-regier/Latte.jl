function test_gaussian_noise_layer(backend::Backend, T, eps)
    println("-- Testing Gaussian noise on $(typeof(backend)){$T}...")

    tensor_dim = abs(rand(Int)) % 6 + 1
    dims = tuple((abs(rand(Int, tensor_dim)) % 8 + 1)...)
    println("    > $dims")

    inputs = Blob[
        make_blob(backend, 5rand(T, dims) - 2),  # mean blob
        make_blob(backend, 2rand(T, dims) + 2.1)]  # sd blob

    diffs = Blob[
        make_blob(backend, rand(T, dims) + 4), # mean diff
        make_blob(backend, 4rand(T, dims) - 7)]  # sd diff
    output_dim = 1 + (abs(rand(Int)) % 20)

    println("    > Setup")
    layer = GaussianNoiseLayer(bottoms=[:z_mean, :z_sd], tops=[:z],
        output_dim=output_dim)
    state = setup(backend, layer, inputs, diffs)

    println("    > Forward")
    forward(backend, state, inputs)
    got_output = zeros(T, dims)
    copy!(got_output, state.blobs[1])
    mvn_sample = zeros(T, dims)
    copy!(mvn_sample, state.mvn_sample)

    deviation = mvn_sample .* inputs[2].data
    expected_output = inputs[1].data .+ deviation
    @test all(abs(got_output - expected_output) .< eps)

    println("    > Backward")
    top_diff = rand(T, dims)
    copy!(state.blobs_diff[1], top_diff)
    backward(backend, state, inputs, diffs)

    expected_mu_grad = top_diff
    got_mu_grad = zeros(T, dims)
    copy!(got_mu_grad, diffs[1])
    @test all(abs(got_mu_grad - expected_mu_grad) .< eps)

    expected_sigma_grad = top_diff .* mvn_sample
    got_sigma_grad = zeros(T, dims)
    copy!(got_sigma_grad, diffs[2])
    @test all(abs(got_sigma_grad - expected_sigma_grad) .< eps)

    shutdown(backend, state)
end

function test_gaussian_noise_layer(backend::Backend)
    test_gaussian_noise_layer(backend, Float64, 1e-10)
    test_gaussian_noise_layer(backend, Float32, 1e-4)
end

function verify_gaussian_noise_derivs(backend::Backend)
    input_dim = abs(rand(Int) % 3) + 1
    batch_size = abs(rand(Int) % 3) + 1
    inputs = Blob[make_blob(backend, randn(input_dim, batch_size)),
                  make_blob(backend, rand(input_dim, batch_size) + 1e-3)]
    diffs = Blob[make_blob(backend, 100randn(input_dim, batch_size)),
                  make_blob(backend, 100randn(input_dim, batch_size))]
    layer = GaussianNoiseLayer(name="z", output_dim=input_dim,
                           bottoms=[:z_mean, :z_sd], tops=[:z])
    state = setup(backend, layer, inputs, diffs)

    for i_in in 1:2, j_in in 1:input_dim, k_in in 1:batch_size
        for i_out in 1:1, j_out in 1:input_dim, k_out in 1:batch_size
            x0 = inputs[i_in].data[j_in, k_in]
            forward(backend, state, inputs)

            fill!(state.blobs_diff[i_out].data, 0.)
            state.blobs_diff[i_out].data[j_out, k_out] = 1.
            backward(backend, state, inputs, diffs)
            dfdx = diffs[i_in].data[j_in, k_in]

            function f(x::FloatingPoint)
                inputs[i_in].data[j_in, k_in] = x
                Mocha.do_forward(state, inputs)
                inputs[i_in].data[j_in, k_in] = x0
                state.blobs[i_out].data[j_out, k_out]
            end

            test_deriv(f, x0, dfdx)
        end
    end
end

if test_cpu
    test_gaussian_noise_layer(backend_cpu)
    verify_gaussian_noise_derivs(backend_cpu)
end
if test_gpu
    test_gaussian_noise_layer(backend_gpu)
    verify_gaussian_noise_derivs(backend_gpu)
end
