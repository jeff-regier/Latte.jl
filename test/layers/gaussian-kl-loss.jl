using Distributions


function test_gaussian_kl_loss_layer(backend::Backend, T, eps)
    println("-- Testing GaussianKLLossLayer on $(typeof(backend)){$T}...")

    ############################################################
    # Prepare Data for Testing
    ############################################################
    tensor_dim = abs(rand(Int)) % 4 + 2
    dims = tuple((abs(rand(Int,tensor_dim)) % 6 + 6)...)
    println("    > $dims")
    mus = rand(T, dims)
    sigmas = sqrt(rand(T, dims).^2)

    ############################################################
    # Setup
    ############################################################
    weight = 1.1
    layer  = GaussianKLLossLayer(; bottoms=[:predictions, :labels], weight=weight)
    mu_blob  = make_blob(backend, T, dims)
    sigma_blob = make_blob(backend, T, dims)

    mu_diff_blob  = make_blob(backend, T, dims)
    sigma_diff_blob  = make_blob(backend, T, dims)

    copy!(mu_blob, mus)
    copy!(sigma_blob, sigmas)
    inputs = Blob[mu_blob, sigma_blob]
    diffs = Blob[mu_diff_blob, sigma_diff_blob]

    state = setup(backend, layer, inputs, diffs)

    forward(backend, state, inputs)

    n = length(mu_blob)
    loss = 0.5(sum(mus.^2 + sigmas.^2 - 2log(sigmas)) - n)
    loss *= weight/get_num(mu_blob)
    @test -eps < loss-state.loss < eps

    backward(backend, state, inputs, diffs)
    grad = mus
    grad *= weight/get_num(mu_blob)
    diff = similar(grad)
    copy!(diff, diffs[1])
    @test all(-eps .< grad - diff .< eps)

    grad = sigmas - 1./sigmas
    grad *= weight/get_num(mu_blob)
    diff = similar(grad)
    copy!(diff, diffs[2])
    @test all(-eps .< grad - diff .< eps)

    shutdown(backend, state)
end

function verify_kl(q_dist, p_dist, claimed_kl)
    sample_size = 2_000_000
    q_samples = rand(q_dist, sample_size)
    empirical_kl_samples = logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
    @test_approx_eq_eps empirical_kl claimed_kl tol
end


function test_gaussian_kl_loss_layer_2(backend::Backend, T, eps)
    println("-- Testing GaussianKLLossLayer again on $(typeof(backend)){$T}...")

    ############################################################
    # Prepare Data for Testing
    ############################################################
    tensor_dim = 2
    dims = (p, n) = (5, 6)
    println("        > $dims")

    ############################################################
    # Setup
    ############################################################
    layer = GaussianKLLossLayer(bottoms=[:mean_dummy, :sd_dummy])

    mu = 5rand(T, dims) - 2
    sigma = 2rand(T, dims) + 0.01

    inputs = Blob[
        make_blob(backend, mu),    # mean blob
        make_blob(backend, sigma)]    # sd blob

    diffs = Blob[
        make_blob(backend, rand(T, dims) + 4), # mean diff
        make_blob(backend, 4rand(T, dims) - 7)]    # sd diff

    state = setup(backend, layer, inputs, diffs)

    ############################################################
    # Forward Propagation
    ############################################################
    forward(backend, state, inputs)

    loss = 0.
    for i in 1:n, j in 1:p
        loss -= 1 + 2log(sigma[j, i]) - mu[j, i]^2 - sigma[j, i]^2
    end
    loss /= 2dims[end]
    @test loss >= 0  # it's a KL divergence
    @test -eps < loss - state.loss < eps

    q_dist = MvNormal(mu[:], sigma[:])
    p_dist = MvNormal(zeros(n * p), 1)
    verify_kl(q_dist, p_dist, n * state.loss)

    ############################################################
    # Backward Propagation
    ############################################################
    backward(backend, state, inputs, diffs)
    grad_mu = mu / dims[end]
    grad_sigma = (sigma - 1./sigma) ./ dims[end]

    got_diff_mu = similar(grad_mu)
    copy!(got_diff_mu, diffs[1])
    @test all(-eps .< grad_mu - got_diff_mu .< eps)

    got_diff_sigma = similar(grad_sigma)
    copy!(got_diff_sigma, diffs[2])
    @test all(-eps .< grad_sigma - got_diff_sigma .< eps)

    for i in 1:2
        input_i = Array(T, size(inputs[i]))
        copy!(input_i, inputs[i])
        diff_i = Array(T, size(diffs[i]))
        copy!(diff_i, diffs[i])

        for k in 1:p, j in 1:n
            x0 = input_i[k, j]
            function f(x::FloatingPoint)
                input_i[k, j] = x
                copy!(inputs[i], input_i)
                forward(backend, state, inputs)
                input_i[k, j] = x0
                copy!(inputs[i], input_i)
                state.loss
            end
            test_deriv(f, x0, diff_i[k, j])
        end
    end

    ############################################################
    # Shutdown
    ############################################################
    shutdown(backend, state)
end

function test_gaussian_kl_loss_layer(backend::Backend)
  test_gaussian_kl_loss_layer(backend, Float32, 1e-2)
  test_gaussian_kl_loss_layer(backend, Float64, 1e-8)
#    test_gaussian_kl_loss_layer_2(backend, Float32, 1e-2)
    test_gaussian_kl_loss_layer_2(backend, Float64, 1e-8)
end

if test_gpu
  test_gaussian_kl_loss_layer(backend_gpu)
end

if test_cpu
  test_gaussian_kl_loss_layer(backend_cpu)
end

