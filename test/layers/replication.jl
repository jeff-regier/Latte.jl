using Distributions


function test_replication_layer(backend::Backend, T, eps)
    println("-- Testing ReplicationLayer on $(typeof(backend)){$T}...")

    ############################################################
    # Prepare Data for Testing
    ############################################################
    dims = (4, 5)
    tensor_dim = length(dims)
    D, N = prod(dims[1:end-1]), dims[end]
    ND = N * D
    K = 2  # K for "kopy"
    println("        > $dims")

    ############################################################
    # Setup
    ############################################################
    layer = ReplicationLayer(num_copies=K, bottoms=[:d0], tops=[:d1])

    input_data = rand(dims)
    inputs = Blob[make_blob(backend, convert(Array{T}, input_data))]
    diffs = Blob[make_blob(backend, rand(T, size(inputs[1])))]

    state = setup(backend, layer, inputs, diffs)

    ############################################################
    # Forward Propagation
    ############################################################
    forward(backend, state, inputs)

    rep_data = Array(T, size(state.blobs[1]))
    copy!(rep_data, state.blobs[1])

    for i in 1:N, j in 1:D
        for k in 1:K
            @test_approx_eq rep_data[k, j, i] input_data[j, i]
        end
    end

    ############################################################
    # Backward Propagation
    ############################################################
    # equally senstive to all outputs
    copy!(state.blobs_diff[1], ones(K, D, N))
    backward(backend, state, inputs, diffs)

    input1 = Array(T, size(inputs[1]))
    copy!(input1, inputs[1])
    diff1 = Array(T, size(diffs[1]))
    copy!(diff1, diffs[1])
    output = Array(T, size(state.blobs[1]))

    for j in 1:D, i in 1:N
        x0 = input1[j, i]
        function f(x::AbstractFloat)
            input1[j, i] = x
            copy!(inputs[1], input1)
            forward(backend, state, inputs)
            input1[j, i] = x0
            copy!(inputs[1], input1)

            copy!(output, state.blobs[1])
            sum(output)
        end
        test_deriv(f, x0, diff1[j, i])
    end

    ############################################################
    # Shutdown
    ############################################################
    shutdown(backend, state)
end


function test_replication_layer(backend::Backend)
    test_replication_layer(backend, Float64, 1e-8)
end

if test_cpu
    test_replication_layer(backend_cpu)
end
if test_gpu
#    test_replication_layer(backend_gpu)
end

