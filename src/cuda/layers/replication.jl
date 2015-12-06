function setup_etc(backend::GPUBackend, layer::ReplicationLayer,
        inputs::Vector{Blob}, diffs::Vector{Blob})

    P, N = size(inputs[1])
    K = layer.num_copies

    mat = zeros(K, P)
    
    for i in 1:n, j in 1:p, k in 1:K
        mat[k, j, i] = 1.
    end

    mat
end

function forward(backend::GPUBackend, state::ReplicationLayerState,
            inputs::Vector{Blob})
    input = inputs[1]
    output = state.blobs[1]

    p, n = size(input)
    for i in 1:n, j in 1:p
        output.data[:, j, i] = input.data[j, i]
    end
end

function backward(backend::GPUBackend, state::ReplicationLayerState,
        inputs::Vector{Blob}, diffs::Vector{Blob})

    if !isa(diffs[1], NullBlob)
        data_type = eltype(inputs[1])
        p, n = size(inputs[1])

        for i in 1:n, j in 1:p
            diffs[1].data[j, i] = sum(state.blobs_diff[1].data[:, j, i])
        end
    end
end

