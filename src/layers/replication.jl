@defstruct ReplicationLayer Layer (
    name :: AbstractString = "power",
    (num_copies :: Int = 1, num_copies > 0),
    (tops :: Vector{Symbol} = Symbol[], length(tops) == 1),
    (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
)
@characterize_layer(ReplicationLayer,
    can_do_bp => true
)

type ReplicationLayerState <: LayerState
    layer      :: ReplicationLayer
    blobs      :: Vector{Blob}
    blobs_diff :: Vector{Blob}
    etc        :: Any
end


function setup_etc(backend::CPUBackend, layer::ReplicationLayer,
        inputs::Vector{Blob}, diffs::Vector{Blob})
    nothing
end

function setup(backend::Backend, layer::ReplicationLayer, inputs::Vector{Blob},
                diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    @assert ndims(inputs[1]) == 2
    output_dim = (layer.num_copies, size(inputs[1])...)
    blobs = [make_blob(backend, data_type, output_dim)]

    if all(map(b -> isa(b, NullBlob), diffs))
        blobs_diff = Blob[NullBlob()]
    else
        blobs_diff = [make_blob(backend, data_type, output_dim)]
    end

    etc = setup_etc(backend, layer, inputs, diffs)
    state = ReplicationLayerState(layer, blobs, blobs_diff, etc)
end

function shutdown(backend::Backend, state::ReplicationLayerState)
    map(destroy, state.blobs)
    map(destroy, state.blobs_diff)
end

function forward(backend::CPUBackend, state::ReplicationLayerState,
            inputs::Vector{Blob})
    input = inputs[1]
    output = state.blobs[1]

    p, n = size(input)
    for i in 1:n, j in 1:p
        output.data[:, j, i] = input.data[j, i]
    end
end

function backward(backend::CPUBackend, state::ReplicationLayerState,
        inputs::Vector{Blob}, diffs::Vector{Blob})

    if !isa(diffs[1], NullBlob)
        data_type = eltype(inputs[1])
        p, n = size(inputs[1])

        for i in 1:n, j in 1:p
            diffs[1].data[j, i] = sum(state.blobs_diff[1].data[:, j, i])
        end
    end
end

