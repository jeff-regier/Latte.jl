@defstruct LookupTableLayer Layer (
    name :: AbstractString = "lookup-table",
    init :: Initializer = ConstantInitializer(0),
    lr   :: AbstractFloat = 1.0,
    (output_dim :: Int = 0, output_dim > 0),
    (tops :: Vector{Symbol} = Symbol[], length(tops) == 1),
    (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops) == 1),
    (n_class :: Int = 0, n_class > 0)
)

@characterize_layer(InnerProductLayer,
  can_do_bp  => true,
  has_param  => true,
)

type LookupTableLayerState <: LayerState
    layer      :: LookupTableLayer
    blobs      :: Vector{Blob}
    blobs_diff :: Vector{Blob}
    parameters :: Vector{Parameter}
    frozen     :: Bool
end

function setup(backend::Backend, layer::LookupTableLayer, inputs::Vector{Blob},
            diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    out_dim = layer.output_dim
    batch_size = get_num(inputs[1])

    blobs = [make_blob(backend, data_type, out_dim, batch_size)]
    blobs_diff = [make_blob(backend, data_type, out_dim, batch_size)]

    parameters = [make_parameter(backend, "lookup", data_type,
                           (out_dim, layer.n_class),
                           layer.init, layer.lr)]
    erase!(parameters[1].gradient)

    return LookupTableLayerState(layer, blobs, blobs_diff, parameters)
end

function shutdown(backend::Backend, state::LookupTableLayerState)
    map(destroy, state.blobs)
    map(destroy, state.blobs_diff)
    map(destroy, state.parameters)
end

function forward(backend::CPUBackend, state::LookupTableLayerState,
            inputs::Vector{Blob})
    output = state.blobs[1].data
    W = state.parameters[1].blob
    input = inputs[1].data

    for i in 1:length(input)
        idx = convert(Int, input[i])
        output[:, i] = W[:, idx]
    end
end

function backward(backend::Backend, state::LookupTableLayerState,
            inputs::Vector{Blob}, diffs::Vector{Blob})
    input = inputs[1].data
    ∇W = state.parameters[1].gradient
    diff = diffs[1].data

    if !state.frozen
        for i in 1:length(input)
            idx = convert(Int, input[i])
            fill!(∇W[:, idx].data, 0.)
        end

        for i in 1:length(input)
            idx = convert(Int, input[i])
            ∇W[:, idx] += diff[:, i]
        end
    end
end
