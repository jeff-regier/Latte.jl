using Devectorize


@defstruct GaussianNoiseLayer Layer (
    name :: String = "gaussian",
    (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
    (tops:: Vector{Symbol} = Symbol[], length(tops) == 1),
    (output_dim :: Int = 0, output_dim > 0)
)

@characterize_layer(GaussianNoiseLayer,
    can_do_bp    => true
)


type GaussianNoiseLayerState <: LayerState
    layer       :: GaussianNoiseLayer
    blobs      :: Vector{Blob}
    blobs_diff :: Vector{Blob}
    mvn_sample  :: Blob
end


function setup(backend::Backend, layer::GaussianNoiseLayer, inputs::Vector{Blob}, 
        diffs::Vector{Blob})
    float_type = eltype(inputs[1])
    mvn_sample = make_blob(backend, float_type, size(inputs[1]))

    # make sure all input blobs has the same dimensions
    @assert length(inputs) == 2
    @assert ndims(inputs[1]) == ndims(inputs[2])
    @assert size(inputs[1]) == size(inputs[2])

    my_dim = size(inputs[1])

    blobs = Blob[make_blob(backend, float_type, my_dim)]
    blobs_diff = Blob[make_blob(backend, float_type, my_dim)]

    return GaussianNoiseLayerState(layer, blobs, blobs_diff, mvn_sample)
end

function shutdown(backend::Backend, state::GaussianNoiseLayerState)
    destroy(state.mvn_sample)
    map(destroy, state.blobs)
    map(destroy, state.blobs_diff)
end


function do_forward{T}(mean_in::Array{T}, sd_in::Array{T}, 
        epsilon::Array{T}, z_sample_out::Array{T})
    @devec z_sample_out[:] = mean_in .+ (epsilon .* sd_in)
end

function forward(backend::CPUBackend, state::GaussianNoiseLayerState,
        inputs::Vector{Blob})
    for i in 1:length(state.mvn_sample.data)
        state.mvn_sample.data[i] = randn()
    end

    do_forward(inputs[1].data, inputs[2].data, state.mvn_sample.data, 
        state.blobs[1].data)
end


function do_backward{T}(d_mean::Array{T}, d_sd::Array{T}, 
        d_z_sample::Array{T}, epsilon::Array{T})
    d_mean[:] = d_z_sample
    @devec d_sd[:] = d_z_sample .* epsilon
end

function backward(backend::CPUBackend, state::GaussianNoiseLayerState,
        inputs::Vector{Blob}, diffs::Vector{Blob})
    do_backward(diffs[1].data, diffs[2].data, state.blobs_diff[1].data, 
        state.mvn_sample.data)
end

