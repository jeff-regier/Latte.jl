# Encoder Loss -- the KL diveregence btw z and N(0, I)

using Devectorize


@defstruct EncoderLossLayer Layer (
    name :: String = "encoder-loss",
    (weight :: FloatingPoint = 1.0, weight >= 0),
    (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(EncoderLossLayer,
    has_loss  => true,
    can_do_bp => true,
    is_sink   => true,
    has_stats => true,
)

type EncoderLossLayerState{T} <: LayerState
    layer      :: EncoderLossLayer
    loss       :: T
    loss_accum :: T
    n_accum    :: Int
end

function setup(backend::Backend, layer::EncoderLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    state = EncoderLossLayerState(layer, zero(data_type), zero(data_type), 0)
    return state
end
function shutdown(backend::Backend, state::EncoderLossLayerState)
end

function reset_statistics(state::EncoderLossLayerState)
    state.n_accum = 0
    state.loss_accum = zero(typeof(state.loss_accum))
end
function dump_statistics(storage, state::EncoderLossLayerState, show::Bool)
    update_statistics(storage, "$(state.layer.name)-encoder-loss", state.loss_accum)

    if show
      loss = @sprintf("%.4f", state.loss_accum)
      @info("  Encoder-loss (avg over $(state.n_accum)) = $loss")
    end
end

function forward(backend::CPUBackend, state::EncoderLossLayerState, inputs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    nn = length(inputs[1])
    mu = inputs[1].data
    sigma = inputs[2].data

    state.loss = zero(data_type)
    for i in 1:nn
        state.loss += mu[i]^2 + sigma[i]^2 - 2log(sigma[i]) - 1
    end
    state.loss *= 0.5 * state.layer.weight / n

    # accumulate statistics
    state.loss_accum *= state.n_accum
    state.loss_accum += state.loss * n
    state.loss_accum /= state.n_accum + n

    state.n_accum += n
end

function backward(backend::CPUBackend, state::EncoderLossLayerState, 
        inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    nn = length(inputs[1])
    mu = inputs[1].data
    sigma = inputs[2].data

    if isa(diffs[1], CPUBlob)
        diffs[1].data[:] = mu
        diffs[1].data[:] *= state.layer.weight / n
    end

    if isa(diffs[2], CPUBlob)
        sigma_diffs = diffs[2].data
        @devec sigma_diffs[:] = sigma - (1 ./ sigma)
        diffs[2].data[:] *= state.layer.weight / n
    end
end


