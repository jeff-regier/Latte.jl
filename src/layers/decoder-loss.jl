# Decoder Loss -- p_\theta(x | z)

using Devectorize


@defstruct DecoderLossLayer Layer (
    name :: String = "decoder-loss",
    (weight :: FloatingPoint = 1.0, weight >= 0),
    (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 3),
)
@characterize_layer(DecoderLossLayer,
    has_loss  => true,
    can_do_bp => true,
    is_sink   => true,
    has_stats => true,
)

type DecoderLossLayerState{T} <: LayerState
    layer      :: DecoderLossLayer
    loss       :: T
    loss_accum :: T
    n_accum    :: Int
end

function setup(backend::Backend, layer::DecoderLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    state = DecoderLossLayerState(layer, zero(data_type), zero(data_type), 0)
    return state
end
function shutdown(backend::Backend, state::DecoderLossLayerState)
end

function reset_statistics(state::DecoderLossLayerState)
    state.n_accum = 0
    state.loss_accum = zero(typeof(state.loss_accum))
end
function dump_statistics(storage, state::DecoderLossLayerState, show::Bool)
    update_statistics(storage, "$(state.layer.name)-decoder-loss", state.loss_accum)

    if show
      loss = @sprintf("%.4f", state.loss_accum)
      @info("  Decoder-loss (avg over $(state.n_accum)) = $loss")
    end
end

function forward(backend::CPUBackend, state::DecoderLossLayerState, inputs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    mu = inputs[1].data
    sigma = inputs[2].data
    x = inputs[3].data

    state.loss = 0
    for i in 1:length(mu)
        state.loss += log(2pi) + 2log(sigma[i]) + (x[i] - mu[i])^2 / sigma[i]^2
    end
    state.loss *= 0.5 * state.layer.weight / n

    # accumulate statistics
    state.loss_accum *= state.n_accum
    state.loss_accum += state.loss * n
    state.loss_accum /= state.n_accum + n

    state.n_accum += n
end

function backward(backend::CPUBackend, state::DecoderLossLayerState, 
        inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    mu = inputs[1].data
    sigma = inputs[2].data
    x = inputs[3].data

    if isa(diffs[1], CPUBlob)
        mu_diffs = diffs[1].data
        @devec mu_diffs[:] = -(x - mu) ./ (sigma.^2)
        diffs[1].data[:] *= state.layer.weight / n
    end

    if isa(diffs[2], CPUBlob)
        sigma_diffs = diffs[2].data
        @devec sigma_diffs[:] = 1 ./ sigma - (x - mu).^2 ./ (sigma.^3)
        diffs[2].data[:] *= state.layer.weight / n
    end

    if isa(diffs[3], CPUBlob)
        raise("last blob should be a data blob")
    end
end


