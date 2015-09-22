# Expected reconstruction error for Bernoulli output with diagonal covariance


@defstruct BernoulliReconLossLayer Layer (
    name :: String = "bernoulli-recon-loss",
    (weight :: FloatingPoint = 1.0, weight >= 0),
    (bottoms :: Vector{Symbol} = Symbol[:p, :x], length(bottoms) == 2),
)

@characterize_layer(BernoulliReconLossLayer,
    has_loss  => true,
    can_do_bp => true,
    is_sink   => true,
    has_stats => true,
)

type BernoulliReconLossLayerState{T, B<:Blob} <: LayerState
    layer      :: BernoulliReconLossLayer
    loss       :: T
    loss_accum :: T
    n_accum    :: Int

    tmp_blobs :: Dict{Symbol, B}
end

function setup(backend::CPUBackend, layer::BernoulliReconLossLayer, 
            inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    return BernoulliReconLossLayerState(layer, zero(data_type), zero(data_type),
                                                0, Dict{Symbol, CPUBlob}())
end

function shutdown(backend::Backend, state::BernoulliReconLossLayerState)
    for blob in values(state.tmp_blobs)
        destroy(blob)
    end
end

function reset_statistics(state::BernoulliReconLossLayerState)
    state.n_accum = 0
    state.loss_accum = zero(typeof(state.loss_accum))
end

function dump_statistics(storage, state::BernoulliReconLossLayerState, show::Bool)
    update_statistics(storage, "$(state.layer.name)-recon-loss", state.loss_accum)

    if show
      loss = @sprintf("%.4f", state.loss_accum)
      @info("  bernoulli-recon-loss (avg over $(state.n_accum)) = $loss")
    end
end

function forward(backend::CPUBackend, state::BernoulliReconLossLayerState,
            inputs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    np = length(inputs[1])
    p = inputs[1].data
    x = inputs[2].data

    state.loss = 0.
    for i in 1:np
        state.loss -= log(1 - p[i] - x[i] + 2p[i]x[i])
    end
    state.loss *= state.layer.weight / n

    # accumulate statistics
    state.loss_accum *= state.n_accum
    state.loss_accum += state.loss * n
    state.loss_accum /= state.n_accum + n

    state.n_accum += n
end

function backward(backend::CPUBackend, state::BernoulliReconLossLayerState, 
            inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    np = length(inputs[1])
    p = inputs[1].data
    x = inputs[2].data

    if isa(diffs[1], CPUBlob)
        for i in 1:np
            diffs[1].data[i] = (1. - 2x[i]) / (1 - p[i] - x[i] + 2p[i]x[i])
        end
        diffs[1].data[:] *= state.layer.weight / n
    end

    if isa(diffs[2], CPUBlob)
        raise("last blob should be a data blob")
    end
end

