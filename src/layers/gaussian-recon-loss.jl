# Expected reconstruction error for Gaussian output with diagonal covariance


@defstruct GaussianReconLossLayer Layer (
    name :: String = "gaussian-recon-loss",
    (weight :: FloatingPoint = 1.0, weight >= 0),
    (bottoms :: Vector{Symbol} = Symbol[:mu, :sigma, :x], length(bottoms) == 3),
)

@characterize_layer(GaussianReconLossLayer,
    has_loss  => true,
    can_do_bp => true,
    is_sink   => true,
    has_stats => true,
)

type GaussianReconLossLayerState{T, B<:Blob} <: LayerState
    layer      :: GaussianReconLossLayer
    loss       :: T
    loss_accum :: T
    n_accum    :: Int

    tmp_blobs :: Dict{Symbol, B}
end

function setup(backend::CPUBackend, layer::GaussianReconLossLayer, 
            inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    return GaussianReconLossLayerState(layer, zero(data_type), zero(data_type),
                                                0, Dict{Symbol, CPUBlob}())
end

function shutdown(backend::Backend, state::GaussianReconLossLayerState)
    for blob in values(state.tmp_blobs)
        destroy(blob)
    end
end

function reset_statistics(state::GaussianReconLossLayerState)
    state.n_accum = 0
    state.loss_accum = zero(typeof(state.loss_accum))
end

function dump_statistics(storage, state::GaussianReconLossLayerState, show::Bool)
    update_statistics(storage, "$(state.layer.name)-recon-loss", state.loss_accum)

    if show
      loss = @sprintf("%.4f", state.loss_accum)
      @info("  gaussian-recon-loss (avg over $(state.n_accum)) = $loss")
    end
end

function forward(backend::CPUBackend, state::GaussianReconLossLayerState,
            inputs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    np = length(inputs[1])
    mu = inputs[1].data
    sigma = inputs[2].data
    x = inputs[3].data

    state.loss = 0
    for i in 1:np
        state.loss += log(2pi) + 2log(sigma[i]) + (x[i] - mu[i])^2 / sigma[i]^2
    end
    state.loss *= 0.5 * state.layer.weight / n

    # accumulate statistics
    state.loss_accum *= state.n_accum
    state.loss_accum += state.loss * n
    state.loss_accum /= state.n_accum + n

    state.n_accum += n
end

function backward(backend::CPUBackend, state::GaussianReconLossLayerState, 
            inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    np = length(inputs[1])
    mu = inputs[1].data
    sigma = inputs[2].data
    x = inputs[3].data

    if isa(diffs[1], CPUBlob)
        for i in 1:np
            diffs[1].data[i] = -(x[i] - mu[i]) / sigma[i]^2
        end
        diffs[1].data[:] *= state.layer.weight / n
    end

    if isa(diffs[2], CPUBlob)
        for i in 1:np
            diffs[2].data[i] = 1 ./ sigma[i] - (x[i] - mu[i])^2 / sigma[i]^3
        end
        diffs[2].data[:] *= state.layer.weight / n
    end

    if isa(diffs[3], CPUBlob)
        raise("last blob should be a data blob")
    end
end

