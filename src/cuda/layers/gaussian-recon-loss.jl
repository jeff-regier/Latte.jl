
function setup(backend::GPUBackend, layer::GaussianReconLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    aux_ones = make_blob(backend, data_type, size(inputs[2])...) # for summing
    fill!(aux_ones, 1.0)
    tmp_blob = make_blob(backend, data_type, size(inputs[2])...)
    scaled_sq_errs = make_blob(backend, data_type, size(inputs[2])...)
    tmp = @compat(Dict(:aux_ones => aux_ones,
                       :tmp => tmp_blob,
                       :scaled_sq_errs => scaled_sq_errs))
    state = GaussianReconLossLayerState(layer, zero(data_type), zero(data_type), 0, tmp)
    return state
end


function forward(backend::GPUBackend, state::GaussianReconLossLayerState, inputs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    np = length(inputs[1])

    μ = inputs[1]
    σ = inputs[2]
    x = inputs[3]
    aux_ones = state.tmp_blobs[:aux_ones]
    logσ² = state.tmp_blobs[:tmp]
    scaled_sq_errs = state.tmp_blobs[:scaled_sq_errs]

    copy!(logσ², σ)
    CuVec.log!(backend, logσ²)
    CuVec.mul_scal!(backend, logσ², convert(data_type, 2.))
    Σlogσ² = CuBLAS.dot(backend.cublas_ctx, data_type, np, logσ².ptr, 1, aux_ones.ptr, 1)

    copy!(scaled_sq_errs, μ)
    CuVec.sub!(backend, scaled_sq_errs, x)
    CuVec.div!(backend, scaled_sq_errs, σ)

    CuVec.pow!(backend, scaled_sq_errs, convert(data_type, 2.))

    Σscaled_sq_errs = CuBLAS.dot(backend.cublas_ctx, data_type, np,
                              scaled_sq_errs.ptr, 1, aux_ones.ptr, 1)

    state.loss = np * log(2pi) + Σlogσ² + Σscaled_sq_errs
    state.loss *= 0.5 * state.layer.weight / n

    # accumulate statistics
    state.loss_accum *= state.n_accum
    state.loss_accum += state.loss * n
    state.loss_accum /= state.n_accum + n

    state.n_accum += n
end


function backward(backend::GPUBackend, state::GaussianReconLossLayerState,
            inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    n = get_num(inputs[1])
    np = length(inputs[1])

    μ = inputs[1]
    σ = inputs[2]
    x = inputs[3]

    tmp = state.tmp_blobs[:tmp]
    scaled_sq_errs = state.tmp_blobs[:scaled_sq_errs]

    c = convert(data_type, state.layer.weight / n)

    if isa(diffs[1], CuTensorBlob)
        copy!(diffs[1], μ)
        CuVec.sub!(backend, diffs[1], x)
        CuVec.div!(backend, diffs[1], σ)
        CuVec.div!(backend, diffs[1], σ)

        CuVec.mul_scal!(backend, diffs[1], c)
    end

    if isa(diffs[2], CuTensorBlob)
        copy!(diffs[2], σ)
        CuVec.pow!(backend, diffs[2], convert(data_type, -1.0))

        copy!(tmp, scaled_sq_errs)
        CuVec.div!(backend, tmp, σ)

        CuVec.sub!(backend, diffs[2], tmp)
        CuVec.mul_scal!(backend, diffs[2], c)
    end

    if isa(diffs[3], CuTensorBlob)
        raise("last blob should be a data blob")
    end
end

