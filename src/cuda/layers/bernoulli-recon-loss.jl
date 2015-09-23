
function setup(backend::GPUBackend, layer::BernoulliReconLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    aux_ones = make_blob(backend, data_type, size(inputs[1])...) # for summing
    fill!(aux_ones, 1.0)
    tmp1_blob = make_blob(backend, data_type, size(inputs[1])...)
    tmp2_blob = make_blob(backend, data_type, size(inputs[1])...)
    tmp = @compat(Dict(:aux_ones => aux_ones,
                       :tmp1 => tmp1_blob,
                       :tmp2 => tmp2_blob))
    state = BernoulliReconLossLayerState(layer, zero(data_type), zero(data_type), 0, tmp)
    return state
end


function forward(backend::GPUBackend, state::BernoulliReconLossLayerState, inputs::Vector{Blob})
    data_type = eltype(inputs[1])
    N = get_num(inputs[1])
    ND = length(inputs[1])

    p = inputs[1]
    x = inputs[2]

    aux_ones = state.tmp_blobs[:aux_ones]
    tmp1 = state.tmp_blobs[:tmp1]
    tmp2 = state.tmp_blobs[:tmp2]

    copy!(tmp1, aux_ones)
    CuVec.sub!(backend, tmp1, p)
    CuVec.log!(backend, tmp1)
    copy!(tmp2, aux_ones)
    CuVec.sub!(backend, tmp2, x)
    CuVec.mul!(backend, tmp1, tmp2)

    copy!(tmp2, p)
    CuVec.log!(backend, tmp2)
    CuVec.mul!(backend, tmp2, x)

    CuVec.add!(backend, tmp1, tmp2)
    ll = CuBLAS.dot(backend.cublas_ctx, data_type, ND, tmp1.ptr, 1, aux_ones.ptr, 1)

    state.loss = -ll * state.layer.weight / N

    # accumulate statistics
    state.loss_accum *= state.n_accum
    state.loss_accum += state.loss * N
    state.loss_accum /= state.n_accum + N

    state.n_accum += N
end


function backward(backend::GPUBackend, state::BernoulliReconLossLayerState,
            inputs::Vector{Blob}, diffs::Vector{Blob})
    data_type = eltype(inputs[1])
    N = get_num(inputs[1])
    ND = length(inputs[1])

    p = inputs[1]
    x = inputs[2]

    aux_ones = state.tmp_blobs[:aux_ones]
    tmp1 = state.tmp_blobs[:tmp1]
    tmp2 = state.tmp_blobs[:tmp2]

    c = convert(data_type, state.layer.weight / N)

    if isa(diffs[1], CuTensorBlob)
        copy!(diffs[1], x)
        CuVec.sub!(backend, diffs[1], aux_ones)
        copy!(tmp1, p)
        CuVec.sub!(backend, tmp1, aux_ones)
        CuVec.div!(backend, diffs[1], tmp1)

        copy!(tmp1, x)
        CuVec.div!(backend, tmp1, p)

        CuVec.sub!(backend, diffs[1], tmp1)
        CuVec.mul_scal!(backend, diffs[1], c)
    end

    if isa(diffs[2], CuTensorBlob)
        raise("last blob should be a data blob")
    end
end

