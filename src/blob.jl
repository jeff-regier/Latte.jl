export Blob
export CPUBlob, NullBlob

import Base: eltype, size, length, ndims, copy!, fill!, show
export       eltype, size, length, ndims, copy!, fill!, erase!, show
export get_num, get_chann, get_height, get_width, get_fea_size, get_whcn, to_array
export make_blob, make_zero_blob, reshape_blob

############################################################
# A blob is an abstract concept that is suppose
# to hold a 4-D tensor of data. The data could
# either live in CPU memory or GPU memory or
# whatever the backend is used to store the data.
############################################################
abstract Blob{T, N}

############################################################
# The following should be implemented for a
# concrete Blob type. Note the following
# procedures are only provided for convenience
# and mainly for components that do not need
# to know the underlying backend (e.g. Filler).
############################################################
function eltype{T}(blob :: Blob{T})
  T
end

function ndims{T,N}(blob :: Blob{T,N})
  N
end
function size(blob :: Blob)
  error("Not implemented (should return the size of data)")
end
function destroy(blob :: Blob)
  error("Not implemented (should destroy the blob)")
end
function size{T,N}(blob :: Blob{T,N}, dim :: Int)
  if dim < 0
    dim = N+1 + dim
  end
  if dim > N
    return 1
  end

  size(blob)[dim]
end
function length(blob :: Blob)
  return prod(size(blob))
end

function get_num(blob :: Blob)
  size(blob, -1)
end
function get_chann(blob :: Blob)
  size(blob, 3)
end
function get_height(blob :: Blob)
  size(blob, 2)
end
function get_width(blob :: Blob)
  size(blob, 1)
end
function get_fea_size(blob :: Blob)
  prod(size(blob)[1:end-1])
end

# Get pseudo 4D dimension
# Note the behavior when the tensor dimension is less than 4.
# For example, for 2D tensor, the two dimensions are considered
# as channels and num. With this convention, InnerProductLayer
# could produce 2D tensors and those tensors could be naturally
# processed by existing loss layers without much modifications.
function get_whcn{T}(blob :: Blob{T,1})
  (1,1,1,size(blob,1))
end
function get_whcn{T}(blob :: Blob{T,2})
  c,n = size(blob)
  (1,1,c,n)
end
function get_whcn{T}(blob :: Blob{T,3})
  h,c,n = size(blob)
  (1,h,c,n)
end
function get_whcn{T}(blob :: Blob{T,4})
  size(blob)
end
function get_whcn{T,N}(blob :: Blob{T,N})
  dims = size(blob)
  (dims[1],dims[2],prod(dims[3:end-1]),dims[end])
end

function show(io::IO, blob :: Blob)
  shape = join(map(x -> "$x", size(blob)), " x ")
  print(io, "Blob($shape)")
end

function to_array(blob::Blob)
  array = Array(eltype(blob), size(blob))
  copy!(array, blob)
  array
end

function copy!(dst :: Array, src :: Blob)
  error("Not implemented (should copy content of src to dst)")
end
function copy!(dst :: Blob, src :: Array)
  error("Not implemented (should copy content of src to dst)")
end
function fill!(dst :: Blob, val)
  error("Not implemented (should fill dst with val)")
end
function erase!(dst :: Blob)
  fill!(dst, 0)
end

############################################################
# A Dummy Blob type holding nothing
############################################################
type NullBlob <: Blob
end
function fill!(dst :: NullBlob, val)
  # do nothing
end
function show(io::IO, blob::NullBlob)
  print(io, "Blob()")
end

function destroy(blob::NullBlob)
  # do nothing
end
function make_blob(backend::Backend, data_type::Type, dims::Int...)
  make_blob(backend, data_type, dims)
end
function make_blob(backend::Backend, data::Array)
  blob = make_blob(backend, eltype(data), size(data))
  copy!(blob, data)
  return blob
end
function make_zero_blob{N}(backend::Backend, data_type::Type, dims::NTuple{N,Int})
  blob = make_blob(backend, data_type, dims)
  erase!(blob)
  return blob
end
function make_zero_blob(backend::Backend, data_type::Type, dims::Int...)
  make_zero_blob(backend, data_type, dims)
end

function reshape_blob(backend::Backend, blob::Blob, dims::Int...)
  reshape_blob(backend, blob, dims)
end

############################################################
# A Blob for CPU Computation
############################################################
immutable CPUBlob{T <: FloatingPoint, N} <: Blob{T, N}
  data :: AbstractArray{T, N}
end
CPUBlob{N}(t :: Type, dims::NTuple{N,Int}) = CPUBlob(Array(t, dims))

function make_blob{N}(backend::CPUBackend, data_type::Type, dims::NTuple{N,Int})
  return CPUBlob(data_type, dims)
end

function reshape_blob{T,N1,N2}(backend::CPUBackend, blob::CPUBlob{T,N1}, dims::NTuple{N2,Int})
  @assert prod(dims) == length(blob)
  return CPUBlob{T,N2}(reshape(blob.data, dims))
end
function destroy(blob::CPUBlob)
  # do nothing... or is there anything that I could do?
end

size(blob::CPUBlob) = size(blob.data)

function copy!{T}(dst :: Array{T}, src :: CPUBlob{T})
  @assert length(dst) == length(src)
  dst[:] = src.data[:]
end
function copy!{T}(dst :: CPUBlob{T}, src :: Array{T})
  @assert length(dst) == length(src)
  dst.data[:] = src[:]
end
function copy!{T}(dst :: CPUBlob{T}, src :: CPUBlob{T})
  dst.data[:] = src.data[:]
end
function fill!{T}(dst :: CPUBlob{T}, src)
  fill!(dst.data, src)
end
