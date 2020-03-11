using LinearAlgebra
using Yao

include("circuit.jl")
include("hamiltonians.jl")

export QNNL, forward, back_propagation
export Layer

abstract type Layer{T<:Real} end

struct QNNL{T} <: Layer{T}
	encoder::YaoBlocks.ChainBlock
	transform::YaoBlocks.ChainBlock
	params::Array{T}
	Hami::Array{<:YaoBlocks.AbstractBlock, 1}
	bias::Array{T,1}
end

function QNNL{T}(he::HardwareEfficientAnsatz, nbit::Integer, in_dim::Integer,
	 			w::Vector{T}, Hami::Array{<:YaoBlocks.AbstractBlock, 1},
				b::Vector{T}) where {T}
	cir_e = YaoQDNN.encoder_circuit(he, nbit, in_dim)
	cir_t = YaoQDNN.transform_circuit(he, nbit, length(w))
	QNNL{T}(cir_e, cir_t, w, Hami, b)
end

function QNNL{T}(he::HardwareEfficientAnsatz, nbit::Integer, in_dim::Integer,
				npara::Integer, Hami::Array{<:YaoBlocks.AbstractBlock, 1};
				no_bias = false) where {T}
	w = (rand(T, npara) .- 0.5) * 2π
	if no_bias
		b = T[]
	else
		b = rand(T, size(Hami)) .- 0.5
	end
	QNNL{T}(he, nbit, in_dim, w, Hami, b)
end

function QNNL{T}(he::HardwareEfficientAnsatz, nbit::Integer, in_dim::Integer,
				w::Vector{T}, Hs::String, b::Vector{T}) where {T}
	Hami = YaoBlocks.ChainBlock[]
	for c in Hs
		Hami = [Hami; genH(nbit, c)]
	end
	QNNL{T}(he, nbit, in_dim, w, Hami, b)
end

function QNNL{T}(he::HardwareEfficientAnsatz, nbit::Integer, in_dim::Integer,
				npara::Integer, Hs::String; no_bias = false) where {T}
	Hami = YaoBlocks.ChainBlock[]
	for c in Hs
		Hami = [Hami; genH(nbit, c)]
	end
	QNNL{T}(he, nbit, in_dim, npara, Hami; no_bias = no_bias)
end
