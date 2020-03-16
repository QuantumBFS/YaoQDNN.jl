using Flux
import Flux: params
using LinearAlgebra

using Zygote
import Zygote: @adjoint, AContext
using YaoBlocks.AD

Flux.@functor QNNL

function apply(ql::QNNL{T}, x) where T
	cir = chain(ql.encoder, ql.transform)
	n = nqubits(cir)
	dispatch!(cir, [x; ql.params])

	m = size(ql.Hami, 1)
	psi = zero_state(n)
	psi |> cir
	y = T.([real(expect((ql.Hami[i]), psi)) for i = 1:m])
	if size(ql.bias, 1) > 0
		y += ql.bias
	end
	return y
end
(ql::QNNL{T})(x) where {T} = apply(ql, x)

params(ql::QNNL{T}) where {T} = params(ql.params, ql.bias)

function Zygote._pullback(cx::AContext, ql::QNNL{T}, x) where {T}
	out = apply(ql, x)
	out, function pullback(h̄)
		∇ql, ∇x = backwardAD(ql, x, h̄)
		∇W = ∇ql[1]
		∇b = ∇ql[2]
		Zygote.accum_param(cx, ql.params, ∇W)
		Zygote.accum_param(cx, ql.bias, ∇b)
		return ((W = ∇W, b = ∇b), ∇x)
	end
end

# TODO need a faithful way
function forwardAD(ql::QNNL{T}, x, h̄) where {T}
	m = size(ql.Hami, 1)
	s1 = size(x, 1)
	s2 = size(ql.params, 1)
	s3 = size(ql.bias, 1)

	L_x = zeros(T, s1)
	L_w = zeros(T, s2)

	for i = 1:s1
		x[i] += T(π/2)
		y_pos = ql(x)
		x[i] -= T(π)
		y_neg = ql(x)
		x[i] += T(π/2)

		L_x[i] = dot((y_pos - y_neg), h̄) / 2
	end

	for i = 1:s2
		ql.params[i] += T(π/2)
		y_pos = ql(x)
		ql.params[i] -= T(π)
		y_neg = ql(x)
		ql.params[i] += T(π/2)

		L_w[i] = dot((y_pos - y_neg), h̄) / 2
	end

	if s3 > 0
		return [L_w, T.(h̄)], L_x
	else
		return [L_w, ql.bias], L_x
	end
end

function backwardAD(ql::QNNL{T}, x, h̄) where {T}
	encoder = ql.encoder
	transform = ql.transform
	ps = ql.params

	n = nqubits(encoder)
	psi = zero_state(n)

	m = length(ql.Hami)

	x̄ = T[]
	w̄ = T[]
	b̄ = length(ql.bias) > 0 ? T.(h̄) : T[]

	dispatch!(encoder, x)
	dispatch!(transform, ps)

	psi |> encoder
	psi	|> transform

	Hs = ql.Hami

	D = ArrayReg(sum([expect'(Hs[i], psi).state for i = 1:m] .* h̄))
	D.state *= 2

	apply_back!((psi, D), transform, w̄)
	apply_back!((psi, D), encoder, x̄)

	return [w̄, b̄], x̄
end
