export encoder_circuit, transform_circuit
export hardware_efficient

# generate a layer of rotation X or Z
layerX(nbit::Integer, pos::Vector{T}) where {T<:Integer} = chain(nbit, put(i=>Rx(0)) for i in pos)
layerZ(nbit::Integer, pos::Vector{T}) where {T<:Integer} = chain(nbit, put(i=>Rz(0)) for i in pos)
layerX(nbit::Integer, m::Integer) = layerX(nbit, Vector(1:min(m, nbit)))
layerZ(nbit::Integer, m::Integer) = layerZ(nbit, Vector(1:min(m, nbit)))
layerX(nbit::Integer) = layerX(nbit, nbit)
layerZ(nbit::Integer) = layerZ(nbit, nbit)

entangler(pairs::Array{Pair{T,T},1} where T) = chain(control(ctrl, target=>X) for (ctrl, target) in pairs)

struct HardwareEfficientAnsatz end
const hardware_efficient = HardwareEfficientAnsatz()

# encoder ansatz circuits
function encoder_circuit(::HardwareEfficientAnsatz, nbit::Integer, ninput::Integer,
		pairs::Array{Pair{T,T},1} where T<:Integer)
	circuit = chain(nbit)
	l = ninput ÷ nbit
	m = ninput % nbit
	for i in 1:l
		if i%3 == 0
			push!(circuit, entangler(pairs))
		end
		if i%3 == 1
			push!(circuit, layerX(nbit))
		else
			push!(circuit, layerZ(nbit))
		end
	end
	if m > 0
		if (l+1)%3 == 0
			push!(circuit, entangler(pairs))
		end
		push!(circuit, ((l+1)%3==1 ? layerX(nbit, m) : layerZ(nbit, m)))
	end
	return circuit
end

function encoder_circuit(he::HardwareEfficientAnsatz, nbit::Integer, ninput::Integer)
	pairs = [i => i+1 for i = 1:(nbit-1)]
	return encoder_circuit(he, nbit, ninput, pairs)
end

# transform ansatz circuits
function transform_circuit(::HardwareEfficientAnsatz, nbit::Integer, npara::Integer,
		pairs::Array{Pair{T,T},1} where T<:Integer)
	circuit = chain(nbit)
	l = npara ÷ nbit
	m = npara % nbit
	for i in 1:l
		if i%3 == 1
			push!(circuit, entangler(pairs))
		end
		if i%3 == 1
			push!(circuit, layerX(nbit))
		else
			push!(circuit, layerZ(nbit))
		end
	end
	if m > 0
		if (l+1)%3 == 1
			push!(circuit, entangler(pairs))
		end
		push!(circuit, ((l+1)%3==1 ? layerX(nbit, m) : layerZ(nbit, m)))
	end
	return circuit
end

function transform_circuit(he::HardwareEfficientAnsatz, nbit::Integer, npara::Integer)
	pairs = [i => i+1 for i = 1:(nbit-1)]
	return transform_circuit(he, nbit, npara, pairs)
end
