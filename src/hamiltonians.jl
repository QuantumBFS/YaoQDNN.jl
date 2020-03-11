export genH

function genH(n::Integer, c::Char)
	if c == 'X'
		return [chain(n, put(n, i=>X)) for i = 1:n]
	elseif c == 'Y'
		return [chain(n, put(n, i=>Y)) for i = 1:n]
	elseif c == 'Z'
		return [chain(n, put(n, i=>Z)) for i = 1:n]
	end
end
