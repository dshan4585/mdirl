struct Categorical <: DiscretePd
   d::Int
   l::Vector{T}
end

const C = Categorical

C(d) = C(d, fill!(Vector{T}(undef, dims),0))

function choice(p)
   n = length(p)
   i = 1
   c = p[1]
   u = rand(T)
   while c < u && i < n
      @inbounds c += p[i += 1]
   end
   return i
end

sample(c::C) = choice(softmax(c.l))
sample!(x, c::C) = choice(softmax(c.l))
logp(x, c::C) = getindex(logsoftmax(c.l), x)
ent(c::C) = -sum(softmax(c.l) * logsoftmax(c.l))
kl(c₁::C, c₂::C) = sum(softmax(c₁.l) * 
   (logsoftmax(c₁.l) - logsoftmax(c₂.l)))
tent(c::C; q::T) = (1/(q-1)) * (1 - sum(softmax(c.l).^q))
function tkl(c₁::C, c₂::C; q::T)
   p₁ = softmax(c₁.l)
   p₂ = softmax(c₂.l)
   Ωp₁ = - (1 / q) * (1 - sum(p₁.^q))
   Ωp₂ = - (1 / q) * (1 - sum(p₂.^q))
   ∇Ωp₁ = - (1 / q) * (1 - q * sum(p₁.^(q-1)))
   Ωp₁ - Ωp₂ - dot(∇Ωp₁, p₁ - p₂) 
end