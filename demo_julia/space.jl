module Spaces
const F32 = Float32
import Base: length, size, rand, ==, ∈, show
import Random: rand!
export Space, Discrete, Continuous, Finite
abstract type Space end=
abstract type Discrete <: Space end
abstract type Continuous <: Space end
struct Finite<:Discrete num::Int end
const F = Finite

length(f::F) = f.num
size(f::F) = (f.num,)
rand(f::F) = rand(1:f.num)
rand!(f::F, x) = rand(f)
(==)(f₁::F, f₂::F) = (f₁.num == f₂.num)
(∈)(x::Int, f::F) = x ∈ 1:f.num
logpᵤ(f::F) = log(1f0 / f.num)

struct Bounded <: Space
   dims::Tuple{Vararg{Int}}
   low::Array{F32}
   high::Array{F32}
   center::Array{F32}
   scale::Array{F32}
   halfscale::Array{F32}
   logp_uniform::F32
end

const B = Bounded
function B(low, high)
   scale = high - low
   halfscale = .5f0scale
   center = low + halfscale
   logp_uniform = -sum(log.(scale))
   Bounded(size(low), low, high, center, scale, halfscale, logp_uniform)
end

B(dims, bounds::Tuple{F32,F32}) =
   B(fill!(Array{F32}(undef, dims), bounds[1]),
     fill!(Array{F32}(undef, dims), bounds[2]))
length(b::B) = prod(b.dims)
size(b::B) = b.dims
rand(b::B) = (@. rand(F32, b.dims) * b.scale + b.low)
rand!(x, b::B) = (rand!(x); @. x = x * b.scale + b.low)

(==)(b₁::B, b₂::B) = b₁.dims == b₂.dims &&
   reduce(&, b₁.low  .== b₂.low;  init=true) &&
   reduce(&, b₁.high .== b₂.high; init=true)

(∈)(x, b::B) = size(x) == c.dims &&
   reduce(&, b.low .≤ x .≤ b.high; init=true)

show(io::IO, b::B) = length(c.dims) == 1 ?
   print(io, "Continuous($(b.dims[1]))") : print(io, "Continuous$(b.dims)")

logpᵤ(b::B) = b.logp_uniform

end