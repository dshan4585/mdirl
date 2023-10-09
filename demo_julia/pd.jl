module Pds

const T = Float32
import Base: length, size, rand, ==, ∈, show
import Random: randn, randn!

using LinearAlgebra
using Flux: softmax, logsoftmax, tanh

export Pd, DiscretePd, ContinuousPd
export Categorical, Gaussian, DiagGaussian
export sample, sample!, logp, ent, kl, tent, tkl

abstract type Pd end
abstract type DiscretePd <: Pd end
abstract type ContinuousPd <: Pd end

include("categorical.jl")
include("gaussian_safe.jl")

length(d::Pd) = d.d
size(d::Pd) = (d.d,)
(==)(d₁::Pd,d₂::Pd) = (typeof(d₁)≡typeof(d₂))&&(d₁.d==d₂.d)
(∈)(x, d::Pd) = length(x) == d.d
show(io::IO, pd::Categorical) = print(io, "Categorical($(pd.d))")
show(io::IO, pd::Gaussian) = print(io, "Gaussian($(pd.d))")
show(io::IO, pd::DiagGaussian) = print(io, "DiagGaussian($(pd.d))")

end