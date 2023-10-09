const UL = UnitLowerTriangular

struct Gaussian <: ContinuousPd
   d::Int
   μ::Vector{T}
   lnσ::Vector{T}
   ult::Array{T,2}
end
struct DiagGaussian <: ContinuousPd
   d::Int
   μ::Vector{T}
   lnσ::Vector{T}
end
Gaussians = Union{Gaussian,DiagGaussian}

const G = Gaussian
const DG = DiagGaussian
const Gs = Gaussians

const □ = convert(T, .5log(2π))
const □e = convert(T, .5log(2π*ℯ))

G(n) = G(n, zeros(T,n), zeros(T,n), Float32[atanh(1/3)*30 0;
                                            0 atanh(1/3)*30])
DG(n) = DG(n,zeros(T,n),zeros(T,n))

ultf(x) = tanh(x/30f0) * 3f0
lnσf(x) = log(tanh(x/9.95f0)*.995f0+1.005f0)

sample(g::G) = (g.μ .+ ultf.(g.ult) * (exp.(lnσf.(g.lnσ)) .* randn(T, size(g.μ))))
sample(g::DG) = (g.μ .+ exp.(lnσf.(g.lnσ)) .* randn(T, size(g.μ)))
sample!(x,g::G)  = (randn!(x);
   x .= g.μ .+ ultf.(g.ult) * (exp.(lnσf.(g.lnσ)) .* x))
sample!(x,g::DG) = (randn!(x);
   x .= g.μ .+ exp.(lnσf.(g.lnσ)) .* x)

logp(x, g::G) = -g.d*□ - sum(lnσf.(g.lnσ) .+
   .5f0 .* (inv(ultf.(g.ult)) * (x .- g.μ)).^2 .* exp.(-2 .* lnσf.(g.lnσ)))
logp(x, g::DG) = -g.d*□ - sum(@. lnσf(g.lnσ) +
   .5f0((x - g.μ)^2 * exp(-2lnσf(g.lnσ))))
ent(g::Gs) = g.d * □e + sum(lnσf.(g.lnσ))

function kl(g₁::G, g₂::G)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = lnσf.(g₁.lnσ)
   lnσ₂ = lnσf.(g₂.lnσ)
   ult₁ = ultf.(g₁.ult)
   ult₂ = ultf.(g₂.ult)
   ult₂⁻¹ = inv(ult₂)
   L₁ = ult₁ .* exp.(lnσ₁)'
   L₂⁻¹ = exp.(-lnσ₂) .* ult₂⁻¹
   Σ₁ = L₁ * L₁'
   Σ₂⁻¹ = L₂⁻¹' * L₂⁻¹

   .5f0(- d + tr(Σ₂⁻¹*Σ₁) + sum(2 .* (lnσ₂ .- lnσ₁) .+
      (ult₂⁻¹ * (μ₂ .- μ₁)).^2 .* exp.(-2 .* lnσ₂)))
end

function kl(g₁::DG, g₂::DG)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = lnσf.(g₁.lnσ)
   lnσ₂ = lnσf.(g₂.lnσ)

   .5f0(- d + sum(@. 2(lnσ₂ - lnσ₁) +
      exp(2(lnσ₁ - lnσ₂)) + (μ₂ - μ₁)^2 * exp(-2lnσ₂)))
end

tent(g::Gs; q::T) =  (h = q-1;
   h\(1-exp(-g.d*.5f0log(q)-h*(g.d*□+sum(lnσf.(g.lnσ))))))

function Ψ(g₁::G, g₂::G, α::T, β::T)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = lnσf.(g₁.lnσ)
   lnσ₂ = lnσf.(g₂.lnσ)
   ult₁⁻¹ = inv(ultf.(g₁.ult))
   ult₂⁻¹ = inv(ultf.(g₂.ult))
   L₁⁻¹ = exp.(-lnσ₁) .* ult₁⁻¹
   L₂⁻¹ = exp.(-lnσ₂) .* ult₂⁻¹

   Σ₁⁻¹ = L₁⁻¹' * L₁⁻¹
   Σ₂⁻¹ = L₂⁻¹' * L₂⁻¹

   a = α .* (Σ₁⁻¹ * μ₁) .+ β .* (Σ₂⁻¹ * μ₂)
   B = α .* Σ₁⁻¹ .+ β .* Σ₂⁻¹

   exp(d * (1 - α - β) * □ +
      .5f0(a'*inv(B)*a) - .5f0logdet(B) -
      sum(α .* (.5f0 .* (ult₁⁻¹ * μ₁).^2 .* exp.(-2 .* lnσ₁) .+ lnσ₁) +
          β .* (.5f0 .* (ult₂⁻¹ * μ₂).^2 .* exp.(-2 .* lnσ₂) .+ lnσ₂)))
end

function Ψ(g₁::DG, g₂::DG, α::T, β::T)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = lnσf.(g₁.lnσ)
   lnσ₂ = lnσf.(g₂.lnσ)
   σ₁⁻² = exp.(-2 .* lnσ₁)
   σ₂⁻² = exp.(-2 .* lnσ₂)
   exp(d * (1 - α - β) * □ +
      sum(@. .5f0((α*μ₁*σ₁⁻²+β*μ₂*σ₂⁻²)^2 / (α*σ₁⁻²+β*σ₂⁻²)) -
      .5f0log(α * σ₁⁻² + β * σ₂⁻²) -
      α * (.5f0(μ₁^2 * σ₁⁻²) + lnσ₁) -
      β * (.5f0(μ₂^2 * σ₂⁻²) + lnσ₂)))
end

tkl(g₁::Gs, g₂::Gs; q::T) = (h = q - 1;
   (q/h)*(1 - Ψ(g₁, g₂, 1f0, h)) - tent(g₁; q=q) - h*tent(g₂; q=q))



