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

G(n) = G(n, zeros(T,n), zeros(T,n), UL{T}(Matrix{T}(I,(n,n))))
DG(n) = DG(n,zeros(T,n),zeros(T,n))

sample(g::G) = (x = randn(T, size(g.μ)); 
   x .= g.μ .+ g.ult * (exp.(g.lnσ) .* x))
sample(g::DG) = (x = randn(T, size(g.μ)); 
   x .= g.μ .+ exp.(g.lnσ) .* x)
sample!(x,g::G)  = (randn!(x);
   x .= g.μ .+ g.ult * (exp.(g.lnσ) .* x))
sample!(x,g::DG) = (randn!(x);
   x .= g.μ .+ exp.(g.lnσ) .* x)

logp(x, g::G) = -g.d*□ - sum(g.lnσ .+
   .5f0 .* (inv(g.ult) * (x .- g.μ)).^2 .* exp.(-2 .* g.lnσ))
logp(x, g::DG) = -g.d*□ - sum(@. g.lnσ +
   .5f0((x - g.μ)^2 * exp(-2g.lnσ)))
ent(g::Gs) = g.d * □e + sum(g.lnσ)

function kl(g₁::G, g₂::G)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = g₁.lnσ
   lnσ₂ = g₂.lnσ
   ult₁ = g₁.ult
   ult₂ = g₂.ult
   ult₂⁻¹ = inv(g₂.ult)
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
   lnσ₁ = g₁.lnσ
   lnσ₂ = g₂.lnσ

   .5f0(- d + sum(@. 2(lnσ₂ - lnσ₁) +
      exp(2(lnσ₁ - lnσ₂)) + (μ₂ - μ₁)^2 * exp(-2lnσ₂)))
end

tent(g::Gs; q::T) =  (h = q-1;
   h\(1-exp(-g.d*.5f0log(q)-h*(g.d*□+sum(g.lnσ)))))

function Ψ(g₁::G, g₂::G, α::T, β::T)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = g₁.lnσ
   lnσ₂ = g₂.lnσ
   ult₁⁻¹ = inv(g₁.ult)
   ult₂⁻¹ = inv(g₂.ult)
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
   lnσ₁ = g₁.lnσ
   lnσ₂ = g₂.lnσ
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



