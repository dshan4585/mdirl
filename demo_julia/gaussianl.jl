const LT = LowerTriangular

struct Gaussian <: ContinuousPd
   d::Int
   μ::Vector{F32}
   L::LT{F32,Array{F32,2}}
end
struct DiagGaussian <: ContinuousPd
   d::Int
   μ::Vector{F32}
   lnσ::Vector{F32}
end
Gaussians = Union{Gaussian,DiagGaussian}

const G = Gaussian
const DG = DiagGaussian
const Gs = Gaussians
const □ = .5f0log(2f0π)
const □e = .5f0log(2f0π*ℯ)
const log4 = log(4f0)

function cholupdate!(L, x)
   n = length(x)
   @inbounds for i ∈ 1:n
      li = L[i,i]
      xi = x[i]

      r = √(li^2 + xi^2)
      c = r / li
      s = xi / li
      L[i,i] = r
      if k < n
         below = (i+1):n
         L[below,i] .= (.+s.*x[below])./c
         x[below] .= c .* x[below] .- s .* L[below,i]
      end
   end
end

G(n) = Gaussian(n,zeros(F32,n),LT{F32}(Matrix{F32}(I,(n,n))))
DG(n) = DiagGaussian(n,zeros(F32,n),zeros(F32,n))

logp(x,g::G) = -g.d*□-logdet(g.L)-.5f0norm(inv(g.L)*(x-g.μ))^2
logp(x,g::DG) = -g.d*□-sum(@. g.lnσ+.5f0(((x-g.μ)/exp(g.lnσ))^2))
sample(g::G) = (x=randn(F32,size(g.μ)); x.=g.μ.+g.L*x)
sample(g::DG) = (x=randn(F32,size(g.μ)); x.=g.μ.+exp.(g.lnσ).*x)
sample!(x,g::G) = (randn!(x);x.=g.μ.+g.L*x)
sample!(x,g::DG) = (randn!(x);x.=g.μ.+exp.(g.lnσ).*x)
ent(g::G) = g.d*□e+logdet(g.L)
ent(g::DG) = g.d*□e+sum(g.lnσ)

function kl(g₁::G, g₂::G)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   L₁ = g₁.L
   L₂ = g₂.L

   Σ₁ = L₁*L₁'
   Σ₂ = L₂*L₂'
   L₂⁻¹ = inv(L₂)
   Σ₂⁻¹ = L₂⁻¹' * L₂⁻¹

   .5f0(- d + 2(logdet(L₂)-logdet(L₁)) + tr(Σ₂⁻¹*Σ₁) + norm(L₂⁻¹*(μ₂-μ₁))^2)
end

function kl(g₁::DG, g₂::DG)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = g₁.lnσ
   lnσ₂ = g₂.lnσ

   σ₁ = exp.(lnσ₁)
   σ₂ = exp.(lnσ₂)

   .5f0sum((@. -1 + 2(lnσ₂ - lnσ₁) + (σ₁/σ₂)^2 + ((μ₂-μ₁)/σ₂)^2))
end

tent(g::G; q::F32) = (1-exp((1-q)*(g.d*□ + logdet(g.L) - g.d*(log(q)/(2-2q))))) / (q-1)
tent(g::DG; q::F32) = (1-exp((1-q)*(g.d*□ + sum(g.lnσ) - g.d*(log(q)/(2-2q))))) / (q-1)

function Ψ(g₁::G, g₂::G, α, β)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   L₁ = g₁.L
   L₂ = g₂.L

   L₁⁻¹ = inv(L₁)
   Σ₁⁻¹ = L₁⁻¹' * L₁⁻¹
   L₂⁻¹ = inv(L₂)
   Σ₂⁻¹ = L₂⁻¹' * L₂⁻¹

   a = α .* (Σ₁⁻¹ * μ₁) .+ β .* (Σ₂⁻¹ * μ₂)
   B = α .* (Σ₁⁻¹) .+ β .* (Σ₂⁻¹)
   B⁻¹ = inv(B)

   Fθ₁ = d * □ + .5f0norm(L₁⁻¹*μ₁)^2 + logdet(L₁)
   Fθ₂ = d * □ + .5f0norm(L₂⁻¹*μ₂)^2 + logdet(L₂)
   Fθ  = d * □ + .5f0(a'*B⁻¹*a) - .5f0logdet(B)
   exp(Fθ - α*Fθ₁ - β*Fθ₂)
end

function Ψ(g₁::DG, g₂::DG, α, β)
   d = g₁.d
   μ₁ = g₁.μ
   μ₂ = g₂.μ
   lnσ₁ = g₁.lnσ
   lnσ₂ = g₂.lnσ

   σ₁ = exp.(lnσ₁)
   σ₂ = exp.(lnσ₂)

   σ₁⁻² = @. exp(-2lnσ₁)
   σ₂⁻² = @. exp(-2lnσ₂)

   μσ⁻² = @. α * μ₁ * σ₁⁻² + β * μ₂ * σ₂⁻²
   σ⁻²  = @. α * σ₁⁻² + β * σ₂⁻² 

   Fθ₁ = d*□ + sum(@. .5f0(μ₁/σ₁)^2+lnσ₁)
   Fθ₂ = d*□ + sum(@. .5f0(μ₂/σ₂)^2+lnσ₂)
   Fθ  = d*□ + sum(@. .5f0(μσ⁻²^2/σ⁻²)-.5f0log(σ⁻²))
   exp(Fθ - α*Fθ₁ - β*Fθ₂)
end

tkl(g₁::Gs, g₂::Gs; q::F32) = 
   (q/(q-1))*(1-Ψ(g₁,g₂,1f0,q-1))-tent(g₁;q=q)-(q-1)*tent(g₂;q=q) 



