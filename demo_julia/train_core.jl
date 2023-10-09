using LinearAlgebra
using Flux
using BSON: @save

include("pd.jl"); using .Pds
include("irl.jl")

qs = [1.0f0, 1.1f0, 1.5f0, 2.0f0]
α0s = [0.1f0, 0.2f0, 0.5f0, 1.0f0, 2.0f0, 5.0f0, 10.0f0, 20.0f0, 50.0f0, 100.0f0,
       0.1f0, 1.0f0, 10.0f0, 0.1f0, 1.0f0, 10f0, 2f0, 2f0]
αTs = [0.1f0, 0.2f0, 0.5f0, 1.0f0, 2.0f0, 5.0f0, 10.0f0, 20.0f0, 50.0f0, 100.0f0,
       1.0f0, 10.0f0, 100.0f0, 10.0f0, 100.0f0, 1000f0, 10f0, 100f0]

gₐ = Gaussian(2)
gₑ = Gaussian(2)

gₐ.μ .+= (1f-2 * randn(Float32,2))
gₐ.lnσ .+= (1f-2 * randn(Float32,2))

gₑ.μ .= [5, 3]
gₑ.lnσ .= -10f0
gₑ.ult .= [atanh(1f0/3f0)*30f0 0f0;
          -atanh(2f0/3f0)*30f0 atanh(1f0/3f0)*30f0]

function train!(gₐ, gₑ, q::Float32,
                α0::Float32, αT::Float32; T=100)
   optₜ = ADAM(0.1, (0.9, 0.999))
   optᵣ = ADAM(0.1, (0.9, 0.999))

   gₜ = deepcopy(gₐ)
   gᵣ = deepcopy(gₐ)

   psₜ = params(gₜ.μ, gₜ.lnσ, gₜ.ult)
   psᵣ = params(gᵣ.μ, gᵣ.lnσ, gᵣ.ult)

   gₐs::Vector{Gaussian} = []
   gᵣs::Vector{Gaussian} = []
   gₜs::Vector{Gaussian} = []

   for t ∈ 0f0:T
      r = t/T
      α = (1f0-r)*α0 + r*αT

      if q == 1.0f0
         estshannon(gₜ, gₑ; opt=optₜ, ps=psₜ)
         mdshannon(gᵣ, gₐ, gₜ; opt=optᵣ, ps=psᵣ, α=α)
      else
         estshannon(gₜ, gₑ; opt=optₜ, ps=psₜ)
         mdtsallis(gᵣ, gₐ, gₜ; opt=optᵣ, ps=psᵣ, q=q, α=α)
      end
        
      push!(gₐs, deepcopy(gₐ))        
      push!(gₜs, deepcopy(gₜ))
      push!(gᵣs, deepcopy(gᵣ))
        
      gₐ.μ .= gᵣ.μ
      gₐ.lnσ .= gᵣ.lnσ
      gₐ.ult .= gᵣ.ult
   end
    
   gₐs, gᵣs, gₜs
end

d = Dict{Tuple{Float32,Float32,Float32},
         Tuple{Vector{Gaussian},
               Vector{Gaussian},
               Vector{Gaussian}}}()

for q ∈ qs, (α0, αT) ∈ zip(α0s, αTs)
   println("[q=$q, α0=$α0, αT=$αT]")
   d[q,α0,αT] = train!(deepcopy(gₐ),deepcopy(gₑ),q,α0,αT)
end
