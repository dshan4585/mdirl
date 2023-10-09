const G = Gaussian
const DG = DiagGaussian

const iters = 1:500
const mask = tril!(ones(Float32, 2, 2), -1)

rs(x, pd) = logp(x, pd)
rt(x, pd; q) = (q-1)\(q*exp((q-1)*logp(x,pd))-1) + (q-1)*tent(pd; q=q) - 1

function estshannon(gₜ::G, gₑ::G; opt, ps)
   for _ ∈ iters
      gs = gradient(ps) do
         batsz \ sum([(-rs(sample(gₜ), gₑ) - ent(gₜ)) for _ ∈ 1:batsz])
      end
      gs[gₜ.ult] .*= mask
      Flux.Optimise.update!(opt, ps, gs)
   end
   nothing
end

function esttsallis(gₜ::G, gₑ::G; opt, ps, q)
   for _ ∈ iters
      gs = gradient(ps) do
         println(batsz \ sum([(-rt(sample(gₜ), gₑ; q=q) - tent(gₜ; q=q)) for _ ∈ 1:batsz]))
         batsz \ sum([(-rt(sample(gₜ), gₑ; q=q) - tent(gₜ; q=q)) for _ ∈ 1:batsz])
      end
      gs[gₜ.ult] .*= mask
      Flux.Optimise.update!(opt, ps, gs)
   end
   nothing
end

function mdshannon(gᵣ::G, gₐ::G, gₜ::G; opt, ps, α)
   for _ ∈ iters
      gs = gradient(ps) do
         (1f0/α) * kl(gᵣ, gₜ) + ((α-1f0)/α) * kl(gᵣ, gₐ)
      end
      gs[gᵣ.ult] .*= mask
      Flux.Optimise.update!(opt, ps, gs)
   end
   nothing
end

function mdtsallis(gᵣ::G, gₐ::G, gₜ::G; opt, ps, q, α)
   for _ ∈ iters
      gs = gradient(params(gᵣ.μ, gᵣ.lnσ, gᵣ.ult)) do
         (1f0/α) * tkl(gᵣ, gₜ; q=q) + ((α-1f0)/α) * tkl(gᵣ, gₐ; q=q)
      end
      gs[gᵣ.ult] .*= mask
      Flux.Optimise.update!(opt, ps, gs)
   end
   nothing
end
