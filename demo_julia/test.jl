using Test
include("pd.jl")
using .Pds

function gau2(d; diag=true)
   g₁ = Gaussian(d)
   g₂ = Gaussian(d)
   if diag
      if d == 1
         g₁.μ .= 0
         g₂.μ .= .2
         g₁.lnσ .= log(1)
         g₂.lnσ .= log(.5)
      elseif d == 2
         g₁.μ .= [.1,.2]
         g₂.μ .= [.5, -1]
         g₁.lnσ .= log.([.9,1.1])
         g₂.lnσ .= log.([.5,.6])   
      else
         g₁.μ .= [.2, .1,-.3]
         g₂.μ .= [.4,-.2,  0]
         g₁.lnσ .= log.([1.1, .6,.7])
         g₂.lnσ .= log.([.5, 1.5,.6])   
      end
   else
      if d == 1
         g₁.μ .= 0
         g₂.μ .= .2
         g₁.lnσ .= log(1)
         g₂.lnσ .= log(.5)
      elseif d == 2
         g₁.μ .= [.1,.2]
         g₂.μ .= [.5, -1]
         g₁.lnσ .= log.([.9,1.1])
         g₂.lnσ .= log.([.5,.6])
         g₁.ult .= [1  0; 
                   1.2 1]
         g₂.ult .= [1  0; 
                   -.8 1]    
      else
         g₁.μ .= [.2, .1,-.3]
         g₂.μ .= [.4,-.2,  0]
         g₁.lnσ .= log.([1.1, .6,.7])
         g₂.lnσ .= log.([.5, 1.5,.6])
         g₁.ult .= [1   0   0; 
                   .1   1   0;
                   .4 -.2   1]
         g₂.ult .= [1  0  0; 
                   -1  1  0;
                   .2 .5  1]  
      end
   end
   return g₁, g₂   
end

function diag2(d)
   dg₁ = DiagGaussian(d)
   dg₂ = DiagGaussian(d)

   if d == 1
      dg₁.μ .= 0
      dg₂.μ .= .2
      dg₁.lnσ .= log(1)
      dg₂.lnσ .= log(.5)
   elseif d == 2
      dg₁.μ .= [.1,.2]
      dg₂.μ .= [.5, -1]
      dg₁.lnσ .= log.([.9,1.1])
      dg₂.lnσ .= log.([.5,.6])    
   else
      dg₁.μ .= [.2, .1,-.3]
      dg₂.μ .= [.4,-.2,  0]
      dg₁.lnσ .= log.([1.1,.6,.7])
      dg₂.lnσ .= log.([.5,1.5,.6])
   end
   return dg₁, dg₂
end

(≃)(x, y) = isapprox(x, y; atol=1f-6)
@testset "Test ($(d)D)" for d ∈ 1:3
   g₁, g₂ = gau2(d)
   dg₁, dg₂ = diag2(d)
   fg₁, fg₂ = gau2(d; diag=false)
   @test ent(g₁) ≃ ent(dg₁)
   @test ent(g₂) ≃ ent(dg₂)

   @test logp(zeros(Float32, d), g₁) ≃ logp(zeros(Float32, d), dg₁)
   @test logp(ones(Float32, d), g₁) ≃ logp(ones(Float32, d), dg₁)
   @test logp(zeros(Float32, d), g₂) ≃ logp(zeros(Float32, d), dg₂)
   @test logp(ones(Float32, d), g₂) ≃ logp(ones(Float32, d), dg₂)
   
   @test tent(g₁; q=1.5f0) ≃ tent(dg₁; q=1.5f0)
   @test tent(g₁; q=2.0f0) ≃ tent(dg₁; q=2.0f0)

   @test tent(g₂; q=1.5f0) ≃ tent(dg₂; q=1.5f0)
   @test tent(g₂; q=2.0f0) ≃ tent(dg₂; q=2.0f0)

   @test tent(g₁; q=.5f0) ≃ tent(dg₁; q=.5f0)
   @test tent(g₁; q=.5f0) ≃ tent(dg₁; q=.5f0)
   @test tent(g₁; q=.5f0) ≃ tent(dg₁; q=.5f0)

   @test kl(g₁, g₁) ≃ 0
   @test kl(g₂, g₂) ≃ 0
   @test kl(dg₁, dg₁) ≃ 0
   @test kl(dg₂, dg₂) ≃ 0 
   @test kl(fg₁, fg₁) ≃ 0
   @test kl(fg₂, fg₂) ≃ 0      
   @test kl(g₁, g₂) ≃ kl(dg₁, dg₂)
   @test kl(g₂, g₁) ≃ kl(dg₂, dg₁)

   @test tkl(g₁, g₁; q=1.5f0) ≃ 0
   @test tkl(g₁, g₁; q=2.0f0) ≃ 0

   @test tkl(g₂, g₂; q=1.5f0) ≃ 0
   @test tkl(g₂, g₂; q=2.0f0) ≃ 0

   @test tkl(fg₁, fg₁; q=1.5f0) ≃ 0
   @test tkl(fg₁, fg₁; q=2.0f0) ≃ 0

   @test tkl(fg₂, fg₂; q=1.5f0) ≃ 0
   @test tkl(fg₂, fg₂; q=2.0f0) ≃ 0    

   @test tkl(dg₁, dg₁; q=1.5f0) ≃ 0
   @test tkl(dg₁, dg₁; q=2.0f0) ≃ 0

   @test tkl(dg₂, dg₂; q=1.5f0) ≃ 0   
   @test tkl(dg₂, dg₂; q=2.0f0) ≃ 0   

   @test tkl(g₁, g₂; q=1.5f0) ≃ tkl(dg₁, dg₂; q=1.5f0)
   @test tkl(g₁, g₂; q=2.0f0) ≃ tkl(dg₁, dg₂; q=2.0f0)

   @test tkl(g₂, g₁; q=1.5f0) ≃ tkl(dg₂, dg₁; q=1.5f0)
   @test tkl(g₂, g₁; q=2.0f0) ≃ tkl(dg₂, dg₁; q=2.0f0)

   @test tent(g₁; q=.97f0) > tent(g₁; q=.99f0) > ent(g₁) > tent(g₁; q=1.1f0) > tent(g₁; q=1.5f0) > tent(g₁; q=2.0f0)
   @test tent(g₂; q=.97f0) > tent(g₂; q=.99f0) > ent(g₂) > tent(g₂; q=1.1f0) > tent(g₂; q=1.5f0) > tent(g₂; q=2.0f0)
   @test tkl(g₁, g₂; q=.97f0) > tkl(g₁, g₂; q=.99f0) > kl(g₁, g₂) > tkl(g₁, g₂; q=1.1f0) > tkl(g₁, g₂; q=1.5f0) > tkl(g₁, g₂; q=2.0f0)
   @test tkl(g₂, g₁; q=.97f0) > tkl(g₂, g₁; q=.99f0) > kl(g₂, g₁) > tkl(g₂, g₁; q=1.1f0) > tkl(g₂, g₁; q=1.5f0) > tkl(g₂, g₁; q=2.0f0)

   @test tent(dg₁; q=.97f0) > tent(dg₁; q=.99f0) > ent(dg₁) > tent(dg₁; q=1.1f0) > tent(dg₁; q=1.5f0) > tent(dg₁; q=2.0f0)
   @test tent(dg₂; q=.97f0) > tent(dg₂; q=.99f0) > ent(dg₂) > tent(dg₂; q=1.1f0) > tent(dg₂; q=1.5f0) > tent(dg₂; q=2.0f0)
   @test tkl(dg₁, dg₂; q=.97f0) > tkl(dg₁, dg₂; q=.99f0) > kl(dg₁, dg₂) > tkl(dg₁, dg₂; q=1.1f0) > tkl(dg₁, dg₂; q=1.5f0) > tkl(dg₁, dg₂; q=2.0f0)
   @test tkl(dg₂, dg₁; q=.97f0) > tkl(dg₂, dg₁; q=.99f0) > kl(dg₂, dg₁) > tkl(dg₂, dg₁; q=1.1f0) > tkl(dg₂, dg₁; q=1.5f0) > tkl(dg₂, dg₁; q=2.0f0)

   @test tent(fg₁; q=.97f0) > tent(fg₁; q=.99f0) > ent(fg₁) > tent(fg₁; q=1.1f0) > tent(fg₁; q=1.5f0) > tent(fg₁; q=2.0f0)
   @test tent(fg₂; q=.97f0) > tent(fg₂; q=.99f0) > ent(fg₂) > tent(fg₂; q=1.1f0) > tent(fg₂; q=1.5f0) > tent(fg₂; q=2.0f0)
   @test tkl(fg₁, fg₂; q=.97f0) > tkl(fg₁, fg₂; q=.99f0) > kl(fg₁, fg₂) > tkl(fg₁, fg₂; q=1.1f0) > tkl(fg₁, fg₂; q=1.5f0) > tkl(fg₁, fg₂; q=2.0f0)
   @test tkl(fg₂, fg₁; q=.97f0) > tkl(fg₂, fg₁; q=.99f0) > kl(fg₂, fg₁) > tkl(fg₂, fg₁; q=1.1f0) > tkl(fg₂, fg₁; q=1.5f0) > tkl(fg₂, fg₁; q=2.0f0)    
end




