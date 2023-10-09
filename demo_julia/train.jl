const batsz = parse(Int, ARGS[1])
const seed = parse(Int, ARGS[2])

using Random; Random.seed!(seed)
include("train_core.jl")
using BSON: @save
@save "expr/batsz_$(batsz)_seed_$(seed).bson" d
