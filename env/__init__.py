from env.env import Env
from env.env import AntEnv
from gym.envs.registration import register

register(
    id='SlopeHopper1-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 1},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeHopper3-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 3},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeHopper5-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 5},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeHopper7-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 7},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeHopper9-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 9},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeHopper12-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 12},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeHopper15-v0',
    entry_point='env.slope_hopper:SlopeHopperEnv',
    kwargs={'degree' : 15},
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='SlopeWalker2d1-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 1}
)

register(
    id='SlopeWalker2d3-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 3}
)

register(
    id='SlopeWalker2d5-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 5}
)

register(
    id='SlopeWalker2d7-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 7}
)

register(
    id='SlopeWalker2d9-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 9}
)

register(
    id='SlopeWalker2d12-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 12}
)

register(
    id='SlopeWalker2d15-v0',
    max_episode_steps=1000,
    entry_point='env.slope_walker2d:SlopeWalker2dEnv',
    kwargs={'degree' : 15}
)

register(
    id='CrippledAnt100-v0',
    kwargs={'crippling_range' : 1.0},
    entry_point='env.crippled_ant:CrippledAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

register(
    id='CrippledAnt75-v0',
    kwargs={'crippling_range' : 0.75},
    entry_point='env.crippled_ant:CrippledAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

register(
    id='CrippledAnt50-v0',
    kwargs={'crippling_range' : 0.5},
    entry_point='env.crippled_ant:CrippledAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

register(
    id='CrippledAnt25-v0',
    kwargs={'crippling_range' : 0.25},
    entry_point='env.crippled_ant:CrippledAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

register(
    id='CrippledAnt1-v0',
    kwargs={'crippling_range' : 0.01},
    entry_point='env.crippled_ant:CrippledAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

register(
    id='MultiGoal-v0',
    entry_point='env.multigoal:MultiGoalEnv',
    max_episode_steps=100
)

register(
    id='MultiGoalAnt-v0',
    entry_point='env.multigoal_ant:MultiGoalAntEnv',
    max_episode_steps=300,
    reward_threshold=6000.0
)

register(
    id='EastAnt-v0',
    entry_point='env.east_ant:EastAntEnv',
    max_episode_steps=300,
    reward_threshold=6000.0
)

register(
    id='WestAnt-v0',
    entry_point='env.west_ant:WestAntEnv',
    max_episode_steps=300,
    reward_threshold=6000.0
)

register(
    id='NorthAnt-v0',
    entry_point='env.north_ant:NorthAntEnv',
    max_episode_steps=300,
    reward_threshold=6000.0
)

register(
    id='SouthAnt-v0',
    entry_point='env.south_ant:SouthAntEnv',
    max_episode_steps=300,
    reward_threshold=6000.0
)

register(
    id='AsymmetricMultiGoal-v0',
    entry_point='env.asym_multigoal:AsymmetricMultiGoalEnv',
    max_episode_steps=100
)

register(
    id='AsymmetricEast-v0',
    entry_point='env.asym_east:AsymmetricEastEnv',
    max_episode_steps=100
)

register(
    id='AsymmetricWest-v0',
    entry_point='env.asym_west:AsymmetricWestEnv',
    max_episode_steps=100
)

register(
    id='AsymmetricSouth-v0',
    entry_point='env.asym_south:AsymmetricSouthEnv',
    max_episode_steps=100
)

register(
    id='AsymmetricNorth-v0',
    entry_point='env.asym_north:AsymmetricNorthEnv',
    max_episode_steps=100
)

register(
    id='AsymmetricHardMultiGoal-v0',
    entry_point='env.asym_hard_multigoal:AsymmetricHardMultiGoalEnv',
    max_episode_steps=300
)

register(
    id='AsymmetricHardEast-v0',
    entry_point='env.asym_hard_east:AsymmetricHardEastEnv',
    max_episode_steps=300
)

register(
    id='AsymmetricHardWest-v0',
    entry_point='env.asym_hard_west:AsymmetricHardWestEnv',
    max_episode_steps=300
)

register(
    id='AsymmetricHardSouth-v0',
    entry_point='env.asym_hard_south:AsymmetricHardSouthEnv',
    max_episode_steps=300
)

register(
    id='AsymmetricHardNorth-v0',
    entry_point='env.asym_hard_north:AsymmetricHardNorthEnv',
    max_episode_steps=300
)