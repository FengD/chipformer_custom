from gym.envs.registration import register


register(
	id = 'place_env-v1',
	entry_point = 'mingpt.trainer_placement:Env'
)
