import gymnasium as gym
from gymnasium.envs.box2d import LunarLander
from gymnasium.wrappers import PassiveEnvChecker, OrderEnforcing, TimeLimit, RecordEpisodeStatistics, TransformReward

#env = gym.make("CartPole-v1", render_mode="human")

#env = gym.make("LunarLander-v3", render_mode="human")

env = LunarLander(render_mode='human')
env = PassiveEnvChecker(env)
env = OrderEnforcing(env)
env = TimeLimit(env, max_episode_steps=1000)

env = RecordEpisodeStatistics(env)
#env = RecordEpisodeStatistics(TransformReward(env, lambda r: r * 2))
#env = TransformReward(RecordEpisodeStatistics(env), lambda r: r * 2)

total_reward, steps = 0.0, 0

observation, info = env.reset(seed=42)
env.action_space.seed(42)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    if terminated or truncated:
        break

env.close()

print(env)
print(f"шагов: {steps}, награда: {total_reward}, "
      f"terminated={terminated}, truncated={truncated}")
print(f"Статистика: {info["episode"]}")