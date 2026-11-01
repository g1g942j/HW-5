import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

env = gym.make("CartPole-v1")
env = RecordEpisodeStatistics(env, buffer_length=100)
env.action_space.seed(42)


BINS = [np.array([]),
        np.array([-0.5, 0.5]),
        np.linspace(-0.2, 0.2, 7),
        np.linspace(-2, 2, 11)]

def cell(obs):
    return tuple(int(np.digitize(x, b)) for x, b in zip(obs, BINS))

Q = np.zeros((1, 3, 8, 12, 2))
rng = np.random.default_rng(42)

for ep in range(1, 1501):
    eps = max(0.01, min(1.0, 1 - np.log10(ep / 25)))
    alpha = max(0.1, min(0.5, 1 - np.log10(ep / 25)))

    observation, info = env.reset(seed=42 if ep == 1 else None)
    s = cell(observation)

    while True:
        action = env.action_space.sample() if rng.random() < eps else int(np.argmax(Q[s]))
        observation, reward, terminated, truncated, info = env.step(action)

        s2 = cell(observation)
        target = reward + (0 if terminated else 0.99 * Q[s2].max())
        Q[s + (action,)] += alpha * (target - Q[s + (action,)])
        s = s2

        if terminated or truncated:
            break

    if ep % 100 == 0:
        print(f"эпизод {ep:4d} | средняя награда за 100 эп.: {np.mean(env.return_queue):5.1f} | eps={eps:.2f}")
env.close()




env = RecordEpisodeStatistics(gym.make("CartPole-v1", render_mode="human"))
observation, info = env.reset(seed=0)
while True:
    observation, reward, terminated, truncated, info = env.step(int(np.argmax(Q[cell(observation)])))
    if terminated or truncated:
        break
env.close()
print(env)
print(f"Обученный агент: {info['episode']['l']} шагов из 500, terminated={terminated}, truncated={truncated}")