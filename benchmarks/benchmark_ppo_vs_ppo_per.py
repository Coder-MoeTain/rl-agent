\
from stable_baselines3 import PPO
from custom_sb3_per import PPO_PER
from gym_pentest.env import PentestEnv
import numpy as np

def evaluate(model, env, episodes=5):
    scores = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward
        scores.append(total)
    return np.mean(scores), np.std(scores)

if __name__ == '__main__':
    env = PentestEnv()
    print('Training PPO baseline (short)...') 
    ppo = PPO('MlpPolicy', env, verbose=0)
    ppo.learn(10000)
    p_mean, p_std = evaluate(ppo, env)
    print('PPO mean, std:', p_mean, p_std)

    print('Training PPO+PER (short)...')
    env2 = PentestEnv()
    ppo_per = PPO_PER('MlpPolicy', env2, per_capacity=1024, per_alpha=0.6, per_beta_start=0.4, per_beta_frames=5000, per_batch_size=32, verbose=0)
    ppo_per.learn(10000)
    per_mean, per_std = evaluate(ppo_per, env2)
    print('PPO+PER mean, std:', per_mean, per_std)
