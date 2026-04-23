from stable_baselines3 import PPO
from environment import ChainReactionEnv
import os

# Create environment
env = ChainReactionEnv()

# model(PPO) setting 
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

print("Starting Training")
model.learn(total_timesteps=500000)

# Save the model for future use (enhance time execution)
if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/chain_reaction_v1")
print("Model Saved")