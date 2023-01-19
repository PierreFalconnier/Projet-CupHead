import gym

from stable_baselines3 import DQN

# env = gym.make("CartPole-v1")
env = gym.make('LunarLander-v2')


###################


# # Gym is an OpenAI toolkit for RL
# import gym
# from gym.spaces import Box
# from gym.wrappers import FrameStack
# # NES Emulator for OpenAI Gym
# from nes_py.wrappers import JoypadSpace
# # Super Mario environment for OpenAI Gym
# import gym_super_mario_bros
# # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
# env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# # Limit the action-space to : walk right, jump right
# env = JoypadSpace(env, [["right"], ["right", "A"]])



# model = DQN("MlpPolicy", env, verbose=1, buffer_size=100000, tensorboard_log="generic/stable_baseline/logdir")
model = DQN('MlpPolicy', env, learning_rate=1e-3,  verbose=1)
# model = DQN("MlpPolicy",
#                env,
#                verbose=1, 
#                buffer_size=50000, 
#                tensorboard_log="generic/stable_baseline/logdir", 
#                learning_starts=25000,
#                learning_rate=0.0004,
#                batch_size=32,
#                gamma=0.99,
#                train_freq=4,
#                target_update_interval=10000,
#                exploration_fraction=0.1,
#                exploration_initial_eps=1,
#                exploration_final_eps=0.05)
model.learn(total_timesteps=1000, log_interval=1)
model.save("generic/stable_baseline/dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("generic/stable_baseline/dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()