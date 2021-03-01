import gym
from IPython import embed
from ur5 import UR5Env, EndEffectorPositioningUR5DoF5
#target_machine_add = '192.168.0.3:2150'
target_machine_add = 'localhost:52349'

print("bla")
# initialize environment
num_episodes = 100
env = EndEffectorPositioningUR5DoF5(rs_address=target_machine_add, gui=True, max_episode_steps=num_episodes)
print("blah")

d = env.reset(is_training=True)
print("length ", len(d))
embed()
for episode in range(num_episodes-10):
    # random step in the environment
    state, reward, done, info = env.step(env.action_space.sample())
    print(done)
    #print(len(env.action_space.sample()))
    #print(reward)
    print(len(state))
