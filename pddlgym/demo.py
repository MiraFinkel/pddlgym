"""Demonstrates basic PDDLGym usage with random action sampling
"""
import matplotlib  # matplotlib.use('agg')

from Agents.RL_agents import rl_agent
from Agents.RL_agents.q_learning_agents import Q_LEARNING
from constants import *

matplotlib.use('agg')  # For rendering

from pddlgym.pddlgym.utils import run_demo
import pddlgym.pddlgym as pddlgym
import imageio


def demo_random(env_name, render=True, problem_index=0, verbose=True):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
    env.fix_problem_index(problem_index)
    policy = lambda s: env.action_space.sample(s)
    video_path = "/tmp/{}_random_demo.mp4".format(env_name)
    run_demo(env, policy, render=render, verbose=verbose, seed=0,
             video_path=video_path)


def run_all(render=True, verbose=True):
    ## Some probabilistic environments
    demo_random("explodingblocks", render=render, verbose=verbose)
    # demo_random("tireworld", render=render, verbose=verbose)
    # demo_random("river", render=render, verbose=verbose)

    ## Some deterministic environments
    # demo_random("sokoban", render=render, verbose=verbose)
    # demo_random("gripper", render=render, verbose=verbose)
    # demo_random("rearrangement", render=render, problem_index=6, verbose=verbose)
    # demo_random("minecraft", render=render, verbose=verbose)
    # demo_random("blocks", render=render, verbose=verbose)
    # demo_random("blocks_operator_actions", render=render, verbose=verbose)
    # demo_random("quantifiedblocks", render=render, verbose=verbose)
    # demo_random("fridge", render=render, verbose=verbose)


def create_env(env_name, operators_as_actions=True):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()), operators_as_actions=operators_as_actions)
    env.fix_problem_index(0)
    obs = env.reset()
    env_action_space = env.action_space
    actions = set()
    for i in range(100):
        actions.add(env_action_space.sample(obs))
    num_states = 600000
    setattr(env_action_space, "n", env.action_space.num_predicates)
    setattr(env, "actions", env.action_space.predicates)
    setattr(env, "num_states", num_states)
    return env


if __name__ == '__main__':
    env_name = "Snake"
    # simple usage example
    env = create_env(env_name, operators_as_actions=True)
    obs = env.reset()
    img = env.render()
    imageio.imsave("frame1.png", img)  # Saved in \PDDLgym\pddlgym\pddlgym\
    obs, reward, done, debug_info = env.step(0)
    img = env.render()
    imageio.imsave("frame2.png", img)  # Saved in \PDDLgym\pddlgym\pddlgym\

    # agent creation and training
    env = create_env(env_name, operators_as_actions=True)
    agent_name = Q_LEARNING
    num_of_episodes = 50
    agent = rl_agent.create_agent(env, agent_name)
    train_result = rl_agent.run(agent, num_of_episodes, method=TRAIN)
    print(train_result)
