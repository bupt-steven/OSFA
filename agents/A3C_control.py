#!/usr/bin/python
# -*- coding: utf-8 -*

# from agents.basic_agent.DQN import DeepQNetwork
# from agents.basic_agent.prioritized_replay_DQN import DeepQNetwork
from agents.basic_agent.A3C import *
import numpy as np


# from environments.basic_class.state_class import state
# from agents.basic_agent.RL_brain import DeepQNetwork




class A3C_ctrl:
    def __init__(self,
                 env_num_actions,
                 env_observation_shape,
                 num_agent,
                 agent_id_table,
                 num_OCS,
                 max_episode,
                 memory_size,
                 explore_strategy,
                 datetime,
                 ):
        self.num_actions = env_num_actions
        self.num_agent = num_agent
        self.agent_id_table = agent_id_table
        self.num_OCS = num_OCS
        self.explore_strategy = explore_strategy
        # self.action_temp = [1, 2]

        # self.agent = []
        # for fun_ID in range(self.num_agent):
        self.agent = DeepQNetwork(num_agent=num_agent,
                                  num_actions=env_num_actions,
                                  single_state_shape=env_observation_shape,
                                  max_episode=max_episode,
                                  datetime=datetime,
                                  learning_rate=0.01,
                                  reward_decay=0.9,
                                  e_greedy=0.9,
                                  replace_target_iter=3000,
                                  memory_size=memory_size,
                                  double_q=True,
                                  output_graph=True,
                                  )

    def choose_action(self, T_observation, D_observation, explore_strategy=None):
        action = []
        if explore_strategy is None:
            strategy = self.explore_strategy
        else:
            strategy = explore_strategy

        for OCS_ID in range(self.num_OCS):
            fun_id = self.agent_id_table[OCS_ID]
            topology = T_observation[OCS_ID, :, :]
            demand = D_observation[OCS_ID, :, :]
            # print(topology)
            # print(demand)
            id_vec = to_one_hot(fun_id, self.num_agent)
            observation = matrix2vector(demand, topology, id_vec)
            if strategy == "egreedy":
                action.append(self.agent.choose_action_egreedy(observation))
            elif strategy == "ucb1":
                action.append(self.agent.choose_action_UCB1(observation))
            elif strategy == "test":
                action.append(self.agent.choose_action_test(observation))
        return action

    def store_transition(self, D_observations, T_observations, action_list, reward, next_D_observations,
                         next_T_observations):
        for OCS_ID in range(self.num_OCS):
            fun_id = self.agent_id_table[OCS_ID]

            demand = D_observations[OCS_ID, :, :]
            topology = T_observations[OCS_ID, :, :]
            action = action_list[OCS_ID]
            # reward = reward
            demand_ = next_D_observations[OCS_ID, :, :]
            topology_ = next_T_observations[OCS_ID, :, :]
            id_vec = to_one_hot(fun_id, self.num_agent)

            transition = make_transition(demand, topology, action, reward, demand_, topology_, id_vec)
            # self.agent[fun_id].store_transition(transition)
            self.agent.store_transition(transition, fun_id)

    def learn(self, done):
        # for fun_id in range(self.num_agent):
        #     self.agent[fun_id].learn()
        self.agent.learn(done)

    def multi_model_save(self, path, global_step):
        self.agent.model_save(path, global_step)

    def multi_model_load(self, path_n_name):
        self.agent.model_load(path_n_name)

    def plot_cost(self):
        # for fun_ID in range(self.num_agent):
        #     self.agent[fun_ID].plot_cost()
        self.agent.plot_cost()


def matrix2vector(demand, topology, id_vec):
    observation = []
    for i in range(len(demand)):
        observation.extend(demand[i])
    for i in range(len(topology)):
        observation.extend(topology[i])
    observation.extend(id_vec)
    return np.array(observation)


def make_transition(demand, topology, action, reward, demand_, topology_, id_vec):
    transition = []
    for i in range(len(demand)):
        transition.extend(demand[i])
    for i in range(len(topology)):
        transition.extend(topology[i])
    transition.append(action)
    transition.append(reward)
    for i in range(len(demand_)):
        transition.extend(demand[i])
    for i in range(len(topology_)):
        transition.extend(topology[i])
    transition.extend(id_vec)
    return np.array(transition)


def to_one_hot(i, n):
    a = np.zeros(n, dtype=int)
    a[i] = 1
    # a = a[np.newaxis, :]
    return a
