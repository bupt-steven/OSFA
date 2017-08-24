"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
# import gym
from environments.OSFA_env import OSFA
import math
import os
import shutil
import matplotlib.pyplot as plt
import logging

MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 20
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.01    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

class ACNet(object):
    def __init__(self, scope, sess, OPT_A, OPT_C, N_S, N_A, num_agent, globalAC=None):
        self.sess = sess
        self.OPT_A = OPT_A
        self.OPT_C = OPT_C
        self.N_S = N_S
        self.N_A = N_A
        self.n_agent = num_agent

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s_d = tf.placeholder(tf.float32, [None, self.N_S], 's_d')
                self.s_t = tf.placeholder(tf.float32, [None, self.N_S], name='s_t')  # input
                self.s_id = tf.placeholder(tf.float32, [None, self.n_agent], name='s_id')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s_d = tf.placeholder(tf.float32, [None, self.N_S], 's_d')
                self.s_t = tf.placeholder(tf.float32, [None, self.N_S], name='s_t')  # input
                self.s_id = tf.placeholder(tf.float32, [None, self.n_agent], name='s_id')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        n_l1 = math.ceil(self.N_S / 100)*100
        with tf.variable_scope('actor'):
            # l_a_d = tf.layers.conv2d(self.s_d, n_l1, kernel_size=4,padding="same", activation=tf.nn.relu6, kernel_initializer=w_init, name='la_d')
            # l_a_t = tf.layers.conv2d(self.s_t, n_l1, kernel_size=4,padding="same", activation=tf.nn.relu6, kernel_initializer=w_init, name='la_t')

            l_a_d = tf.layers.dense(self.s_d, n_l1, tf.nn.relu6, kernel_initializer=w_init, name='la_d')
            l_a_t = tf.layers.dense(self.s_t, n_l1, tf.nn.relu6, kernel_initializer=w_init, name='la_t')

            total_l1 = tf.concat([l_a_d, l_a_t], 1)
            total_l1 = tf.concat([total_l1, self.s_id], 1)
            # total_l1_shape = n_l1 * 2 + self.n_agent

            a_prob = tf.layers.dense(total_l1, self.N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            # l_c_d = tf.layers.conv2d(self.s_d, n_l1, kernel_size=4,padding="same", activation=tf.nn.relu6, kernel_initializer=w_init, name='lc_d')
            # l_c_t = tf.layers.conv2d(self.s_t, n_l1, kernel_size=4,padding="same", activation=tf.nn.relu6, kernel_initializer=w_init, name='lc_t')

            l_c_d = tf.layers.dense(self.s_d, n_l1, tf.nn.relu6, kernel_initializer=w_init, name='lc_d')
            l_c_t = tf.layers.dense(self.s_t, n_l1, tf.nn.relu6, kernel_initializer=w_init, name='lc_t')

            total_l1_ = tf.concat([l_c_d, l_c_t], 1)
            total_l1_ = tf.concat([total_l1_, self.s_id], 1)
            # total_l1_shape = n_l1 * 2 + self.n_agent

            v = tf.layers.dense(total_l1_, 1, kernel_initializer=w_init, name='v')  # state value
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, observation, is_test):  # run by a local
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s_d: observation[np.newaxis, :self.N_S],
                                                             self.s_t: observation[np.newaxis, self.N_S:self.N_S*2],
                                                             self.s_id: observation[np.newaxis, -self.n_agent:]})
        if is_test:
            action = np.argmax(prob_weights)
        else:
            action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(
            self,
            sess,
            name,
            globalAC,
            COORD,
            opt_a,
            opt_c,
            n_ToR_per_unit,
            m_level,
            k_regular,
            distribution_id,
            change_interval,
            file_path,
            total_size,
    ):
        self.sess = sess
        self.OPT_A = opt_a
        self.OPT_C = opt_c
        self.file_path = file_path

        self.env = OSFA(n_ToR_per_unit=n_ToR_per_unit, m_level=m_level, k_regular=k_regular,
                        distribution_id=distribution_id, change_interval=change_interval,
                        file_path=file_path, total_size=total_size)
        self.num_feature = self.env.observation_shape[0]**2
        self.num_actions = self.env.num_actions
        self.num_OCS = self.env.num_OCS
        self.agent_id_table = self.env.OCS_fun_ID_table
        self.num_agent = self.env.k_regular
        self.total_size = total_size

        self.name = name
        self.AC = ACNet(scope=name, sess=self.sess, globalAC=globalAC, OPT_A=self.OPT_A, OPT_C=self.OPT_C,
                        N_S=self.num_feature, N_A=self.num_actions, num_agent=self.num_agent)
        self.coord = COORD
        data_id = [i for i in range(self.total_size)]
        if self.name == 'W_0':
            self.env.demand_memory_setup(sample_num=self.total_size, sample_index=data_id,
                                         read_from_file=False, file_path=self.file_path)
        else:
            self.env.demand_memory_setup(sample_num=self.total_size, sample_index=data_id,
                                         read_from_file=True, file_path=self.file_path)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s_d, buffer_s_t, buffer_s_id, buffer_a, buffer_r = [], [], [], [], []
        while not self.coord.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            D_observations, T_observations = self.env.reset(demand_id=0)
            ep_r = 0
            while True:
                if GLOBAL_EP % 200 == 0:
                    running_mode = "test"
                    action, id_vec_list = self.choose_action_merge(T_observations, D_observations, is_test=True)
                else:
                    running_mode = "train"
                    action, id_vec_list = self.choose_action_merge(T_observations, D_observations)
                # print(action)
                next_D_observations, next_T_observations, reward, done, disconnected = self.env.step(action)
                # s_, r, done, info = self.env.step(action)
                ep_r += reward
                for i in range(self.num_OCS):
                    buffer_s_d.append(np.reshape(D_observations[i][:], -1))
                    buffer_s_t.append(np.reshape(T_observations[i][:], -1))
                    buffer_s_id.append(id_vec_list[i][:])
                    buffer_a.append(action[i])
                    buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s_d: np.reshape(next_D_observations, [self.num_OCS, -1]),
                                                         self.AC.s_t: np.reshape(next_T_observations, [self.num_OCS, -1]),
                                                         self.AC.s_id: id_vec_list})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s_d, buffer_s_t, buffer_s_id, buffer_a, buffer_v_target \
                        = np.vstack(buffer_s_d), np.vstack(buffer_s_t), np.vstack(buffer_s_id), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s_d: buffer_s_d,
                        self.AC.s_t: buffer_s_t,
                        self.AC.s_id: buffer_s_id,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s_d, buffer_s_t, buffer_s_id, buffer_a, buffer_r = [], [], [], [], []
                    self.AC.pull_global()

                D_observations = next_D_observations
                T_observations = next_T_observations
                # logging.info("current time: %6.2f" % self.env.current_time)
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| running mode: %5s" % running_mode,
                        "| Ep_r: %6.2f" % GLOBAL_RUNNING_R[-1],
                        "| current time: %6.2f" % self.env.current_time,
                          )
                    GLOBAL_EP += 1
                    break


    def choose_action_merge(self, T_observation, D_observation, is_test=False):
        action = []
        id_vec_list = []
        for OCS_ID in range(self.env.num_OCS):
            fun_id = self.agent_id_table[OCS_ID]
            topology = T_observation[OCS_ID, :, :]
            demand = D_observation[OCS_ID, :, :]
            # print(topology)
            # print(demand)
            id_vec = to_one_hot(fun_id, self.num_agent)
            id_vec_list.append(id_vec)
            observation = matrix2vector(demand, topology, id_vec)

            action.append(self.AC.choose_action(observation, is_test))
        return action, id_vec_list

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

