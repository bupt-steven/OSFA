#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# from environments.basic_class.state_class import state
# import tensorlayer
import math
import logging
import datetime


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # update data_frame
        self.update(leaf_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.8  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)  # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)

class DeepQNetwork:
    def __init__(
            self,
            num_agent,
            num_actions,
            single_state_shape,
            max_episode,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment="linear",
            datetime="to_be_added",
            output_graph=True,
            double_q=True,
            prioritized=True,
    ):
        self.num_agent = num_agent
        # self.fun_id = fun_id                 # one dimension input

        self.num_actions = num_actions
        # self.input_shape = input_shape     # [[num_ToR, num_ToR][self.num_OCS, self.num_ToR, self.num_ToR][num_ToR]
        self.single_state_shape = single_state_shape     # [[num_ToR, num_ToR], [self.num_ToR, self.num_ToR]]
        self.num_feature = single_state_shape[0]*single_state_shape[1]
        self.lr = learning_rate
        self.gamma = reward_decay

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.max_episode = max_episode

        self.double_q = double_q
        self.prioritized = prioritized

        self.datetime = datetime
        # total learning step
        self.learn_step_counter = 0

        # UCB1
        self.c = 2
        self.t = 0
        self.action_attempts = np.ones([self.num_actions])

        # initialize zero memory [s, a, r, s_]
        # state_shape = single_state_shape
        # state_shape.insert(0, 2)
        # self.memory = [[np.zeros(state_shape), 1, 1, np.zeros(state_shape)]
        #                for _ in range(self.memory_size)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            memory_class = np.zeros((self.memory_size, self.num_feature * 2 * 2 + 2 + self.num_agent))
            self.memory = [memory_class for _ in range(self.num_agent)]

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            self.writer = tf.summary.FileWriter("logs/" + str(self.datetime), self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        # print("this is DQN: " + str(self.fun_id))
        self.cost_history = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # input_shape = self.single_state_shape
        # input_shape.insert(0, None)
        self.s_d = tf.placeholder(tf.float32, [None, self.num_feature], name='s_d')  # input
        self.s_t = tf.placeholder(tf.float32, [None, self.num_feature], name='s_t')  # input
        self.s_id = tf.placeholder(tf.float32, [None, self.num_agent], name='s_id')
        self.q_target = tf.placeholder(tf.float32, [None, self.num_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        # scope_name = 'eval_net_'+str(self.fun_id)
        # c_names_temp = 'eval_net_params_' + str(self.fun_id)

        with tf.variable_scope("eval_net"):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES], math.ceil(self.single_state_shape[0] **2 /100)*100,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer for demand. collections is used later when assign to target net

            with tf.variable_scope('l1_d'):
                w1_d = tf.get_variable('w1_d', [self.num_feature, n_l1], initializer=w_initializer, collections=c_names)
                b1_d = tf.get_variable('b1_d', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_d = tf.nn.relu(tf.matmul(self.s_d, w1_d) + b1_d)

            # first layer for topology. collections is used later when assign to target net
            with tf.variable_scope('l1_t'):
                w1_t = tf.get_variable('w1_t', [self.num_feature, n_l1], initializer=w_initializer, collections=c_names)
                b1_t = tf.get_variable('b1_t', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_t = tf.nn.relu(tf.matmul(self.s_t, w1_t) + b1_t)

            total_l1_ = tf.concat([l1_d, l1_t], 1)
            total_l1 = tf.concat([total_l1_, self.s_id], 1)
            total_l1_shape = n_l1 * 2 + self.num_agent

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [total_l1_shape, total_l1_shape//4], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, total_l1_shape//4], initializer=b_initializer, collections=c_names)
                l2_t = tf.nn.relu(tf.matmul(total_l1, w2) + b2)

            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w2', [total_l1_shape//4, self.num_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, self.num_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2_t, w3) + b3

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            tf.summary.scalar("loss", self.loss)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_d_ = tf.placeholder(tf.float32, [None, self.num_feature], name='s_d')  # input
        self.s_t_ = tf.placeholder(tf.float32, [None, self.num_feature], name='s_t')  # input
        self.s_id_ = tf.placeholder(tf.float32, [None, self.num_agent], name='s_id')
        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        # scope_name = ['target_net'].append(str(self.fun_id))
        # scope_name = 'target_net_'+str(self.fun_id)
        # c_names_temp = 'target_net_params_' + str(self.fun_id)
        with tf.variable_scope("target_net"):
            # c_names(collections_names) are the collections to store variables
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            # # first layer. collections is used later when assign to target net
            # with tf.variable_scope('l1'):
            #     w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            #     b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            #     l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # first layer for demand. collections is used later when assign to target net
            with tf.variable_scope('l1_d'):
                w1_d = tf.get_variable('w1_d', [self.num_feature, n_l1], initializer=w_initializer,
                                       collections=c_names)
                b1_d = tf.get_variable('b1_d', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_d = tf.nn.relu(tf.matmul(self.s_d_, w1_d) + b1_d)

            # first layer for topology. collections is used later when assign to target net
            with tf.variable_scope('l1_t'):
                w1_t = tf.get_variable('w1_t', [self.num_feature, n_l1], initializer=w_initializer,
                                       collections=c_names)
                b1_t = tf.get_variable('b1_t', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_t = tf.nn.relu(tf.matmul(self.s_t_, w1_t) + b1_t)

            total_l1_ = tf.concat([l1_d, l1_t], 1)
            total_l1 = tf.concat([total_l1_, self.s_id_], 1)
            total_l1_shape = n_l1 * 2 + self.num_agent

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [total_l1_shape, total_l1_shape//4], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, total_l1_shape//4], initializer=b_initializer, collections=c_names)
                l2_t = tf.nn.relu(tf.matmul(total_l1, w2) + b2)

            # third layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w2', [total_l1_shape//4, self.num_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, self.num_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2_t, w3) + b3

    def store_transition(self, transition, fun_id):
        if self.prioritized:
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = np.zeros([self.num_agent], dtype=int)
            # replace the old memory with new memory
            index = int(self.memory_counter[fun_id] % self.memory_size)
            # print(len(transition))
            self.memory[fun_id][index][:] = transition
            # print("memory counter: ", self.memory_counter)
            # print(transition[512:514])
            self.memory_counter[fun_id] += 1

    # naive explore strategy with e-greedy
    def choose_action_egreedy(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s_d: observation[:, :self.num_feature],
                                                                  self.s_t: observation[:, self.num_feature:self.num_feature*2],
                                                                  self.s_id: observation[:, -self.num_agent:]})
            # print(actions_value)
            action = np.argmax(actions_value)

            if not hasattr(self, 'q'):  # 记录选的 Qmax 值
                self.q = []
                self.running_q = 0
                # tf.summary.scalar("q", self.running_q)
            self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
            # self.writer.add_summary(self.running_q, self.learn_step_counter)
            self.q.append(self.running_q)
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    # no explore when test, only choose action from DQN
    def choose_action_test(self, observation):
        # ob = hstack[d,t]
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]


        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s_d: observation[:, :self.num_feature],
                                                              self.s_t: observation[:, self.num_feature:self.num_feature*2],
                                                              self.s_id: observation[:, -self.num_agent:]})
        # print(actions_value)
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # 记录选的 Qmax 值
            self.q = []
            self.running_q = 0
            # tf.summary.scalar("q", self.running_q)
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
            # self.writer.add_summary(self.running_q, self.learn_step_counter)
        self.q.append(self.running_q)
        return action

    # explore strategy with UCB1 algorithm
    def choose_action_UCB1(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        exploration = np.log(self.t + 1) / self.action_attempts
        # exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)

        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s_d: observation[:, :self.num_feature],
                                                              self.s_t: observation[:, self.num_feature:self.num_feature*2],
                                                              self.s_id: observation[:, -self.num_agent:]})

        if not hasattr(self, 'q'):  # 记录选的 Qmax 值
            self.q = []
            self.running_q = 0
            # tf.summary.scalar("q", self.running_q)
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        # self.writer.add_summary(self.running_q, self.learn_step_counter)
        self.q.append(self.running_q)

        q_ucb = actions_value + exploration
        action = np.argmax(q_ucb)
        self.action_attempts[action] += 1
        check = np.where(q_ucb == action)[0]
        self.t += 1
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)

    def _replace_target_params(self):
        # target_names_temp = 'target_net_params_' + str(self.fun_id)
        # eval_names_temp = 'eval_net_params_' + str(self.fun_id)
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    # to be modified
    def learn(self, done):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            logging.debug('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            batch_memory = []
            single_size = int(self.batch_size / self.num_agent)
            for fun_id in range(self.num_agent):
                # sample batch memory from all memory
                if self.memory_counter[fun_id] > self.memory_size:
                    sample_index = np.random.choice(self.memory_size, size=single_size)
                else:
                    sample_index = np.random.choice(self.memory_counter[fun_id], size=single_size)

                if fun_id == 0:
                    batch_memory = self.memory[fun_id][sample_index][:]
                else:
                    batch_memory = np.row_stack([batch_memory, self.memory[fun_id][sample_index][:]])

        # ob_fun_id = [self.to_one_hot(self.fun_id, self.num_agent) for i in range(self.batch_size)]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_d_: batch_memory[:, -self.num_feature*2-self.num_agent:-self.num_feature-self.num_agent],  # fixed params
                self.s_t_: batch_memory[:, -self.num_feature-self.num_agent:-self.num_agent],
                self.s_id_: batch_memory[:, -self.num_agent:],
                self.s_d: batch_memory[:, :self.num_feature],  # newest params
                self.s_t: batch_memory[:, self.num_feature:self.num_feature*2],
                self.s_id: batch_memory[:, -self.num_agent:],
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.num_feature * 2].astype(int)
        reward = batch_memory[:, self.num_feature * 2 + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """
        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.s_d: batch_memory[:, :self.num_feature],
                                                                self.s_t: batch_memory[:, self.num_feature:self.num_feature*2],
                                                                self.s_id: batch_memory[:, -self.num_agent:],
                                                                self.q_target: q_target,
                                                                self.ISWeights: ISWeights})
            for i in range(len(tree_idx)):  # update priority
                idx = tree_idx[i]
                self.memory.update(idx, abs_errors[i])
            if self.learn_step_counter % 10 == 0:
                rs = self.sess.run(self.merged, feed_dict={self.s_d: batch_memory[:, :self.num_feature],
                                                           self.s_t: batch_memory[:,
                                                                     self.num_feature:self.num_feature * 2],
                                                           self.s_id: batch_memory[:, -self.num_agent:],
                                                           self.q_target: q_target,
                                                           self.ISWeights: ISWeights})
                self.writer.add_summary(rs, self.learn_step_counter)
        else:
            # train eval network
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s_d: batch_memory[:, :self.num_feature],
                                                    self.s_t: batch_memory[:, self.num_feature:self.num_feature*2],
                                                    self.s_id: batch_memory[:, -self.num_agent:],
                                                    self.q_target: q_target})
            if self.learn_step_counter % 10 == 0:
                rs = self.sess.run(self.merged, feed_dict={self.s_d: batch_memory[:, :self.num_feature],
                                                           self.s_t: batch_memory[:, self.num_feature:self.num_feature*2],
                                                           self.s_id: batch_memory[:, -self.num_agent:],
                                                           self.q_target: q_target})
                self.writer.add_summary(rs, self.learn_step_counter)


        self.cost_history.append(self.cost)
        # print("cost: %f" % self.cost)

        # increasing epsilon
        if self.epsilon_increment == "linear":
            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_max / self.max_episode
            else:
                pass

        else:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        logging.debug("epsilon: "+str(self.epsilon))
        self.learn_step_counter += 1

    def plot_cost(self):
        # plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        logging.debug("q:")
        logging.debug(self.q[-10:])
        plt.plot(np.arange(len(self.q)), self.q)
        plt.ylabel('q value')
        plt.xlabel('training steps')
        plt.show()

    def to_one_hot(self, i, n):
        a = np.zeros(n, dtype=int)
        a[i] = 1
        # a = a[np.newaxis, :]
        return a

    def model_save(self, path, global_step):
        save_path = self.saver.save(self.sess, path + "DQN", global_step=global_step)
        # save_path = self.saver.save(self.sess, "./model/DQN8_0.ckpt")
        logging.debug("Model saved in file: "+save_path)

    def model_load(self, path_n_name):
        # self.saver.restore(self.sess, "./model/DQN8_0.ckpt")
        self.saver.restore(self.sess, path_n_name)
        logging.debug("Model restored.")

