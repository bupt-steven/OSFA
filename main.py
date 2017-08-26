#!/usr/bin/python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environments.OSFA_env import OSFA
from agents.basic_agent.A3C import *
from agents.multi_DQN import multi_DQN
import math
import datetime
import time
import random
import logging
# from environments.basic_class.state_class import state
import os
import multiprocessing

# random seed
seed = 123 #625
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# output path
file_path = "./data/"
# stat_file_path = "./data/statistic/"

# topology parameter
N_TOR_PER_UNIT = 4
M_LEVEL = 1
FACTOR = 2
K_REGULAR = int(N_TOR_PER_UNIT / FACTOR)

# env parameter
ALPHA = 300                      # reward parameter : if used
CHANGE_INTERVAL = 100            # change topology every 10 * 0.1 ms = 1 ms
DISCONNECTED_MAXINT = 9999999

# todo Don't touch this ↓！
# I will solve this soon!
TEST = [True, False]
DESIGN = [False, True]
BIG_OCS = [True, True]
SMALL_OCS = [False, False]
SINGLE_MODEL, CODED_ACTION = [False, True]

# agent parameter
MEMORY_SIZE = 500

# training parameter
UPDATE_FREQ = 2
MAX_EPISODE = 1000

# training set
TRAIN_SIZE = 1
TEST_SIZE = 0
TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE
TEST_FREQUENCY = 100

#
data_id = [i for i in range(TOTAL_SIZE)]
res_list = [[] for _ in range(TOTAL_SIZE)]
reward_list = [[] for _ in range(TOTAL_SIZE)]

# statistic
min_res = 100000000
max_res = -1
columns = ["max", "min", "avg", "avg_reward", "learned", "count"]
statistic = pd.DataFrame(np.zeros([TOTAL_SIZE, 6]), columns=columns)
statistic.iloc[:]["max"] = max_res
statistic.iloc[:]["min"] = min_res
statistic.iloc[:]["learned"] = min_res

# logger
DATE = time.strftime('%Y-%m-%d', time.localtime(time.time()))

logger = logging.getLogger("OSFA")
logging.basicConfig(filename='./logs/logs/OSFA_' + str(DATE) + '.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(message)s')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


N_WORKERS = multiprocessing.cpu_count()
OUTPUT_GRAPH = True
LOG_DIR = './log'


env = OSFA(alpha=ALPHA, factor=FACTOR, n_ToR_per_unit=N_TOR_PER_UNIT, m_level=M_LEVEL, k_regular=K_REGULAR,
           distribution_id=0, change_interval=CHANGE_INTERVAL, coded_action=CODED_ACTION, single_model=SINGLE_MODEL,
           file_path=file_path, total_size=TOTAL_SIZE)

def run_single_demand(demand_id, step, test=False):
    D_observations, T_observations = env.reset(demand_id=demand_id)
    cnt = 0
    avg_reward = 0
    if test:
        explore_strategy = "test"
    else:
        explore_strategy = None

    while True:
        # agent choose action based on observation
        action = agent.choose_action(T_observations, D_observations, explore_strategy)
        logging.debug("action: " + str(action))

        # RL take action and get next observation and reward
        next_D_observations, next_T_observations, reward, done, disconnected = env.step(action)
        # print("reward: ", reward, "done:", done)
        # reward_list.append(reward)
        avg_reward = (avg_reward * cnt + reward) / (cnt + 1)

        if not test:
            agent.store_transition(D_observations, T_observations, action, reward, next_D_observations,
                                   next_T_observations)

            if (step > MEMORY_SIZE) and (step % UPDATE_FREQ == 0):
                # print(step)
                agent.learn(done)
                # print("I have learned")

        # swap observation
        D_observations = next_D_observations
        T_observations = next_T_observations

        if test and disconnected:
            break

        # break while loop when end of this episode
        if done:
            break

        if not test:
            step += 1
        cnt += 1
    if test and disconnected:
        current_time = DISCONNECTED_MAXINT
    else:
        current_time = env.current_time
    return step, cnt, current_time, avg_reward
    # return step, cnt, current_time, reward


def run():
    episode = 0
    test_episode = 0
    step = 0
    start = time.clock()
    env.demand_memory_setup(sample_num=TOTAL_SIZE, sample_index=data_id,
                            read_from_file=False, file_path=file_path_dmemore[0])
    while episode <= MAX_EPISODE * TRAIN_SIZE:
        if episode % (TRAIN_SIZE * TEST_FREQUENCY) == 0 and test_episode < TOTAL_SIZE:
            running_mode = "test"
            test_flag = True
            demand_id = data_id[test_episode % TOTAL_SIZE]
            # print(demand_id)
        else:
            # print("感觉有问题！")
            running_mode = "train"
            test_flag = False
            demand_id = data_id[episode % TRAIN_SIZE]
            test_episode = 0

        current_step, cnt, finish_time, reward = run_single_demand(demand_id, step, test=test_flag)

        if finish_time == DISCONNECTED_MAXINT:
            pass
        else:
            if finish_time > statistic.loc[demand_id]["max"]:
                statistic.loc[demand_id, "max"] = finish_time
            elif finish_time < statistic.loc[demand_id]["min"]:
                statistic.loc[demand_id, "min"] = finish_time
            statistic.loc[demand_id, "count"] += 1
            res_list[demand_id].append(finish_time)

            statistic.loc[demand_id, "avg"] = (statistic.loc[demand_id]["avg"] * (statistic.loc[demand_id]["count"] - 1)
                                               + finish_time) / statistic.loc[demand_id]["count"]
            statistic.loc[demand_id, "avg_reward"] = (statistic.loc[demand_id]["avg_reward"] *
                                                      (statistic.loc[demand_id]["count"] - 1) +
                                                      reward) / statistic.loc[demand_id]["count"]
            reward_list[demand_id].append(statistic.loc[demand_id, "avg_reward"])

        if running_mode == "train":
            episode += 1
        elif running_mode == "test":
            test_episode += 1
            # if finish_time < statistic.loc[demand_id]["learned"]:
            statistic.loc[demand_id, "learned"] = finish_time
            if test_episode == TOTAL_SIZE:
                # print("here we output!")
                DATETIME = time.strftime('%H-%M-%S', time.localtime(time.time()))

                metrics_file_path = file_path_dmemore[1] + "m_" + str(episode) + "_" + str(DATETIME) + ".csv"
                statistic.to_csv(metrics_file_path)

                res_pd = pd.DataFrame(res_list)
                res_file_path = file_path_dmemore[3] + "res_" + str(episode) + "_" + str(DATETIME) + ".csv"
                res_pd.to_csv(res_file_path)

                reward_pd = pd.DataFrame(reward_list)
                reward_file_path = file_path_dmemore[4] + "rew_" + str(episode) + "_" + str(DATETIME) + ".csv"
                reward_pd.to_csv(reward_file_path)

                q_pd = pd.DataFrame(agent.agent.q)
                q_file_path = file_path_dmemore[4] + "q_" + str(episode) + "_" + str(DATETIME) + ".csv"
                q_pd.to_csv(q_file_path)

                # episode += 1
                test_episode = TOTAL_SIZE

        step = current_step
        if episode % (TRAIN_SIZE+1) == 0 or running_mode == "test":
            logging.info(" episode: %8d  step: %8d  *%5s*  demand_id: %4d  change_times: %6d  current_time: %6.1f ms  "
                         % (episode, step, running_mode, demand_id, cnt, finish_time/10))
        # logging.info(" episode: " + str(episode) + "  step: " + str(step) + "  *" + running_mode + "* current time: "+
        #              str(finish_time/10) + " ms demand id: " + str(demand_id))

    DATETIME = time.strftime('%H-%M-%S', time.localtime(time.time()))

    model_path = file_path_dmemore[2]
    agent.multi_model_save(model_path, episode)

    end = time.clock()
    time_interval = end - start
    logging.info("time_interval: "+str(time_interval)+" s")
    logging.info("game over\n")

if __name__ == '__main__':
    # todo config output
    logging.info("--------------------OSFA     GAME     START!--------------------")
    file_path += str(DATE) + "/"
    file_path_dmemore = []
    if os.path.isdir(file_path):
        pass
    else:
        os.mkdir(file_path)
        logging.info("directory " + file_path + " has been created!")

    file_path_dmemore.append(file_path + "Demand/")     # [0]
    file_path_dmemore.append(file_path + "Metrics/")    # [1]
    file_path_dmemore.append(file_path + "Model/")      # [2]
    file_path_dmemore.append(file_path + "Result/")     # [3]
    file_path_dmemore.append(file_path + "Reward/")     # [4]
    for i in range(len(file_path_dmemore)):
        if os.path.isdir(file_path_dmemore[i]):
            pass
        else:
            os.mkdir(file_path_dmemore[i])
            logging.info("directory " + file_path_dmemore[i] + " has been created!")
    # is_test = False

    SESS = tf.Session()
    COORD = tf.train.Coordinator()
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(scope=GLOBAL_NET_SCOPE,
                          sess=SESS,
                          OPT_A=OPT_A,
                          OPT_C=OPT_C,
                          N_S=env.observation_shape[0]**2,
                          N_A=env.num_actions,
                          num_agent=K_REGULAR,
                          )  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
        # for i in range(8):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(name=i_name,
                                  globalAC=GLOBAL_AC,
                                  sess=SESS,
                                  COORD=COORD,
                                  opt_a=OPT_A,
                                  opt_c=OPT_C,
                                  n_ToR_per_unit=N_TOR_PER_UNIT,
                                  m_level=M_LEVEL,
                                  k_regular=K_REGULAR,
                                  distribution_id=0,
                                  change_interval=CHANGE_INTERVAL,
                                  file_path=file_path_dmemore[0],
                                  total_size=TOTAL_SIZE,
                                  ))

    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        # if os.path.exists(LOG_DIR):
        #     shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)


    # worker_threads = []
    worker_process = []
    for worker in workers:
        # job = lambda: worker.work()
        # t = threading.Thread(target=job)
        t = multiprocessing.Process(target=worker.work())
        t.start()
        t.join()
        # worker_threads.append(t)
        worker_process.append(t)

    # COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

    # # environment set up
    # # distribution_id : 0 1 2 3 data mining,cache,web search,hadoop "uniform"
    # env = OSFA(alpha=ALPHA, factor=FACTOR, n_ToR_per_unit=N_TOR_PER_UNIT, m_level=M_LEVEL, k_regular=K_REGULAR,
    #            distribution_id=0, change_interval=CHANGE_INTERVAL, coded_action=CODED_ACTION, single_model=SINGLE_MODEL,
    #            file_path=file_path_dmemore[0], total_size=TOTAL_SIZE)
    #
    # # env.save_demand(110, file_path=file_path, distribution_id=0)
    #
    # # agent set up
    # DATETIME = time.strftime('%H-%M-%S', time.localtime(time.time()))
    # agent = multi_DQN(env_num_actions=env.num_actions, env_observation_shape=env.observation_shape, num_agent=K_REGULAR,
    #                   agent_id_table=env.OCS_fun_ID_table, explore_strategy=explore_strategy, num_OCS=env.num_OCS,
    #                   max_episode=egreedy_max_step, memory_size=MEMORY_SIZE, datetime=DATETIME)
    #
    # # play and have fun
    # run()
