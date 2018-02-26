
# coding: utf-8

import numpy as np
import tensorflow as tf
import random
from collections import deque
import gym
from Network import DQN
import Exploration
import Train
import matplotlib.pyplot as plt

class RL:
    def __init__(self, sess, dis = 0.99, REPLAY_MEMORY = 10000, batch_size = 256, max_episodes = 500,
                 training_step = 10, copy_step = 40,
                 layer_size = 64, learning_rate = 0.01, action_res = None,
                 alpha = 0.6, beta_init = 0.4, eps = 0.01, eps_div = 10, repu_num = 1,
                 ending_cond_epis = 100, ending_cond_reward = 195, reward_p = 100, reward_n = -100,
                 seed_n = 0, Exp = 'softmax', Game = 'CartPole-v0', save_epi = 100,
                 Double = True, Dueling = True, Prioritized = True):
        
        env = gym.make(Game)
        tf.set_random_seed(seed_n)
        
        input_size = env.observation_space.shape[0]
        output_size, conti_action_flag, action_map = self.get_action_information(env, Game, action_res=action_res)
        
        self.sess = sess
        self.dis = dis
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.batch_size = batch_size
        self.training_step = training_step
        self.copy_step = copy_step
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta_init = beta_init
        self.eps = eps
        self.eps_div = eps_div
        self.repu_num = repu_num
        self.seed_n = seed_n
        self.Exp = Exp
        self.Game = Game
        self.save_epi = save_epi
        self.max_episodes = max_episodes
        self.env = env
        self.input_size = input_size
        self.output_size = output_size
        self.Double = Double
        self.Dueling = Dueling
        self.Prioritized = Prioritized
        
        #tf.reset_default_graph()
        self.mainDQN = DQN(sess, input_size, output_size, seed_n, layer_size, learning_rate, name="main")
        self.targetDQN = DQN(sess, input_size, output_size, seed_n, layer_size, learning_rate, name="target")
        self.copy_ops = Train.get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
        
        self.reward_p = reward_p
        self.reward_n = reward_n
        self.ending_cond_epis = ending_cond_epis
        self.ending_cond_reward = ending_cond_reward
        
        self.conti_action_flag=conti_action_flag
        self.action_map=action_map
    
    
    def get_action_information(self, env, env_name, action_res=None):
        action_map = []
        if isinstance(env.action_space, gym.spaces.Box):
            conti_action_flag = True
            if env_name == "Pendulum-v0" or env_name == "InvertedPendulum-v1" or env_name == "MountainCarContinuous-v0" or env_name == "InvertedDoublePendulum-v1":
                action_map = np.linspace(env.action_space.low[0],env.action_space.high[0],num=action_res)
            
            elif env_name == "Reacher-v1":
                action_map = np.zeros([np.prod(action_res), 2])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        s = action_res[1] * i + j
                        action_map[s, :] = [u[i], v[j]]
            
            elif env_name == "Swimmer-v1" or env_name == "LunarLanderContinuous-v2" or env_name == "MultiGoal-v0":
                action_map = np.zeros([np.prod(action_res), 2])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        s = action_res[1] * i + j
                        action_map[s, :] = [u[i], v[j]]
            
            elif env_name == "Hopper-v1":
                action_map = np.zeros([np.prod(action_res), 3])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                w = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        for k in range(action_res[2]):
                            s = action_res[2] * action_res[1] * i + action_res[2] * j + k
                            action_map[s, :] = [u[i], v[j], w[k]]
            
            elif env_name == "Walker2d-v1":
                action_map = np.zeros([np.prod(action_res), 6])
                x = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                y = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                z = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                u = np.linspace(env.action_space.low[3], env.action_space.high[3], num=action_res[3])
                v = np.linspace(env.action_space.low[4], env.action_space.high[4], num=action_res[4])
                w = np.linspace(env.action_space.low[5], env.action_space.high[5], num=action_res[5])
                for i0 in range(action_res[0]):
                    for i1 in range(action_res[1]):
                        for i2 in range(action_res[2]):
                            for i3 in range(action_res[3]):
                                for i4 in range(action_res[4]):
                                    for i5 in range(action_res[5]):
                                        s = np.prod(action_res[1:]) * i0
                                        s += np.prod(action_res[2:]) * i1
                                        s += np.prod(action_res[3:]) * i2
                                        s += np.prod(action_res[4:]) * i3
                                        s += np.prod(action_res[5:]) * i4
                                        s += i5
                                        action_map[s, :] = [x[i0], y[i1], z[i2], u[i3], v[i4], w[i5]]
            else:
                print(env.action_space.high.shape[0])
            n_action = np.prod(action_res)
        
        elif isinstance(env.action_space, gym.spaces.Discrete):
            conti_action_flag = False
            n_action = env.action_space.n
        
        else:
            raise NotImplementedError("{} action spaces are not supported yet.".format(type(env.action_space)))
        return n_action, conti_action_flag, action_map
    
    
    def run_RL(self, case_n = 1, Exp = 'softmax', seed_n = 0,
                                   Double = True, Dueling = True, Prioritized = True):
        sess = self.sess
        dis = self.dis
        REPLAY_MEMORY = self.REPLAY_MEMORY
        batch_size = self.batch_size
        training_step = self.training_step
        copy_step = self.copy_step
        alpha = self.alpha
        beta_init = self.beta_init
        eps = self.eps
        eps_div = self.eps_div
        repu_num = self.repu_num
        Game = self.Game
        save_epi = self.save_epi
        max_episodes = self.max_episodes
        env = self.env
        input_size = self.input_size
        output_size = self.output_size
        
        reward_p = self.reward_p
        reward_n = self.reward_n
        ending_cond_epis = self.ending_cond_epis
        ending_cond_reward = self.ending_cond_reward
        
        conti_action_flag = self.conti_action_flag
        action_map = self.action_map
        
        env.seed(seed_n)
        np.random.seed(seed_n)
        tf.set_random_seed(seed_n)
        random.seed(seed_n)
        
        mainDQN = self.mainDQN
        if Double:
            targetDQN = self.targetDQN
            copy_ops = self.copy_ops
        
        end_episode = 0
        step_count_total = 0
        global_step = 0
        loss = 0
        
        replay_buffer = deque()
        TD_error_list = []
        steps_list = []
        step_avg_list = []
        
        tf.initialize_all_variables().run()
        if Double:
            sess.run(copy_ops)
        
        print("")
        print("CASE {}".format(case_n))
        print("  Exp : {}".format(Exp))
        print("  Strategy : Double : {}, Dueling : {}, Prioritized : {}".format(Double, Dueling, Prioritized))
        
        for episode in range(1, max_episodes+1):
            done = False
            step_count = 0
            TD_error = 0
            state = env.reset()
            
            while not done:
                #######
                e = 1. / (((episode - 1) / eps_div) + 1)
                action = Exploration.choice_action(Exp, e, mainDQN.predict(state)[0])
                
                if conti_action_flag:
                    action0 = [action_map[action]]
                else:
                    action0 = action
                
                next_state, reward, done, _ = env.step(action0)
                step_count += reward
                global_step += 1

                if Prioritized:
                    if done:
                        if step_count < 200:
                            reward = reward_n
                        TD_error = reward
                    else:
                        if Exp == 'epsilon':
                            if Double:
                                action1 = np.argmax(mainDQN.predict(next_state))
                                TD_error = reward + dis * (targetDQN.predict(next_state)[0, action1])
                            else:
                                TD_error = reward + dis * np.max(mainDQN.predict(next_state))
                        else:
                            if Double:
                                TD_error = Exploration.Q_value(Exp, reward, dis, targetDQN.predict(next_state)[0])
                            else:
                                TD_error = Exploration.Q_value(Exp, reward, dis, mainDQN.predict(next_state)[0])
                    TD_error -= np.max(mainDQN.predict(state))
                    TD_error = pow((abs(TD_error) + eps), alpha)

                else:
                    #if done:
                    #    if step_count < 200:
                    #        reward = reward_n
                    TD_error = 1

                TD_error_list.append(TD_error)

                #if beta < 1:
                #    beta +=(1 - beta_init)/REPLAY_MEMORY

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state

                if global_step > batch_size and global_step % training_step == 0:
                    for re in range(repu_num):
                        TD_copy = []
                        TD_norm_list = []
                        W_is_list = []

                        start = 0
                        len_TD = len(TD_error_list)
                        if(len_TD > REPLAY_MEMORY):
                            start = len_TD - REPLAY_MEMORY
                            TD_copy = TD_error_list[start : len_TD]
                            len_TD = REPLAY_MEMORY
                        else:
                            TD_copy = TD_error_list[:]
                        sum_TD = sum(TD_copy)

                        TD_norm_list = np.array(TD_copy) / float(sum_TD)

                        #W_is_list = [np.power((REPLAY_MEMORY * TD_norm_list[i]), -beta) for i in range(len_TD)]
                        #maxW = np.max(W_is_list)
                        #W_is_list = [W_is_list[i] / maxW for i in range(len_TD)]

                        W_is_list = np.ones([len(TD_norm_list)])

                        minibatch = []
                        w_batch = []

                        TDT = np.zeros([len(TD_norm_list)])
                        TD_index = 0
                        for i in range(batch_size):
                            check = True
                            while check:
                                TD_choice = np.random.choice(len(TD_norm_list),size=1,p = TD_norm_list)
                                TD_index = TD_choice[0]
                                if TDT[TD_index] == 0:
                                    TDT[TD_index] = 1
                                    check = False

                            w_batch.append(W_is_list[TD_index])
                            minibatch.append(replay_buffer[TD_index])

                        if Double:
                            loss, _ = Train.replay_train(mainDQN, targetDQN, minibatch, w_batch,
                                                                         Exp, dis, input_size, output_size)
                        else:
                            loss, _ = Train.replay_train_unDouble(mainDQN, minibatch, w_batch,
                                                                         Exp, dis, input_size, output_size)

                if Double and global_step % copy_step == 0:
                    sess.run(copy_ops)

            #print("episode: {}   steps: {}".format(episode, step_count))
            steps_list.append(step_count)
            #print("GLOBAL STEP  :  ", global_step)
            
            if episode % save_epi == 0:
                mainDQN.save_network(episode = episode, save_epi = save_epi)
            
            if episode < ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / episode)

            if episode == ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / ending_cond_epis)
                #print ("Step Average 100:  ", step_avg_list[episode - 1])

            if episode > ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                step_avg_list.append(step_count_total / ending_cond_epis)
                #print ("Step Average 100:  ", step_avg_list[episode - 1])

            print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
            print ("                   ( Result : {},  Loss : {},  epsilon : {} )"
                               .format(round(step_count, 5), round(loss, 8), round(e, 3)))
            end_episode += 1
            if step_avg_list[episode - 1] > ending_cond_reward:
                break
        
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        for episode in range(end_episode + 1, max_episodes+1):
            s = env.reset()
            reward_sum = 0
            while True :
                #env.render()
                action = np.argmax(mainDQN.predict(s))
                if conti_action_flag:
                    action0 = [action_map[action]]
                else:
                    action0 = action
                
                s,reward,done,_ = env.step(action0)
                reward_sum += reward
                
                if episode % save_epi == 0:
                    mainDQN.save_network(episode = episode, save_epi = save_epi)
                
                if done :
                    #print("episode: {}   steps: {}".format(episode, reward_sum))
                    steps_list.append(reward_sum)
                    step_count_total += steps_list[episode - 1]
                    step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                    step_avg_list.append(step_count_total / ending_cond_epis)
                    print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
                    print ("                   ( Result : {},  Loss : {},  epsilon : {} )".format(step_count, loss, e))
                    break
        
        x_values = list(range(1, max_episodes+1))
        y_values = step_avg_list[:]
        plt.plot(x_values, y_values, c='green')
        plt.grid(True)
        plt.show()

        
    def run_EV(self, case_n = 1, Exp = 'softmax', seed_n = 0, saved_num = 1):
        sess = self.sess
        Game = self.Game
        save_epi = self.save_epi
        max_episodes = self.max_episodes
        env = self.env
        input_size = self.input_size
        output_size = self.output_size
        ending_cond_epis = self.ending_cond_epis
        
        conti_action_flag = self.conti_action_flag
        action_map = self.action_map
        
        env.seed(seed_n)
        np.random.seed(seed_n)
        tf.set_random_seed(seed_n)
        random.seed(seed_n)
        
        mainDQN = self.mainDQN
        
        step_count_total = 0
        global_step = 0
        loss = 0
        
        steps_list = []
        step_avg_list = []
        
        tf.initialize_all_variables().run()
        mainDQN.load_network(saved_num = saved_num)
        
        print("")
        print("CASE {}".format(case_n))
        print("  Exp : {}".format(Exp))
        #print("  Strategy : Double : {}, Dueling : {}, Prioritized : {}".format(Double, Dueling, Prioritized))
        
        for episode in range(1, max_episodes+1):
            s = env.reset()
            reward_sum = 0
            while True :
                #env.render()
                action = np.argmax(mainDQN.predict(s))
                if conti_action_flag:
                    action0 = [action_map[action]]
                else:
                    action0 = action
                
                s,reward,done,_ = env.step(action0)
                reward_sum += reward
        
                if done :
                    #print("episode: {}   steps: {}".format(episode, reward_sum))
                    steps_list.append(reward_sum)
                    if episode < ending_cond_epis:
                        step_count_total += steps_list[episode - 1]
                        step_avg_list.append(step_count_total / episode)

                    if episode == ending_cond_epis:
                        step_count_total += steps_list[episode - 1]
                        step_avg_list.append(step_count_total / ending_cond_epis)
                        #print ("Step Average 100:  ", step_avg_list[episode - 1])

                    if episode > ending_cond_epis:
                        step_count_total += steps_list[episode - 1]
                        step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                        step_avg_list.append(step_count_total / ending_cond_epis)
                        #print ("Step Average 100:  ", step_avg_list[episode - 1])

                    print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
                    print ("                   ( Result : {},  Loss : {} )"
                                       .format(round(reward_sum, 5), round(loss, 8)))
                    break
        
        x_values = list(range(1, max_episodes+1))
        y_values = step_avg_list[:]
        plt.plot(x_values, y_values, c='green')
        plt.grid(True)
        plt.show()    

