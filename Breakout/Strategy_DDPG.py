


import numpy as np
import tensorflow as tf
import random
from collections import deque
import gym
from Action_Network import Action_Network
from Q_Network import Q_Network
from tensorflow_grad_inverter import grad_inverter
import Train_DDPG
import matplotlib.pyplot as plt
import pickle
import time

class DDPG:
    def __init__(self, sess, dis = 0.99, REPLAY_MEMORY = 10000, batch_size = 64, max_steps = 10000000, max_episodes = 100000,
                 
                 layer_size_Q1 = 64, layer_size_Q2 = 64, learning_rate_Q = 0.01,
                 layer_size_A1 = 64, layer_size_A2 = 64, learning_rate_A = 0.01,
                 
                 tau = 0.001, action_scale = 1,
                 
                 ending_cond_epis = 100, ending_cond_reward = 195,
                 seed_n = 0, Game = 'CartPole-v0', file_name = 'steps', case_n = 0, save_epi = 100):
        
        env = gym.make(Game)
        tf.set_random_seed(seed_n)
        
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        
        self.sess = sess
        self.dis = dis
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.batch_size = batch_size
        
        self.layer_size_Q1 = layer_size_Q1
        self.layer_size_Q2 = layer_size_Q2
        self.learning_rate_Q = learning_rate_Q
        self.layer_size_A1 = layer_size_A1
        self.layer_size_A2 = layer_size_A2
        self.learning_rate_A = learning_rate_A
        
        self.tau = tau
        self.seed_n = seed_n
        self.Game = Game
        self.save_epi = save_epi
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.env = env
        self.file_name = file_name
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.Q_Network = Q_Network(sess, input_size, output_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q, batch_size)
        
        self.Action_Network = Action_Network(sess, input_size, output_size, seed_n, layer_size_A1, layer_size_A2, learning_rate_A, action_scale = action_scale)
        
        self.ending_cond_epis = ending_cond_epis
        self.ending_cond_reward = ending_cond_reward
        
        self.run_DDPG(case_n = case_n, seed_n = seed_n)

        
    def run_DDPG(self, case_n, seed_n):
        sess = self.sess
        dis = self.dis
        REPLAY_MEMORY = self.REPLAY_MEMORY
        batch_size = self.batch_size

        tau = self.tau
        Game = self.Game
        save_epi = self.save_epi
        max_episodes = self.max_episodes
        max_steps = self.max_steps
        env = self.env
        input_size = self.input_size
        output_size = self.output_size

        ending_cond_epis = self.ending_cond_epis
        ending_cond_reward = self.ending_cond_reward

        env.seed(seed_n)
        np.random.seed(seed_n)
        tf.set_random_seed(seed_n)
        random.seed(seed_n)

        Q_Network = self.Q_Network
        Action_Network = self.Action_Network
        
        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()        
        action_bounds = [action_max,action_min] 
        grad_inv = grad_inverter(sess, action_bounds)
        
        end_episode = 0
        step_count_total = 0
        global_step = 0
        loss = 0

        replay_buffer = deque()
        Q_list = []
        TD_error_list = []
        steps_list = []
        step_avg_list = []
        global_step_list = []

        print("")
        print("CASE {}".format(case_n))
        print("  Exp : DDPG")
        print("  STATE DIM : {}, ACTION DIM : {}".format(input_size, output_size))

        for episode in range(1, max_episodes+1):
            done = False
            step_count = 0
            current_step = 0
            TD_error = 0
            state = env.reset()

            while not done:
                
                action = Action_Network.evaluate_actor(np.reshape(state, [1, input_size]))[0]
                next_state, reward, done, _ = env.step(action)
                step_count += reward
                global_step += 1
                current_step += 1

                replay_buffer.append((state, next_state, action, reward, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state

                if global_step > batch_size:
                    minibatch = random.sample(replay_buffer, batch_size)
                    Train_DDPG.train(Q_Network, Action_Network, minibatch, output_size, grad_inv)

            steps_list.append(step_count)
            global_step_list.append(global_step)

            # Print the average of result 
            if episode < ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / episode)

            if episode == ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / ending_cond_epis)

            if episode > ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                step_avg_list.append(step_count_total / ending_cond_epis)

            print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
            print ("                   ( Result : {},  Steps : {},  Global Steps : {} )"
                                                           .format(round(step_count, 5), current_step, global_step))
            
            if episode % save_epi == 0:
                file_case = str(case_n)
                #with open('/home/jolp/Desktop/Data/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                #    pickle.dump(step_avg_list, fout)
                #with open('/home/jolp/Desktop/Data/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                #    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
            
            end_episode += 1
            if step_avg_list[episode - 1] > ending_cond_reward:
                break
            if global_step > max_steps:
                break

        print("--------------------------------------------------")
        print("--------------------------------------------------")
        for episode in range(end_episode + 1, max_episodes+1):
            if global_step > max_steps:
                break
            
            s = env.reset()
            reward_sum = 0
            while True :
                #env.render()
                action = Action_Network.evaluate_actor(np.reshape(state, [1, input_size]))[0]
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                global_step += 1

                if done :
                    steps_list.append(reward_sum)
                    global_step_list.append(global_step)
                    step_count_total += steps_list[episode - 1]
                    step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                    step_avg_list.append(step_count_total / ending_cond_epis)
                    print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
                    print ("                   ( Result : {} )".format(reward_sum))
                    break
                state = next_state
            
            if episode % save_epi == 0:
                file_case = str(case_n)
                #with open('/home/jolp/Desktop/Data/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                #    pickle.dump(step_avg_list, fout)
                #with open('/home/jolp/Desktop/Data/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                #    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
            
        
        file_case = str(case_n)
        with open('/home/jolp/Desktop/Data/'+self.file_name+file_case, 'wb') as fout:
            pickle.dump(step_avg_list, fout)
        with open('/home/jolp/Desktop/Data/'+self.file_name+'_global_'+file_case, 'wb') as fout2:
            pickle.dump(global_step_list, fout2)
        
        x_values = list(range(1, len(step_avg_list)+1))
        y_values = step_avg_list[:]
        
        plt.plot(x_values, y_values, c='green')
        plt.grid(True)
        plt.show()

