

import numpy as np
import tensorflow as tf

GAMMA = 0.99
  
def train(Q_Network, Action_Network, train_batch, num_actions, grad_inv):
    state_t_batch = [item[0] for item in train_batch]
    state_t_batch = np.array(state_t_batch)
    #state t+1        
    state_t_1_batch = [item[1] for item in train_batch]
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = [item[2] for item in train_batch]
    action_batch = np.array(action_batch)
    action_batch = np.reshape(action_batch, [len(action_batch), num_actions])
    reward_batch = [item[3] for item in train_batch]
    reward_batch = np.array(reward_batch)
    done_batch = [item[4] for item in train_batch]
    done_batch = np.array(done_batch) 
    
    action_t_1_batch = Action_Network.evaluate_target_actor(state_t_1_batch)
    #Q'(s_i+1,a_i+1)        
    q_t_1 = Q_Network.evaluate_target_critic(state_t_1_batch, action_t_1_batch) 
    #y_i_batch=[]         
    #for i in range(0, len(train_batch)):
    #    y_i_batch.append(reward_batch[i] + GAMMA*q_t_1[i][0])                 
    
    q_t_1 = np.reshape(q_t_1, [1,-1])
    #y_i_batch = reward_batch + GAMMA*q_t_1*(1-done_batch)
    y_i_batch = reward_batch + GAMMA*q_t_1
    
    #y_i_batch = np.array(y_i_batch)
    y_i_batch = np.reshape(y_i_batch,[-1,1])

    # Update critic by minimizing the loss
    Q_Network.train_critic(state_t_batch, action_batch, y_i_batch)

    # Update actor proportional to the gradients:
    action_for_delQ = Action_Network.evaluate_actor(state_t_batch)

    del_Q_a = Q_Network.compute_delQ_a(state_t_batch, action_for_delQ)
    del_Q_a = grad_inv.invert(del_Q_a, action_for_delQ)
    
    #del_Q_a = Q_Network.compute_delQ_a(self.state_t_batch,action_for_delQ)[0]
    
    # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
    Action_Network.train_actor(state_t_batch, del_Q_a)

    # Update target Critic and actor network
    Q_Network.update_target_critic()
    Action_Network.update_target_actor()
    