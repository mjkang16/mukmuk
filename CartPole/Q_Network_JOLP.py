import numpy as np
import tensorflow as tf
import math
import time

TAU = 0.001

class Q_Network:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self, sess, num_states, num_actions, size_action_batch, batch_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q, learning_rate_A, action_map, random_action, action_scale):
        
        self.sess = sess
        self.seed_n = seed_n
        tf.set_random_seed(seed_n)
        
        self.batch_size = batch_size
        self.size_action_batch = size_action_batch
        
        self.learning_rate_A = learning_rate_A
        
        self.A_batch = tf.placeholder("float32", [None, num_actions])
        self.t_A_batch = tf.placeholder("float32", [None, num_actions])
        
        #create action batch
        self.action_batch = self.create_action_batch("", size_action_batch, num_actions, action_map, random_action, action_scale)
        
        #create target action batch
        self.t_action_batch = self.create_action_batch("t_", size_action_batch, num_actions, action_scale)
        
        #critic_q_model parameters:
        self.W1_c, self.B1_c, self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c,\
        self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net("", num_states, num_actions, layer_size_Q1, layer_size_Q2)

        #create target_q_model:
        self.t_W1_c, self.t_B1_c, self.t_W2_c, self.t_W2_action_c, self.t_B2_c, self.t_W3_c, self.t_B3_c,\
        self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net("t_", num_states, num_actions, layer_size_Q1, layer_size_Q2)
        
        #create w_batch_model:
        self.w_W1_c, self.w_B1_c, self.w_W2_c, self.w_W2_action_c, self.w_B2_c, self.w_W3_c, self.w_B3_c,\
        self.q_critic_q_model, self.w_critic_q_model, self.w_critic_state_in, self.w_critic_action_in, self.w_weighted_action_in = self.create_w_batch_net("w_", num_states, num_actions, layer_size_Q1, layer_size_Q2)
        
        self.saver = tf.train.Saver()
        
        self.q_value_in=tf.placeholder("float",[None,1])
        self.w_value_in=tf.placeholder("float",[None,1])
        self.w_t_value_in=tf.placeholder("float",[None,1])
        
        self.error = self.critic_q_model-self.q_value_in
        self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2_c,2))+ 0.0001*tf.reduce_sum(tf.pow(self.B2_c,2)) 
        
        #self.cost = tf.reduce_mean(tf.square(self.critic_q_model-self.q_value_in))
        self.cost = tf.pow(self.error,2)/batch_size + self.l2_regularizer_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_Q).minimize(self.cost)
        
        self.cost_prioritized = self.w_value_in*tf.pow(self.error,2)/batch_size + tf.reduce_mean(self.w_value_in)*self.l2_regularizer_loss
        self.optimizer_prioritized = tf.train.AdamOptimizer(learning_rate=learning_rate_Q).minimize(self.cost_prioritized)

        #action train
        self.actor = [self.critic_action_in]
        self.actions_gradients = tf.gradients(self.critic_q_model, self.actor, tf.fill((size_action_batch,1),-1.))
        
        self.actor_2 = [self.action_batch]
        self.actions_optimizer =tf.train.AdamOptimizer(self.learning_rate_A).apply_gradients(zip(self.actions_gradients,self.actor_2))
        
        #weighted action train
        self.w_actor = [self.w_critic_action_in]
        self.w_actions_gradients = tf.gradients(self.w_critic_q_model, self.w_actor, tf.fill((size_action_batch,1),-1.))
        
        self.w_actor_2 = [self.action_batch]
        self.w_actions_optimizer =\
                    tf.train.AdamOptimizer(self.learning_rate_A).apply_gradients(zip(self.w_actions_gradients,[self.action_batch]))
        
        #action gradient to be used in actor network:
        self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
        self.act_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])]
        
        
        #initialize all tensor variable parameters:
        self.sess.run(tf.initialize_all_variables())
        
        self.sess.run([
            self.t_W1_c.assign(self.W1_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_W2_c.assign(self.W2_c),
            self.t_W2_action_c.assign(self.W2_action_c),
            self.t_B2_c.assign(self.B2_c),
            self.t_W3_c.assign(self.W3_c),
            self.t_B3_c.assign(self.B3_c),
            self.t_action_batch.assign(self.action_batch)
        ])
        
        self.update_target_critic_op = [
            self.t_W1_c.assign(TAU*self.W1_c+(1-TAU)*self.t_W1_c),
            self.t_B1_c.assign(TAU*self.B1_c+(1-TAU)*self.t_B1_c),
            self.t_W2_c.assign(TAU*self.W2_c+(1-TAU)*self.t_W2_c),
            self.t_W2_action_c.assign(TAU*self.W2_action_c+(1-TAU)*self.t_W2_action_c),
            self.t_B2_c.assign(TAU*self.B2_c+(1-TAU)*self.t_B2_c),
            self.t_W3_c.assign(TAU*self.W3_c+(1-TAU)*self.t_W3_c),
            self.t_B3_c.assign(TAU*self.B3_c+(1-TAU)*self.t_B3_c)
        ]
        
        self.update_action_target_critic_op = [
            self.t_action_batch.assign(TAU*self.action_batch+(1-TAU)*self.t_action_batch)
        ]
        
        self.update_w_batch_2_critic_op = [
            self.w_W1_c.assign(self.W1_c),
            self.w_B1_c.assign(self.B1_c),
            self.w_W2_c.assign(self.W2_c),
            self.w_W2_action_c.assign(self.W2_action_c),
            self.w_B2_c.assign(self.B2_c),
            self.w_W3_c.assign(self.W3_c),
            self.w_B3_c.assign(self.B3_c)
        ]
        
        self.update_w_batch_2_target_op = [
            self.w_W1_c.assign(self.t_W1_c),
            self.w_B1_c.assign(self.t_B1_c),
            self.w_W2_c.assign(self.t_W2_c),
            self.w_W2_action_c.assign(self.t_W2_action_c),
            self.w_B2_c.assign(self.t_B2_c),
            self.w_W3_c.assign(self.t_W3_c),
            self.w_B3_c.assign(self.t_B3_c)
        ]
        
        self.update_action_batch_op = [
            self.action_batch.assign(self.A_batch),
            self.t_action_batch.assign(self.t_A_batch)
        ]
        

    def create_critic_net(self, name, num_states=4, num_actions=1, N_HIDDEN_1=400, N_HIDDEN_2=300):
        critic_state_in = tf.placeholder("float",[None,num_states])
        critic_action_in = tf.placeholder("float",[None,num_actions])
        
        W1_c = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)), name=name+"W1_c")
        B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)), name=name+"B1_c")
        W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)), name=name+"W2_c")
        W2_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)), name=name+"W2_action_c")
        
        B2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)), name=name+"B2_c") 
        W3_c= tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003), name=name+"W3_c")
        B3_c= tf.Variable(tf.random_uniform([1],-0.003,0.003), name=name+"B3_c")
        
        H1_c=tf.nn.softplus(tf.matmul(critic_state_in,W1_c)+B1_c)
        H2_c=tf.nn.tanh(tf.matmul(H1_c,W2_c)+tf.matmul(critic_action_in,W2_action_c)+B2_c)
        
        critic_q_model=tf.matmul(H2_c,W3_c)+B3_c
        
        return W1_c, B1_c, W2_c, W2_action_c, B2_c, W3_c, B3_c, critic_q_model, critic_state_in, critic_action_in
    
    def create_w_batch_net(self, name, num_states=4, num_actions=1, N_HIDDEN_1=400, N_HIDDEN_2=300):
        critic_state_in = tf.placeholder("float",[None,num_states])
        critic_action_in = tf.placeholder("float",[None,num_actions])
        weighted_action_in = tf.placeholder("float",[self.batch_size,self.size_action_batch])
        
        W1_c = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)), name=name+"W1_c")
        B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)), name=name+"B1_c")
        W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)), name=name+"W2_c")
        W2_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)), name=name+"W2_action_c")
        
        B2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)), name=name+"B2_c") 
        W3_c= tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003), name=name+"W3_c")
        B3_c= tf.Variable(tf.random_uniform([1],-0.003,0.003), name=name+"B3_c")
    
        H1_c=tf.nn.softplus(tf.matmul(critic_state_in,W1_c)+B1_c)
        
        A = tf.matmul(critic_action_in, W2_action_c)
        A = tf.tile(A, [self.batch_size, 1])
        
        B = tf.matmul(H1_c,W2_c)
        B = tf.tile(B, [1, self.size_action_batch])
        B = tf.reshape(B, [-1, N_HIDDEN_2])
        
        H2_c = tf.nn.tanh(A + B + B2_c)
        
        H3_c = tf.matmul(H2_c,W3_c)+B3_c
        critic_q_model = tf.reshape(H3_c, [self.batch_size, -1])
        critic_q_model = critic_q_model * weighted_action_in
        
        critic_q_model = tf.reduce_sum(critic_q_model, axis = 0)
        critic_q_model = tf.reshape(critic_q_model, [-1, 1])
        
        return W1_c, B1_c, W2_c, W2_action_c, B2_c, W3_c, B3_c, H3_c, critic_q_model, critic_state_in, critic_action_in, weighted_action_in
    
    
    def create_action_batch(self, name, size_action_batch=4, num_actions=1, action_map=None, random_action=True, action_scale=1):
        if not random_action:
            action_batch = tf.Variable(action_map, dtype=tf.float32, name = name+"action_batch")
        else:
            action_batch = tf.Variable(tf.random_uniform([size_action_batch, num_actions], minval = -action_scale, maxval = action_scale), name = name+"action_batch")
        
        return action_batch
    
    def realign_action_batch(self, A_batch, t_A_batch):
        self.sess.run(self.update_action_batch_op, feed_dict={self.A_batch : A_batch, self.t_A_batch : t_A_batch})
    
    
    def train_critic(self, state_t_batch, action_batch, y_i_batch):
        return self.sess.run([self.cost, self.optimizer],
                 feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in:action_batch, self.q_value_in: y_i_batch})
             
    def train_critic_prioritized(self, state_t_batch, action_batch, y_i_batch, w_batch):
        return self.sess.run([self.error, self.cost_prioritized, self.optimizer_prioritized],
                     feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in:action_batch, self.q_value_in: y_i_batch, self.w_value_in: w_batch})
    
    
    def evaluate_critic(self,state_t_1,action_t_1):
        return self.sess.run(self.critic_q_model, feed_dict={self.critic_state_in: state_t_1, self.critic_action_in: action_t_1})    
    
    def evaluate_target_critic(self,state_t_1,action_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1})    
    
    
    def get_q_batch(self, state_t):
        action_t_1 = self.sess.run(self.action_batch)
        q_batch = self.sess.run(self.critic_q_model,\
                             feed_dict={self.critic_state_in: state_t, self.critic_action_in: action_t_1})
        return q_batch
    
    def get_target_q_batch(self, state_t_1):
        self.sess.run(self.update_w_batch_2_target_op)
        
        action_t_1 = self.sess.run(self.t_action_batch)
        t_q_batch = self.sess.run(self.q_critic_q_model,\
                             feed_dict={self.w_critic_state_in: state_t_1, self.w_critic_action_in: action_t_1})
        return t_q_batch
    
    
    def get_action_batch(self):
        return self.sess.run(self.action_batch)
    
    def get_target_action_batch(self):
        return self.sess.run(self.t_action_batch)
    
    def train_actor(self, state_t):
        action_t = self.sess.run(self.action_batch)
        self.sess.run(self.actions_optimizer, feed_dict={self.critic_state_in: state_t, self.critic_action_in: action_t})
        
    def train_weighted_actor(self, state_t, weighted_t):
        self.sess.run(self.update_w_batch_2_critic_op)
        action_t = self.sess.run(self.action_batch)
        self.sess.run(self.w_actions_optimizer,\
              feed_dict={self.w_critic_state_in: state_t, self.w_critic_action_in: action_t, self.w_weighted_action_in: weighted_t})
    
    def compute_delQ_a(self, state_t, action_t):
        return self.sess.run(self.act_gradients, feed_dict={self.critic_state_in: state_t, self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)
        
    def update_action_target_critic(self):
        self.sess.run(self.update_action_target_critic_op)
    
    
    def save_network(self, game_name, episode, save_epi):
        ep = str(episode)
        self.saver.save(self.sess, "/home/minjae/Reinforcement\Learning/Network/model"+ep+"_"+game_name+".ckpt")
        
    def load_network(self, game_name, saved_num):
        ep = str(saved_num)
        self.saver.restore(self.sess, "/home/minjae/Desktop/Network/model"+ep+"_"+game_name+".ckpt")
