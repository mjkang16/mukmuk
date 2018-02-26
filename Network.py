
# coding: utf-8

import numpy as np
import tensorflow as tf

class DQN :
    def __init__(self, session, input_size, output_size, seed_n, layer_size, learning_rate, name="main") :
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.seed_n = seed_n
        self.net_name = name
        self._build_network(h_size=layer_size, l_rate=learning_rate)
    
    def _build_network(self, h_size=64, l_rate=0.01) :
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            
            #######
            #W4 = tf.get_variable("W4", shape=[h_size, h_size],
            #                     initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
            #layer4 = tf.nn.relu(tf.matmul(layer1, W4))
            #######
            
            W2 = tf.get_variable("W2", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
            W3 = tf.get_variable("W3", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
            layer3 = tf.nn.tanh(tf.matmul(layer1, W3))
            
            W_V = tf.get_variable("W_V", shape=[h_size, 1],
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
            
            W_A = tf.get_variable("W_A", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
            
            self.Value = tf.matmul(layer2, W_V)
            self.Advantage = tf.matmul(layer3, W_A)
            
            self._Qpred = self.Value + self.Advantage - tf.reduce_mean(self.Advantage,
                                                                       reduction_indices=1,keep_dims=True)
            
        self.saver = tf.train.Saver()
            
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        
        self._WIS = tf.placeholder(shape=[1, 1], dtype=tf.float32)
        
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        #self._loss = tf.reduce_mean(tf.multiply(self._WIS, tf.square(self._Y - self._Qpred)))
        #self._loss = tf.reduce_mean(self._WIS * tf.square(self._Y - self._Qpred))
        
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)
    
    def predict(self, state):
        x = np.reshape(state, [1,self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X : x})
    
    def update(self, x_stack, y_stack, w_stack):
        return self.session.run([self._loss, self._train],
                                feed_dict={self._X : x_stack, self._Y : y_stack, self._WIS : w_stack})

    def save_network(self, episode, save_epi):
        if episode == save_epi*1:
            self.saver.save(self.session, "/tmp/model1.ckpt")
        elif episode == save_epi*2:
            self.saver.save(self.session, "/tmp/model2.ckpt")
        elif episode == save_epi*3:
            self.saver.save(self.session, "/tmp/model3.ckpt")
        elif episode == save_epi*4:
            self.saver.save(self.session, "/tmp/model4.ckpt")
        elif episode == save_epi*5:
            self.saver.save(self.session, "/tmp/model5.ckpt")
        elif episode == save_epi*6:
            self.saver.save(self.session, "/tmp/model6.ckpt")
        elif episode == save_epi*7:
            self.saver.save(self.session, "/tmp/model7.ckpt")
        elif episode == save_epi*8:
            self.saver.save(self.session, "/tmp/model8.ckpt")
        elif episode == save_epi*9:
            self.saver.save(self.session, "/tmp/model9.ckpt")
        elif episode == save_epi*10:
            self.saver.save(self.session, "/tmp/model10.ckpt")
    
    def load_network(self, saved_num):
        if saved_num == 1:
            self.saver.restore(self.session, "/tmp/model1.ckpt")
        elif saved_num == 2:
            self.saver.restore(self.session, "/tmp/model2.ckpt")
        elif saved_num == 3:
            self.saver.restore(self.session, "/tmp/model3.ckpt")
        elif saved_num == 4:
            self.saver.restore(self.session, "/tmp/model4.ckpt")
        elif saved_num == 5:
            self.saver.restore(self.session, "/tmp/model5.ckpt")
        elif saved_num == 6:
            self.saver.restore(self.session, "/tmp/model6.ckpt")
        elif saved_num == 7:
            self.saver.restore(self.session, "/tmp/model7.ckpt")
        elif saved_num == 8:
            self.saver.restore(self.session, "/tmp/model8.ckpt")
        elif saved_num == 9:
            self.saver.restore(self.session, "/tmp/model9.ckpt")
        elif saved_num == 10:
            self.saver.restore(self.session, "/tmp/model10.ckpt")
        