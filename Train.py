
# coding: utf-8

import numpy as np
import tensorflow as tf
#from Network import DQN
import Exploration

def replay_train (mainDQN, targetDQN, train_batch, w_batch, Exp, dis, input_size, output_size) :
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)
    w_stack = np.empty(0).reshape(0, 0)
    
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        
        if done :
            Q[0,action] = reward
        else :
            if Exp == 'epsilon':
                action0 = np.argmax(mainDQN.predict(next_state))
                Q[0,action] = reward + dis * (targetDQN.predict(next_state)[0, action0])
            else:
                Q[0,action] = Exploration.Q_value(Exp, reward, dis, targetDQN.predict(next_state)[0])
        
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
        
    for w in w_batch:
        w_stack = np.vstack([w])
        
    return mainDQN.update(x_stack, y_stack, w_stack)

def replay_train_unDouble (mainDQN, train_batch, w_batch, Exp, dis, input_size, output_size) :
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)
    w_stack = np.empty(0).reshape(0, 0)
    
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        
        if done :
            Q[0,action] = reward
        else :
            if Exp == 'epsilon':
                Q[0,action] = reward + dis * np.max(mainDQN.predict(next_state))
            else:
                Q[0,action] = Exploration.Q_value(Exp, reward, dis, mainDQN.predict(next_state)[0])
            
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
        
    for w in w_batch:
        w_stack = np.vstack([w])
        
    return mainDQN.update(x_stack, y_stack, w_stack)

def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    
    return op_holder

