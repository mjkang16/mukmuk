import tensorflow as tf
from Strategy import RL

case_num = 3

# 1 : epsilon , 2 : softmax , 3 : sparsemax
Exp_list = ['epsilon', 'softmax', 'sparsemax']

#action_res_list = [5, 5, 5, 5, 5, 5]
#action_res_list = [4, 4, 4, 4, 4, 4]
action_res_list = [3, 3, 3, 3, 3, 3]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
for i in range(case_num):
    tf.reset_default_graph()
    
    with tf.Session(config=config) as sess :
        
        CartPole = RL(sess, REPLAY_MEMORY = 40000, max_episodes = 800, action_res = action_res_list,
                      
                      batch_size = 16, training_step = 1, copy_step = 4,
                      eps_div = 4, s_scale = 0.03,
                      layer_size_Q1 = 400, layer_size_Q2 = 300, learning_rate_Q = 0.0001,
                      
                      save_epi = 100, ending_cond_reward = 10000,
                      
                      Game = 'HalfCheetah-v1', file_name = 'HalfCheetah_E(3)_TEST',
                      Exp = 'epsilon', seed_n = i, Double = True, Dueling = True, Prioritized = True)
    
    sess.close()      