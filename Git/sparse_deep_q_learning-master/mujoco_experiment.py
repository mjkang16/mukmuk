import gym
import policy
import maxapproxi
import network
import replaymemory
import numpy as np
import tensorflow as tf

class Experiments:
    def __init__(self, seed=0, env_name = 'CartPole-v1', action_res=None, dqn_hidden_spec=None, batch_size = 128, learning_rate=1e-3,
                discount = 0.99, max_epi = 500, max_train = 50, max_step = 1000, target_update_period = 5,
                replay_memory_size = 10000, eps_decay_rate=0.999, scale=1.,
                strategy="Epsilon", backuprule="Bellman"):
        # Fix the numpy random seed
        rng = np.random.RandomState(seed)
        
        # Gen environment
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
        
        # Get environment information
        observ_dim = env.observation_space.high.shape[0]
        n_action, conti_action_flag, action_map = self.get_action_information(env, env_name, action_res=action_res)

        # Set network spec
        if dqn_hidden_spec is None:
            dqn_hidden_spec = [
                {'dim': 512,'activation': tf.nn.tanh},
                {'dim': 512,'activation': tf.nn.tanh}
            ]

        # Initialize Tensorflow Graph
        tf.reset_default_graph()
        
        # Gen value network
        value_func = network.Network(input_dim=observ_dim,output_dim=n_action,hidden_spec=dqn_hidden_spec,learning_rate=learning_rate,seed=seed)

        # Set session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        
        # Initialize tf variable and old variable of value network
        tf.set_random_seed(seed)
        session.run(tf.global_variables_initializer())
        session.run(value_func.update_ops)
        
        # Gen replay memory
        replay_memory = replaymemory.ReplayMemory(rng=rng,memory_size=replay_memory_size)
        
        # Gen policy function
        policy_func = policy.Policy(n_action,rng=rng,strategy=strategy,eps_decay_rate=eps_decay_rate,scale=scale)

        # Store All Variable to Class
        self.seed=seed
        self.env_name=env_name
        self.action_res=action_res
        self.dqn_hidden_spec=dqn_hidden_spec
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.discount=discount
        self.max_epi=max_epi
        self.max_step=max_step
        self.max_train = max_train
        self.target_update_period=target_update_period
        self.replay_memory_size=replay_memory_size
        self.eps_decay_rate=eps_decay_rate
        self.strategy=strategy
        self.backuprule=backuprule
        
        self.observ_dim=observ_dim
        self.n_action=n_action
        self.conti_action_flag=conti_action_flag
        self.action_map=action_map
        
        self.env=env
        self.eval_env=eval_env
        self.value_func=value_func
        self.replay_memory=replay_memory
        self.policy_func=policy_func
        self.session = session
        self.config = config
        self.scale = scale
        
        
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
    
    def format_experience(self,experience):
        states_b, actions_b, rewards_b, states_n_b, done_b = zip(*experience)
        states_b = np.array(states_b)
        actions_b = np.array(actions_b)
        rewards_b = np.array(rewards_b)
        states_n_b = np.array(states_n_b)
        done_b = np.array(done_b).astype(int)
        return states_b, actions_b, rewards_b, states_n_b, done_b

    def run(self, display_period=10):
        env = self.env
        eval_env = self.eval_env
        
        max_epi = self.max_epi
        max_step = self.max_step
        max_train = self.max_train
        value_func = self.value_func
        replay_memory = self.replay_memory
        policy_func = self.policy_func
        session = self.session
        conti_action_flag = self.conti_action_flag
        action_map = self.action_map
        target_update_period=self.target_update_period
        discount=self.discount
        n_action=self.n_action
        backuprule=self.backuprule
        
        rollout_return_list = np.zeros((max_epi,))
        eval_return_list = np.zeros((max_epi,))
        max_return = -np.inf

        env.seed(self.seed)
        eval_env.seed(self.seed)
        
        epi = 0
        global_step = 0
        for epi in range(max_epi):
            #Rollout Phase
            obs = env.reset()            
            done = False
            policy_func.explore = True
            total_reward = 0
            for step in range(max_step):
                
                if done:
                    obs = env.reset()
                    done = False
                    
                fetches, feeds = value_func.get_predictions([obs])
                q_value, = session.run(fetches=fetches,feed_dict=feeds)
                q_value = q_value[0]

                action = policy_func.get_action(q_value)
                if conti_action_flag:
                    action_val = action_map[action]
                else:
                    action_val = action

                next_obs, reward, done, info = env.step(action_val)
                total_reward += reward
                replay_memory.save_experience(obs, action, reward, next_obs, done)
                obs = next_obs
                policy_func.update_policy()
                
            rollout_return_list[epi] = total_reward
            if (epi%target_update_period)==0:
                session.run(value_func.update_ops)
            
            total_v_loss = 0
            batch_size = self.batch_size
            if replay_memory.memory.n_entries > batch_size:
                for step_train in range(max_train):
                    replay_memory.anneal_per_importance_sampling(step_train,max_train)
                    
                    idx, priorities, w, experience = replay_memory.retrieve_experience(batch_size)

                    states_b, actions_b, rewards_b, states_n_b, done_b = self.format_experience(experience)

                    fetches, feeds = value_func.get_predictions(states_n_b)
                    q_n_b, = session.run(fetches=fetches,feed_dict=feeds)

                    fetches, feeds = value_func.get_predictions_old(states_n_b)
                    q_n_target_b, = session.run(fetches=fetches,feed_dict=feeds)

                    best_a = np.argmax(q_n_b, axis=1)
                    if backuprule == 'Bellman':
                        targets_b = rewards_b + (1. - done_b) * discount * q_n_target_b[np.arange(batch_size), best_a]
                    elif backuprule == 'SoftBellman':
                        targets_b = rewards_b + (1. - done_b) * discount * maxapproxi.logsumexp(q_n_target_b, scale=self.scale)
                    elif self.backuprule == 'SparseBellman':
                        targets_b = rewards_b + (1. - done_b) * discount * maxapproxi.sparsemax(q_n_target_b, scale=self.scale)

                    fetches, feeds = value_func.get_predictions(states_b)
                    targets, = session.run(fetches=fetches,feed_dict=feeds)
                    for j, action in enumerate(actions_b):
                        targets[j, action] = targets_b[j]

                    fetches, feeds = value_func.get_train(states_b,targets, np.transpose(np.tile(w, (n_action, 1))))
                    v_loss, errors, _ = session.run(fetches=fetches,feed_dict=feeds)
                    errors = errors[np.arange(len(errors)), actions_b]

                    replay_memory.update_experience_weight(idx, errors)
                    total_v_loss += v_loss                    
            
            total_reward = 0
            obs = eval_env.reset()      
            done = False
            policy_func.explore = False
            while not done:
                fetches, feeds = value_func.get_predictions([obs])
                q_value, = session.run(fetches=fetches,feed_dict=feeds)
                q_value = q_value[0]

                action = policy_func.get_action(q_value)
                if conti_action_flag:
                    action_val = action_map[action]
                else:
                    action_val = action

                next_obs, reward, done, _ = eval_env.step(action_val)
                total_reward += reward
                obs = next_obs
            eval_return_list[epi] = total_reward
            
            if epi < 99:
                rollout_return = np.mean(rollout_return_list[:epi+1])
                eval_return = np.mean(eval_return_list[:epi+1])
            else:
                rollout_return = np.mean(rollout_return_list[epi-99:epi])
                eval_return = np.mean(eval_return_list[epi-99:epi])
            
            if epi >= 99 and max_return < rollout_return:
                max_return = rollout_return
                
            if ((epi+1)%display_period)==0:
                print('[{}/{}] Rollout Return {}, Eval Return {}, Max Return {}, DQN Loss {}, Epsilon {}'.format(epi+1,max_epi,rollout_return, eval_return,max_return,total_v_loss,policy_func.eps))
        return rollout_return_list, eval_return_list
    
    def evaluation(self,max_eval_epi=100):
        env = self.env
        
        max_step = self.max_step
        value_func = self.value_func
        policy_func = self.policy_func
        session = self.session
        conti_action_flag = self.conti_action_flag
        action_map = self.action_map
        n_action=self.n_action
        
        return_list = np.zeros((max_eval_epi,))
        
        policy_func.explore = False
        env.seed(self.seed)
        for epi in range(max_eval_epi):
            obs = env.reset()

            total_reward = 0
            done = False
            while not done:
                fetches, feeds = value_func.get_predictions([obs])
                q_value, = session.run(fetches=fetches,feed_dict=feeds)
                q_value = q_value[0]

                action = policy_func.get_action(q_value)
                if conti_action_flag:
                    action_val = action_map[action]
                else:
                    action_val = action

                next_obs, reward, done, _ = env.step(action_val)
                total_reward += reward
                obs = next_obs
            return_list[epi] = total_reward
        print("Evaluation Result: {}".format(np.mean(return_list)))
        return return_list
        
        
        