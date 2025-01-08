class algo_base:
    def __init__(self, **kwargs):
        self._training_kwargs = kwargs

    @property
    def training_kwargs(self):
        return self._training_kwargs
    
    def _update_kwargs(self, kwargs):
        for k, v in kwargs.items():
            self._training_kwargs[k] = v

class PPO(algo_base):
    @property
    def algo(self):
        return "PPO"

class IMPALA(algo_base):
    @property
    def algo(self):
        return "IMPALA"

class APPO(algo_base):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'grad_clip': 30.0,
        }

        self._update_kwargs(kwargs)

    @property
    def algo(self):
        return "APPO"

class DQN(algo_base):

    @property
    def algo(self):
        return "DQN"
    
class Rainbow_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            "target_network_update_freq": 500,
            'replay_buffer_config': {
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            },
            # "replay_buffer_config": {
            #     "_enable_replay_buffer_api": False,
            #     "type": "ReplayBuffer",
            #     "type": "PrioritizedReplayBuffer",
            #     "capacity": 50000,
            #     "prioritized_replay_alpha": 0.6,
            #     "prioritized_replay_beta": 0.4,
            #     "prioritized_replay_eps": 1e-6,
            #     "replay_sequence_length": 1
            # },
            "epsilon": [[0, 1.0], [1000000, 0.1]],
            "adam_epsilon": 1e-8,
            "grad_clip": 40.0,
            "num_steps_sampled_before_learning_starts": 10000,
            "tau": 1,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "noisy": True,
            "sigma0": 0.5,
            "dueling": True,
            "hiddens": [512],
            "double_q": True,
            "n_step": 3,
        }

        self._update_kwargs(kwargs)

class Double_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'double_q': True,
        }

        self._update_kwargs(kwargs)

class Dueling_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'dueling': True,
        }

        self._update_kwargs(kwargs)

class DQN_PER(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'replay_buffer_config': {
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            },
        }

        self._update_kwargs(kwargs)

class Noisy_DQN(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'noisy': True,
        }
        
        self._update_kwargs(kwargs)

class DQN_C51(DQN):
    def __init__(self, **kwargs):
        self._training_kwargs = {
            'num_atoms': 51,
            'v_min': -10.0,
            'v_max': 10.0,
        }
        
        self._update_kwargs(kwargs)

def simplify_rllib_metrics(data, out_func=print):
    important_metrics = {
        "环境运行器": {},
        "评估": {},
        "学习者": {},
    }

    if 'counters' in data:
        if 'num_env_steps_sampled' in data["counters"]:
            important_metrics["环境运行器"]["采样环境总步数"] = data["counters"]["num_env_steps_sampled"]

    if 'env_runners' in data:
        if 'episode_return_mean' in data["env_runners"]:
            important_metrics["环境运行器"]["episode平均回报"] = data["env_runners"]["episode_return_mean"]
        if 'episode_return_max' in data["env_runners"]:
            important_metrics["环境运行器"]["episode最大回报"] = data["env_runners"]["episode_return_max"]
        if 'episode_len_mean' in data["env_runners"]:
            important_metrics["环境运行器"]["episode平均步数"] = data["env_runners"]["episode_len_mean"]
        if 'episode_len_max' in data["env_runners"]:
            important_metrics["环境运行器"]["episode最大步数"] = data["env_runners"]["episode_len_max"]
        if 'num_env_steps_sampled' in data["env_runners"]:
            important_metrics["环境运行器"]["采样环境总步数"] = data["env_runners"]["num_env_steps_sampled"]
        if 'num_episodes' in data["env_runners"]:
            important_metrics["环境运行器"]["episodes计数"] = data["env_runners"]["num_episodes"]

    if 'evaluation' in data:
        if 'env_runners' in data["evaluation"]:
            if 'episode_return_mean' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode平均回报"] = data["evaluation"]["env_runners"]["episode_return_mean"]
            if 'episode_return_max' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode最大回报"] = data["evaluation"]["env_runners"]["episode_return_max"]
            if 'episode_len_mean' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode平均步数"] = data["evaluation"]["env_runners"]["episode_len_mean"]
            if 'episode_len_max' in data["evaluation"]["env_runners"]:
                important_metrics["评估"]["episode最大步数"] = data["evaluation"]["env_runners"]["episode_len_max"]

    if 'learners' in data:
        if 'default_policy' in data["learners"]:
            important_metrics["学习者"]["默认策略"] = {}
            if 'entropy' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["熵"] = data["learners"]["default_policy"]["entropy"]
            if 'policy_loss' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["策略损失"] = data["learners"]["default_policy"]["policy_loss"]
            if 'vf_loss' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["值函数损失"] = data["learners"]["default_policy"]["vf_loss"]
            if 'total_loss' in data["learners"]["default_policy"]:
                important_metrics["学习者"]["默认策略"]["总损失"] = data["learners"]["default_policy"]["total_loss"]

    if 'time_this_iter_s' in data:
        important_metrics["本轮时间"] = data["time_this_iter_s"]
    if 'num_training_step_calls_per_iteration' in data:
        important_metrics["每轮训练步数"] = data["num_training_step_calls_per_iteration"]
    if 'training_iteration' in data:
        important_metrics["训练迭代次数"] = data["training_iteration"]
            
    out_func(f"--------- 训练迭代: {important_metrics['训练迭代次数']} ---------")
    out_func("环境运行器:")
    if important_metrics['环境运行器']:
        for k, v in important_metrics['环境运行器'].items():
            out_func(f"  {k}: {v:.4f}")
    else:
        out_func("  无环境运行器数据")
    
    out_func("\n评估:")
    for k, v in important_metrics['评估'].items():
        out_func(f"  {k}: {v:.4f}")
    else:
        out_func("  无评估数据")

    out_func("\n学习者(默认策略):")
    if '默认策略' in important_metrics['学习者'] and important_metrics['学习者']['默认策略']:
        for k, v in important_metrics['学习者']['默认策略'].items():
            out_func(f"  {k}: {v:.4f}")
    else:
        out_func("  无学习者数据")
    
    if '本轮时间' in important_metrics:
        out_func(f"\n本轮时间: {important_metrics['本轮时间']:.4f}")
    if '每轮训练步数' in important_metrics:
        out_func(f"每轮训练步数: {important_metrics['每轮训练步数']}")
    out_func('-'*30)
