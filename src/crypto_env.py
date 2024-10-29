import os
import yaml
import pickle
import numpy as np
from yaml import dump, Dumper

#region init

class CryptoEnv:
    def __init__(self, train_configs=None, train_log_dir=None, mode='train'):
        self.mode = mode
        self.train_configs = train_configs
        self.train_log_dir = train_log_dir
        self.configs = self.get_configs()
        self.train_symbols, self.val_symbols = self.get_symbols()
        self.data = self.load_data()
        self.validate_action_dim()
        self.rng = np.random.default_rng(self.train_configs['jax_seed'] if self.train_configs else 0)
        self.symbol_probabilities = self.calculate_symbol_probabilities()
        self.interval_timedelta = self.calculate_interval_timedelta()
    

    def get_configs(self):
        if self.train_configs['load_run'] is None:
            configs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs.yaml"))
        else:
            configs_path = os.path.join(self.train_log_dir, "configs.yaml")
        with open(configs_path, 'r') as configs_file:
            configs = yaml.safe_load(configs_file)
        return configs


    def save_configs(self):
        save_dir = self.train_log_dir if self.train_log_dir else os.getcwd()
        config_file_path = os.path.join(save_dir, "configs.yaml")
        with open(config_file_path, 'w') as config_file:
            dump(self.configs, config_file, Dumper=Dumper, default_flow_style=False, sort_keys=False)

    
    def save_env_states(self):
        with open(f'{self.train_log_dir}/env_states.pkl', 'wb') as f:
            pickle.dump(self.env_states, f)


    def validate_action_dim(self):
        calculated_action_dim = int(2 / self.configs['position_step_size'] + 1)
        
        if self.train_configs is None:
            self.action_dim = calculated_action_dim
        else:
            action_dim = self.train_configs['parameters'][self.train_configs['architecture']]['action_dim']
            if calculated_action_dim != action_dim:
                raise ValueError(f"Mismatch in action dimensions. Calculated: {calculated_action_dim}, Provided: {action_dim}")
            self.action_dim = action_dim


    def get_symbols(self):
        data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "env_data"))
        shortest_interval = self.configs['intervals'][-1]
        interval_dir = os.path.join(data_directory, shortest_interval)
        available_symbols = [f.split('.')[0] for f in os.listdir(interval_dir) if f.endswith('.npz')]
        filtered_symbols = []

        for symbol in available_symbols:
            file_path = os.path.join(interval_dir, f"{symbol}.npz")
            with np.load(file_path) as npz_data:
                first_timestamp = npz_data['timestamps'][0]
                if self.configs['earliest_launch_time'] <= first_timestamp <= self.configs['last_launch_time']:
                    filtered_symbols.append(symbol)

        train_symbols = filtered_symbols.copy()
        val_symbols = filtered_symbols.copy()

        if self.configs.get('train_symbols'):
            train_symbols = [s for s in self.configs['train_symbols'] if s in filtered_symbols]
        if self.configs.get('val_symbols'):
            val_symbols = [s for s in self.configs['val_symbols'] if s in filtered_symbols]

        if self.configs.get('exclude_train'):
            train_symbols = [s for s in train_symbols if s not in self.configs['exclude_train']]
        if self.configs.get('exclude_val'):
            val_symbols = [s for s in val_symbols if s not in self.configs['exclude_val']]
        if self.configs.get('exclude_both'):
            train_symbols = [s for s in train_symbols if s not in self.configs['exclude_both']]
            val_symbols = [s for s in val_symbols if s not in self.configs['exclude_both']]
        
        return train_symbols, val_symbols

#endregion
#region loading

    def load_data(self):
        data = {}
        data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "env_data"))
        selected_symbols = list(set(self.train_symbols + self.val_symbols))

        for interval in self.configs['intervals']:
            interval_dir = os.path.join(data_directory, interval)
            data[interval] = {}
            for symbol in selected_symbols:
                file_path = os.path.join(interval_dir, f"{symbol}.npz")
                with np.load(file_path) as npz_data:
                    data[interval][symbol] = {
                        'klines_processed': npz_data['klines_processed'],
                        'close': npz_data['close'],
                        'volume': npz_data['volume'],
                        'timestamps': npz_data['timestamps']}
        
        return data


    def load_klines(self, symbol, timestamp):
        klines = []
        for interval in self.configs['intervals']:
            interval_data = self.data[interval][symbol]
            
            if interval in self.configs['intervals'][:]:
                end_index = max(0, np.searchsorted(interval_data['timestamps'], timestamp + 1000) - 1)
            else:
                end_index = max(0, np.searchsorted(interval_data['timestamps'], timestamp + 1000) - 0)  # includes next future kline
            start_index = max(0, end_index - self.configs['sequence_length'])

            interval_klines = interval_data['klines_processed'][start_index:end_index]

            if (end_index - start_index) < self.configs['sequence_length']:
                pad_length = self.configs['sequence_length'] - (end_index - start_index)
                interval_klines = np.pad(interval_klines, ((pad_length, 0), (0, 0)), mode='constant')

            klines.append(interval_klines)

        if self.configs['data_version'] == 'parallel':
            result_klines = np.hstack(klines)
        elif self.configs['data_version'] == 'sequential':
            result_klines = np.vstack(klines)
        return result_klines

#endregion
#region reset

    def reset(self):
        self.env_states = []
        if self.mode == 'train':
            if self.train_configs['load_run'] is None:
                env_types = ['train'] * self.train_configs['train_envs']
                env_types += ['train_val'] * (self.train_configs['train_val_envs'] // 2)
                env_types += ['val'] * (self.train_configs['val_envs'] // 4)
                env_types += ['train_val'] * (self.train_configs['train_val_envs'] // 2)
                env_types += ['val'] * (self.train_configs['val_envs'] // 4)
                for env_type in env_types:
                    self.env_states.append(EnvState(env_type=env_type))
            else:
                with open(f'{self.train_log_dir}/env_states.pkl', 'rb') as f:
                    self.env_states = pickle.load(f)
        elif self.mode == 'eval':
            env_types = ['val'] * len(self.val_symbols)
            for env_type in env_types:
                self.env_states.append(EnvState(env_type=env_type))

        observations, infos = [], []
        for i, state in enumerate(self.env_states):
            obs, info = self.reset_(i, first=True)
            observations.append(obs)
            infos.append(info)

        return np.array(observations), infos


    def reset_(self, env_index, first=False):
        state = self.env_states[env_index]

        if self.train_configs['load_run'] is None or not first:

            episode_length_range = self.configs[f'{state.env_type}_episode_length']
            if self.train_configs['load_run'] is None and first:
                state.episode_length = self.rng.integers(episode_length_range[0] * 0.5, episode_length_range[0] * 1.5 + 1)
            else:
                state.episode_length = self.rng.integers(episode_length_range[0], episode_length_range[1] + 1)

            state.symbol = self.choose_symbol(state, env_index)
            state.timestamp = self.choose_start_timestamp(state)
            state.step = 0
            state.remaining_balance = self.configs['initial_balance']
            state.reset_history(self.configs['sequence_length'])

        klines = self.load_klines(state.symbol, state.timestamp)
        observation = self.create_observation(klines, state.history)
        info = {
            'symbol': state.symbol, 
            'timestamp': state.timestamp,
            'action_length': state.action_length,
            'history': state.history
        }
        return observation, info

#endregion
#region step

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for i, (action, state) in enumerate(zip(actions, self.env_states)):
            obs, reward, terminated, truncated, info = self.step_(i, action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return np.array(observations), np.array(rewards), np.array(terminateds), np.array(truncateds), infos


    def step_(self, env_index, action):
        state = self.env_states[env_index]
        converted_action = self.convert_action(action)
        state.step += 1
        state.timestamp += self.interval_timedelta

        train_reward, val_reward, terminated, truncated = self.execute_trade_and_update_state(state, converted_action)
        
        if state.env_type == 'train':
            reward = train_reward
        elif state.env_type == 'train_val' or state.env_type == 'val':
            reward = val_reward
        
        if self.configs['include_history']:
            if self.configs['data_version'] == 'parallel':
                state.action_length += 1
            elif self.configs['data_version'] == 'sequential':
                state.action_length = 1
            state.cumulative_reward += train_reward
            scaled_reward = self.scale_cumulative_reward(state.cumulative_reward)
            state.history['action_history'] = self.update_history(state.history['action_history'], converted_action)
            state.history['cumulative_reward_history'] = self.update_history(state.history['cumulative_reward_history'], scaled_reward)

        if terminated or truncated:
            observation, reset_info = self.reset_(env_index)
            info = reset_info
        else:
            klines = self.load_klines(state.symbol, state.timestamp)
            observation = self.create_observation(klines, state.history)
            
            info = {
                'symbol': state.symbol,
                'timestamp': state.timestamp,
                'action_length': state.action_length,
                'history': state.history
            }
        
        return observation, reward, terminated, truncated, info

#endregion
#region trade

    def execute_trade_and_update_state(self, state, new_action):
        if new_action < 0:
            old_total_value, new_total_value = self.execute_short_trade(state, new_action)
        elif new_action > 0:
            old_total_value, new_total_value = self.execute_long_trade(state, new_action)
        elif new_action == 0:
            if state.action < 0:
                old_total_value, new_total_value = self.close_short_trade(state)
            elif state.action > 0:
                old_total_value, new_total_value = self.close_long_trade(state)
            elif state.action == 0:
                old_total_value = state.remaining_balance
                new_total_value = state.remaining_balance

        return self.finalize_trade(state, old_total_value, new_total_value)


    def execute_long_trade(self, state, new_action):
        if state.action < 0:
            old_total_value, new_total_value = self.close_short_trade(state)
        else:
            old_total_value = state.remaining_balance + state.position_value
        
        if new_action != state.action:
            target_position_value = old_total_value * new_action
            position_value_change = target_position_value - state.position_value
            
            fee = abs(position_value_change) * self.configs['fee']
            state.remaining_balance -= position_value_change + fee
            state.position_value = target_position_value
            state.action = new_action
        
        current_price = self.get_price(state.symbol, state.timestamp)
        previous_price = self.get_price(state.symbol, state.timestamp - self.interval_timedelta)
        price_ratio = current_price / previous_price
        state.position_value *= price_ratio
        
        new_total_value = state.remaining_balance + state.position_value
        return old_total_value, new_total_value


    def execute_short_trade(self, state, new_action):
        if state.action > 0:
            old_total_value, new_total_value = self.close_long_trade(state)
        else:
            old_total_value = state.remaining_balance + state.position_value
        
        if new_action != state.action:
            target_position_value = old_total_value * new_action
            position_value_change = target_position_value - state.position_value
            
            fee = abs(position_value_change) * self.configs['fee']
            state.remaining_balance -= position_value_change + fee
            state.position_value = target_position_value
            state.action = new_action
        
        current_price = self.get_price(state.symbol, state.timestamp)
        previous_price = self.get_price(state.symbol, state.timestamp - self.interval_timedelta)
        price_ratio = current_price / previous_price
        state.position_value *= price_ratio
        
        new_total_value = state.remaining_balance + state.position_value
        return old_total_value, new_total_value


    def close_long_trade(self, state):
        old_total_value = state.remaining_balance + state.position_value
        closing_fee = state.position_value * self.configs['fee']
        state.remaining_balance += state.position_value - closing_fee
        new_total_value = state.remaining_balance

        state.reset_history(self.configs['sequence_length'])
        
        return old_total_value, new_total_value


    def close_short_trade(self, state):
        old_total_value = state.remaining_balance + state.position_value
        closing_fee = abs(state.position_value) * self.configs['fee']
        state.remaining_balance += state.position_value - closing_fee
        new_total_value = state.remaining_balance

        state.reset_history(self.configs['sequence_length'])

        return old_total_value, new_total_value


    def finalize_trade(self, state, old_total_value, new_total_value):
        terminated = False
        truncated = False
        
        if new_total_value <= self.configs['initial_balance'] * 0.1:
            terminated = True
        
        if state.step >= state.episode_length:
            truncated = True
        
        if state.env_type == 'train' or state.env_type == 'train_val':
            end_timestamp = self.configs['train_data_end']
        elif state.env_type == 'val':
            end_timestamp = self.configs['val_data_end']

        if state.timestamp >= end_timestamp:
            truncated = True
        
        if terminated or truncated:
            if state.action < 0:
                _, new_total_value = self.close_short_trade(state)
            elif state.action > 0:
                _, new_total_value = self.close_long_trade(state)
        
        train_reward = np.log2(max(new_total_value / old_total_value, 0.1)) * 2 * 2
        val_reward = new_total_value - old_total_value
        
        return train_reward, val_reward, terminated, truncated
    

    def get_price(self, symbol, timestamp):
        shortest_interval = self.configs['intervals'][-1]
        timestamps = self.data[shortest_interval][symbol]['timestamps']
        prices = self.data[shortest_interval][symbol]['close']
        index = np.searchsorted(timestamps, timestamp + 1000) - 2
        return prices[index]

#endregion
#region other

    def calculate_symbol_probabilities(self):
        if self.mode == 'train':
            symbols = self.train_symbols
            shortest_interval = self.configs['intervals'][-1]
            data_end = self.configs['train_data_end']
            weights = [data_end - self.data[shortest_interval][symbol]['timestamps'][0] for symbol in symbols]
            probabilities = np.array(weights) / np.sum(weights)
            return probabilities


    def calculate_interval_timedelta(self):
        shortest_interval = self.configs['intervals'][-1]
        symbol = next(iter(self.data[shortest_interval]))
        timestamps = self.data[shortest_interval][symbol]['timestamps']
        return timestamps[1] - timestamps[0]


    def choose_symbol(self, state, env_index):
        if state.env_type == 'train' or state.env_type == 'train_val':
            return self.rng.choice(self.train_symbols, p=self.symbol_probabilities)
        elif state.env_type == 'val' and self.mode == 'train':
            return self.rng.choice(self.val_symbols)
        elif state.env_type == 'val' and self.mode == 'eval':
            return self.val_symbols[env_index]


    def choose_start_timestamp(self, state):
        shortest_interval = self.configs['intervals'][-1]
        timestamps = self.data[shortest_interval][state.symbol]['timestamps']
        
        if self.mode == 'train':
            if state.env_type == 'train' or state.env_type == 'train_val':
                start = timestamps[0] + self.configs['min_history_length']
                end = self.configs['train_data_end']
            elif state.env_type == 'val':
                start = self.configs['train_data_end']
                end = self.configs['val_data_end']
            
            start_index = np.searchsorted(timestamps, start)
            end_index = np.searchsorted(timestamps, end) - state.episode_length
            chosen_index = self.rng.integers(start_index, end_index)
            return timestamps[chosen_index]
        
        elif self.mode == 'eval':
            start = self.configs['train_data_end']
            start_index = np.searchsorted(timestamps, start)
            return timestamps[start_index]


    def create_observation(self, klines, history):
        if not self.configs['include_history']:
            return klines
        else:
            if self.configs['data_version'] == 'parallel':
                observation = [klines]
                for value in history.values():
                    observation.append(value)
                return np.hstack(observation)
            elif self.configs['data_version'] == 'sequential':
                observation = [klines]
                for value in history.values():
                    value = np.full(5, value[-1])
                    observation.append(value)
                return np.vstack(observation)


    def convert_action(self, action):
        position_size = (action - (self.action_dim - 1) / 2) * self.configs['position_step_size']
        return np.clip(position_size, -1, 1)
    

    def scale_cumulative_reward(self, x):
        x = x * self.train_configs['cumulative_reward_scaling_x']
        x_log = np.log(np.abs(x) + 1)
        x = np.where(np.sign(x) > 0, x_log, -x_log)
        x = x * self.train_configs['cumulative_reward_scaling_y']
        return x
    

    def update_history(self, history_array, new_value, cumulative=False):
        if cumulative:
            history_array = np.roll(history_array, -1, axis=0)
            history_array[-1] = history_array[-2] + new_value
        else:
            history_array = np.roll(history_array, -1, axis=0)
            history_array[-1] = new_value
        return history_array


    def close(self):
        self.env_states = []
        self.data = None

#endregion
#region EnvState

class EnvState:
    def __init__(
            self,
            env_type=None,
            symbol=None,
            timestamp=None,
            episode_length=None,
            step=None,
            action=None,
            action_history=None,
            remaining_balance=None,
            position_value=None,
            cumulative_reward=None,
            cumulative_reward_history=None,
            action_length=None):
        self.env_type = env_type
        self.symbol = symbol
        self.timestamp = timestamp
        self.episode_length = episode_length
        self.step = step
        self.action = action
        self.remaining_balance = remaining_balance
        self.position_value = position_value
        self.cumulative_reward = cumulative_reward
        self.action_length = action_length
        self.history = {
            'action_history': action_history,
            'cumulative_reward_history': cumulative_reward_history
        }

    def reset_history(self, sequence_length):
        self.action = 0
        self.action_length = 0
        self.position_value = 0
        self.cumulative_reward = 0
        self.history['action_history'] = np.zeros((sequence_length, 1))
        self.history['cumulative_reward_history'] = np.zeros((sequence_length, 1))

#endregion