import os
import sys
import yaml
import pandas as pd
import numpy as np
rl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Reinforcement Learning', 'src'))
sys.path.append(rl_path)
from downloader import Downloader
from processor import Processor
from bandit_jax_new import Bandits
from actor_jax import Actor as ActorJax
from training import Training
import configs_trader as configs



class CombinedState:
    def __init__(self, symbols, configs):
        self.symbol_states = {
            symbol: {
                'action': 0,
                'action_prob': 0,
                'q_action': 0,
                'v': 0,
                'close': 0,
                'turnover': 0,
                'timestamp': 0,
                'balance': configs['initial_balance'],
                'history': {
                    'action_history': np.zeros((configs['sequence_length'], 1)),
                    'cumulative_reward_history': np.zeros((configs['sequence_length'], 1)),
                }
            } for symbol in symbols
        }
        self.trading_symbols = {}


class Actor:
    def __init__(self):
        self.downloader = Downloader()
        self.processor = Processor()
        with open('configs.yaml', 'r') as file:
            self.configs = yaml.safe_load(file)
        # self.symbols = self.load_symbols()
        self.symbols = ['BTCUSDT']
        self.state = CombinedState(self.symbols, self.configs)


    async def initialize(self):
        env_name = 'Crypto-v0'
        load_run = configs.ACTOR
        train_parameters = {"load_run": load_run, "train_frames": 0}
        
        self.training = Training(env_name, train_parameters, {})
        self.bandits = Bandits(self.training)
        self.index = self.bandits.get_index_data(only_index=True)
        self.actor = ActorJax(self.training)
        await self.actor.pull_weights(training=False, target_eval=True)
        return self

#region load data

    def load_symbols(self):
        last_launch_time = self.configs['last_launch_time']
        exclude_symbols = self.configs['exclude_symbols']
        return self.downloader.load_trading_symbols(last_launch_time, exclude_symbols)['Symbol'].tolist()
    

    def reload_symbols(self):
        self.symbols = self.load_symbols()
        old_symbols = set(self.state.symbol_states.keys())
        for symbol in old_symbols - set(self.symbols):
            del self.state.symbol_states[symbol]


    async def fetch_and_process_klines(self, intervals_back=0):
        processed_results, klines_complete, incomplete_klines, last_klines = await self.downloader.fetch_recent_klines(self.symbols, self.configs['intervals'], self.configs['sequence_length'], intervals_back)
        processed_klines, processing_issues = self.processor.process_recent_klines(processed_results)
        return processed_klines, klines_complete, incomplete_klines, processing_issues, last_klines


    def combine_recent_klines(self, processed_klines):
        combined_klines = {}
        for (symbol, interval), klines in processed_klines.items():
            if symbol not in combined_klines:
                combined_klines[symbol] = []
            padded_klines = self.pad_klines(klines['data'], self.configs['sequence_length'])
            combined_klines[symbol].append(padded_klines)
        
        for symbol, klines_list in combined_klines.items():
            sorted_klines = [klines for _, klines in sorted(
                zip(self.configs['intervals'], klines_list),
                key=lambda x: self.configs['intervals'].index(x[0])
            )]
            combined_klines[symbol] = np.hstack(sorted_klines)
        
        return combined_klines


    def pad_klines(self, klines, target_length):
        current_length = klines.shape[0]
        if current_length < target_length:
            pad_length = target_length - current_length
            padded_klines = np.pad(klines, ((pad_length, 0), (0, 0)), mode='constant')
        else:
            padded_klines = klines[-target_length:]
        return padded_klines


    def create_observation(self, combined_klines):
        observations = []
        for symbol in self.symbols:
            klines = combined_klines[symbol]
            symbol_state = self.state.symbol_states[symbol]
            symbol_observation = np.hstack((
                klines,
                symbol_state['history']['action_history'],
                symbol_state['history']['cumulative_reward_history']
            ))
            observations.append(symbol_observation)
        return np.array(observations)
    
#endregion
#region actions

    def get_action_values(self, observations):
        actions, action_probs, q_action, v = self.actor.get_actions(
            observations, 
            np.expand_dims(self.index, axis=0), 
            stochastic=False, 
            training=False,
            return_values=True
        )
        return actions, action_probs, q_action, v


    def create_action_summary(self, actions, action_probs, q_action, v, last_klines, create_df=False):
        summary = {}
        timestamps = []
        for i, symbol in enumerate(self.symbols):
            kline = last_klines.get(symbol, {})
            timestamp = kline.get('timestamp')
            if timestamp is not None:
                timestamps.append(timestamp)
            summary[symbol] = {
                'action': self.convert_action(actions[i]),
                'action_prob': action_probs[i, 0],
                'q_action': q_action[i, 0],
                'v': v[i, 0],
                'close': kline.get('close', None),
                'turnover': kline.get('turnover', None),
                'timestamp': timestamp
            }
        
        if timestamps:
            most_common_timestamp = max(set(timestamps), key=timestamps.count)
            summary = {symbol: data for symbol, data in summary.items() 
                       if data['timestamp'] == most_common_timestamp}
        
        if create_df:
            sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1]['q_action'], reverse=True))
            return pd.DataFrame.from_dict(sorted_summary, orient='index').reset_index().rename(columns={'index': 'Symbol'})
        
        return summary
    

    def choose_symbols(self, summary):
        chosen_symbols = {}

        for symbol in self.state.trading_symbols:
            if symbol in summary:
                chosen_symbols[symbol] = summary[symbol]
        
        remaining_slots = configs.MAX_TRADING_SYMBOLS - len(chosen_symbols)

        if remaining_slots > 0:
            new_symbols = {}
            for symbol, data in summary.items():
                if symbol not in self.state.trading_symbols:
                    if (data['action_prob'] >= configs.PROBABILITY_THRESHOLD and 
                        data['q_action'] >= configs.Q_THRESHOLD and
                        data['action'] != 0):
                        new_symbols[symbol] = data
            
            sorted_new_symbols = sorted(new_symbols.items(), key=lambda x: x[1]['q_action'], reverse=True)
            chosen_symbols.update(dict(sorted_new_symbols[:remaining_slots]))
        
        return chosen_symbols
    

    def update_trading_symbols(self, chosen_symbols):
        for symbol in self.state.trading_symbols:
            old_action = self.state.trading_symbols[symbol]['action']
            new_action = chosen_symbols[symbol]['action']
            
            if old_action * new_action <= 0:
                chosen_symbols[symbol]['action'] = 0
                self.state.trading_symbols[symbol].update(chosen_symbols[symbol])
                self.state.symbol_states[symbol]['action_history'] = np.zeros((configs.SEQUENCE_LENGTH, 1))
                
            elif old_action == new_action:
                self.state.trading_symbols[symbol].update(chosen_symbols[symbol])
                action_history = self.state.symbol_states[symbol]['action_history']
                self.state.symbol_states[symbol]['action_history'] = self.update_history(action_history, new_action)
                del chosen_symbols[symbol]
            
        for symbol, data in chosen_symbols.items():
            if symbol not in self.state.trading_symbols:
                self.state.trading_symbols[symbol] = data
                action_history = self.state.symbol_states[symbol]['action_history']
                self.state.symbol_states[symbol]['action_history'] = self.update_history(action_history, data['action'])

        self.state.trading_symbols = {symbol: data for symbol, data in self.state.trading_symbols.items() if data['action'] != 0}

        return chosen_symbols


    def convert_action(self, action):
        action_dim = int(2 / self.configs['position_step_size'] + 1)
        position_size = (action - (action_dim - 1) / 2) * self.configs['position_step_size']
        return np.clip(position_size, -1, 1)
    

    def update_history(self, history_array, new_value, cumulative=False):
        if cumulative:
            history_array = np.roll(history_array, -1, axis=0)
            history_array[-1] = history_array[-2] + new_value
        else:
            history_array = np.roll(history_array, -1, axis=0)
            history_array[-1] = new_value
        return history_array

#endregion
#region trade

    async def run(self):
        processed_klines, klines_complete, incomplete_klines, processing_issues, last_klines = await self.fetch_and_process_klines()
        combined_klines = self.combine_recent_klines(processed_klines)
        observations = self.create_observation(combined_klines)
        
        actions, action_probs, q_action, v = self.get_action_values(observations)
        summary = self.create_action_summary(actions, action_probs, q_action, v, last_klines)

        chosen_symbols = self.choose_symbols(summary)
        chosen_actions = self.update_trading_symbols(chosen_symbols)

        return chosen_actions