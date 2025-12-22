import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from traffic import run_simulation_for_config
from dt import run_prediction_model

# Define IAT and QoS values and action pairs
IAT_VALUES = [0.1, 0.02, 0.4]
QOS_VALUES = ['UGS', 'RTPS', 'ERTPS', 'NRTPS', 'BE']
ACTION_PAIRS = [(iat, qos) for iat in IAT_VALUES for qos in QOS_VALUES]
ACTION_SIZE = len(ACTION_PAIRS)

# Hyperparameters
EPISODES = 50
MAX_STEPS = 20
BATCH_SIZE = 32
STATE_SIZE = 2  # Example: throughput, delay, previous action index

class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_dim=self.state_size))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_network_simulation(iat, qos):
    run_simulation_for_config(iat, qos)

    actual_throughput, predicted_throughput, mse, r2 = run_prediction_model()

    metrics = {"throughput": actual_throughput.mean()}
    simulation_status = True
    return simulation_status, metrics, mse, r2, actual_throughput, predicted_throughput

def calculate_reward(predicted_throughput, mse):
    return predicted_throughput - mse

def build_state(metrics, prev_action_idx):
    return np.array([
        metrics['throughput'],
        prev_action_idx / (len(ACTION_PAIRS) - 1)
    ])

def check_terminal_condition(step):
    return step >= MAX_STEPS

# Main training loop
def train_dqn():
    agent = DQNAgent(state_size=STATE_SIZE)
    prev_action_idx = 0

    best_pairs_csv = "best_pairs_dqn.csv"
    if not os.path.exists(best_pairs_csv):
        pd.DataFrame(columns=["episode", "iat", "qos", "avg_mse", "r2_score"]).to_csv(best_pairs_csv, index=False)

    for episode in range(EPISODES):
        print(f"Episode {episode+1}/{EPISODES}")
        initial_metrics = {'throughput': 0}
        state = build_state(initial_metrics, prev_action_idx)
        state = np.reshape(state, [1, STATE_SIZE])

        best_mse = float("inf")
        best_pair = None    

        for step in range(MAX_STEPS):
            action_idx = agent.act(state)
            iat, qos = ACTION_PAIRS[action_idx]

            result =  run_network_simulation(iat, qos)
            simulation_status, metrics, mse, r2, actual_series, predicted_series = result
            reward = calculate_reward(metrics['Throughput'], mse)

            next_state = build_state(metrics, action_idx)
            next_state = np.reshape(next_state, [1, STATE_SIZE])

            done = check_terminal_condition(step)

            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            prev_action_idx = action_idx

            if mse<best_mse:
                best_mse = mse
                best_pair = (iat, qos)

            if done:
                print(f"Episode {episode+1} ended after {step+1} steps.")
                break

        if best_pair:
            iat, qos = best_pair
            print(f"Best pair in Episode {episode+1}: IAT={iat}, QoS={qos}, Avg MSE={best_mse:.4f}, R2_Score={r2:.4f}")
            df = pd.DataFrame([[episode+1, iat, qos, best_mse, r2]], columns=["episode", "iat", "qos", "avg_mse", "r2_score"])
            df.to_csv(best_pairs_csv, mode="a", header=False, index=False)

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    print("Training complete.")

if __name__ == "__main__":
    train_dqn()