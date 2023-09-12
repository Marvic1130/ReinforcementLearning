import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

# CartPole 환경 로드
env = gym.make("CartPole-v1")
num_actions = env.action_space.n

# 사용자 입력 받기
user_input = input("학습된 모델을 사용하시겠습니까? (Y/N): ")

if user_input.lower() == "y":
    # 기존에 학습된 모델을 불러오기
    model = tf.keras.models.load_model("dqn_model.h5")
else:
    # Q-Network 모델 생성
    model = tf.keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_actions)
    ])

    # 최적화 알고리즘 및 손실 함수 설정
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.losses.mean_squared_error

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss=loss_fn)

    # 학습 매개변수 설정
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    batch_size = 32
    gamma = 0.99

    # Experience Replay를 위한 데이터 저장소 초기화
    replay_memory = []

    # DQN 학습
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(np.reshape(state, [1, 4]))
                action = np.argmax(q_values[0])

            # 환경에서 행동 실행 및 다음 상태 및 보상 얻기
            step_result = env.step(action)
            next_state, reward, done, info = step_result

            if done:
                reward = -10

            replay_memory.append((state, action, reward, next_state, done))

            if len(replay_memory) >= batch_size:
                minibatch = random.sample(replay_memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target += gamma * np.amax(model.predict(np.reshape(next_state, [1, 4]))[0])
                    target_f = model.predict(np.reshape(state, [1, 4]))
                    target_f[0][action] = target
                    model.fit(np.reshape(state, [1, 4]), target_f, epochs=1, verbose=0)

            state = next_state
            total_reward += reward

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # 학습된 모델 저장
    model.save("dqn_model.h5")

# 학습된 모델을 사용하여 CartPole을 실행하여 결과 확인
for _ in range(5):
    state = env.reset()
    total_reward = 0
    while True:
        action = np.argmax(model.predict(np.reshape(state, [1, 4]))[0])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()
        if done:
            break

env.close()
