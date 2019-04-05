import numpy as np
import random
from collections import deque
from datetime import datetime

from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import History
import keras.backend.tensorflow_backend as K


class DQN:
    def __init__(self, input_shape, output_size, name="main"):

        self.input_shape = input_shape
        self.output_size = output_size
        self.net_name = name
        self.model = None

        self.__build_network()

    def __build_network(self, hidden_dimension=1024, learning_rate=1e-3):
        with K.tf.variable_scope(self.net_name):
            inputs = Input(shape=self.input_shape)

            m = Flatten()(inputs)
            m = Dense(hidden_dimension, activation='relu')(m)
            m = Dense(hidden_dimension, activation='relu')(m)
            m = Dense(hidden_dimension, activation='relu')(m)

            # 3D sample
            #             m = Conv2D(128, kernel_size=(64, 14), strides=(1, 1), padding='valid')(inputs)
            #             m = BatchNormalization()(m)
            #             m = Activation('relu')(m)

            #             m = Conv2D(128, kernel_size=(32, 1), strides=(1, 1), padding='valid')(m)
            #             m = BatchNormalization()(m)
            #             m = Activation('relu')(m)

            #             m = Conv2D(128, kernel_size=(32, 1), strides=(1, 1), padding='valid')(m)
            #             m = BatchNormalization()(m)
            #             m = Activation('relu')(m)

            #             m = Flatten()(m)
            #             m = Dense(512)(m)
            #             m = Activation('relu')(m)

            outputs = Dense(self.output_size)(m)

        model = Model(inputs=inputs, outputs=outputs)
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def predict(self, state):
        input_shape = list(self.input_shape)
        input_shape.insert(0, 1)
        x = np.reshape(state, input_shape)
        return self.model.predict(x)

    def update(self, x_stack, y_stack):
        history = History()
        return self.model.fit(x_stack, y_stack, epochs=3, batch_size=256, verbose=0, callbacks=[history]).history['loss']


class Space:
    def __init__(self, n):
        self.n = n
        self.actions = list(range(self.n))

    def sample(self):
        return random.choice(self.actions)


class EEGEnv:
    def __init__(self, x_data, y_data):
        self._max_episode_steps = 10001
        self.window = 1  # currently not used
        self.stride = 1

        self.input_shape = (128 * 1, 14)
        self.observation_space = np.zeros(shape=self.input_shape)  # shape
        self.action_space = 9
        self.x_data = x_data
        self.y_data = y_data.astype(np.float)
        self.action_space = Space(n=9)

        self.num_episodes = self.x_data.shape[0]
        self.current_episode = 0
        self.agent = None
        self.shape_3d = False

    def reset(self, shape_3d=False):
        #         print('reset')
        self.shape_3d = shape_3d
        if self.shape_3d:
            self.input_shape = (128 * 1, 14, 1)
        else:
            self.input_shape = (128 * 1, 14)
        state, done = self._get_next(reset=True)

        return state

    def step(self, action):

        info = dict()
        state, done = self._get_next()
        if done:
            if action == self.y_data[self.episode]:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0

        return state, reward, done, info

    def _get_next_episode(self):
        self.current_episode += 1
        if self.current_episode + 1 > self.x_data.shape[0]:
            self.current_episode = 0

    def _get_next(self, reset=False):
        if reset:
            self.episode = 0  # should do random pick from episode list
            self.agent = self.generator()

        #         print('_get_next')
        state, pos = next(self.agent)

        done = False
        if (pos + self.window * 128 > self.x_data.shape[1]) or np.count_nonzero(
                np.sum(state, axis=1) == 0) == self.window * 128:
            #             print(self.current_episode, pos, np.count_nonzero(np.sum(state, axis=1) == 0))

            done = True
            state = np.zeros(shape=(self.window * 128, self.x_data.shape[2]))
            self._get_next_episode()
            self.agent = self.generator()

        if self.shape_3d:
            state = self._to_3d_state(state)

        return [state, done]

    def _to_3d_state(self, state):
        state = state.reshape(state.shape[0], state.shape[1], 1)
        return state

    def generator(self):
        for pos in range(0, self.x_data.shape[1], self.stride * 128):
            state = self.x_data[self.current_episode, pos: min(pos + self.window * 128, self.x_data.shape[1]), :]

            #             print('generator', pos)
            #             print(state, reward, done)
            yield [state, pos]


class ReinforcementLearningDQN:
    def __init__(self, initial_params, gpu=0):

        self.env = initial_params[0]
        # Q learning values
        self.REPLAY_MEMORY = 50000
        self.dis = 0.9

        self.mainDQN = DQN(self.env.input_shape, self.env.action_space.n, name='main')
        self.targetDQN = DQN(self.env.input_shape, self.env.action_space.n, name='target')
        self.model = None

    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=128, verbose=1):
        self.env = EEGEnv(x_data=x_train, y_data=y_train)

        env = self.env
        mainDQN = self.mainDQN
        targetDQN = self.targetDQN

        # max_episodes = x_train.shape[0] * epochs
        max_episodes = 1

        # store the previous observations in replay memory
        replay_buffer = deque()
        state = env.reset(shape_3d=False)

        self.copy_model(dest_model=targetDQN.model, src_model=mainDQN.model)

        print('Number of samples:', x_train.shape[0])
        t = datetime.now()

        # Reinforcement Learning
        for episode in range(max_episodes):

            e = 1. / ((episode / 10) + 1)
            done = False

            while not done:
                # Exploration & Exploit
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                new_state, reward, done, info = env.step(action)

                replay_buffer.append((state, action, reward, new_state, done))
                #             if len(replay_buffer) > REPLAY_MEMORY:
                #                 replay_buffer.popleft()

                state = new_state
                if done:
                    break

            if episode % 10 == 1:  # train every 10 episodes
                # Get a random batch of experiences
                loss = 0
                # replay with samples
                #             for _ in range(50):
                #                 # Minibatch works better
                #                 minibatch = random.sample(replay_buffer, 10)
                #                 loss = simple_replay_train(mainDQN, targetDQN, minibatch)
                loss = self.simple_replay_train(mainDQN, targetDQN, replay_buffer)
                replay_buffer = deque()

                print('[Elapsed Time: %i] Episode: %i Loss: %f' % ((datetime.now() - t).seconds, episode, np.mean(loss)))
                t = datetime.now()

                # copy model
                self.copy_model(dest_model=targetDQN.model, src_model=mainDQN.model)

        self.test(mainDQN, x_test, y_test)

        self.model = [self.mainDQN, self.targetDQN]

        return [self.mainDQN, self.targetDQN]


    @staticmethod
    def get_initial_params(x_train, y_train):
        # input_shape = (x_train.shape[1], x_train.shape[2])
        env = EEGEnv(x_data=x_train, y_data=y_train)
        return [env]


    # Extras
    def test(self, dqn, x_data, y_data):
        env = EEGEnv(x_data=x_data, y_data=y_data)

        # See our trained network in action
        state = env.reset()
        reward_sum = 0

        while True:
            # env.render()
            action = np.argmax(dqn.predict(state))
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                print('Total score: {}'.format(reward_sum))
                break

    def copy_model(self, dest_model, src_model):
        # Copy variables src_scope to dest_scope
        dest_model.set_weights(src_model.get_weights())

    # Replay to train
    def simple_replay_train(self, mainDQN, targetDQN, train_batch):
        env = self.env
        dis = self.dis

        input_shape = list(env.input_shape)
        input_shape.insert(0, 0)
        state_shape = list(env.input_shape)
        state_shape.insert(0, 1)
        #     print('input_shape', input_shape)
        x_stack = np.empty(shape=input_shape)
        y_stack = np.empty(shape=[0, env.action_space.n])

        # Get stored information from the buffer
        for state, action, reward, next_state, done in train_batch:
            Q = mainDQN.predict(state)

            # terminal?
            if done:
                #             print(reward)
                Q[0, action] = reward
            else:
                # Q[0, action] = reward + dis * np.max(mainDQN.predict(next_state)) # normal DQN
                Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))  # network separate targetDQN

            #         print(y_stack.shape)
            #         print(Q.shape)
            #         print('x_stack.shape', x_stack.shape)
            #         print('state.shape', state.shape)
            #         print('state_shape', state_shape)
            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state.reshape(state_shape)])

        return mainDQN.update(x_stack, y_stack)

