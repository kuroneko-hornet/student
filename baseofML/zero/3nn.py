from keras import Model
from keras.layers import Input, Dense, Activation
from keras.datasets import mnist
import numpy as np
import pickle

def load_weight(path = 'sample_weight.pkl'):
    with open(path, 'rb') as f:
        weights = pickle.load(f)
        weight_array = [weights['W1'], weights['b1'],
                        weights['W2'], weights['b2'],
                        weights['W3'], weights['b3']]
        return weight_array


def preprocess(data, data_n, data_shape):
    data = data.reshape(data_n, data_shape)
    data = data.astype('float')
    data /= 255.
    return data


def set_model(shape, uni_1,  uni_2, uni_3, out_activation):
    _input = Input(shape=shape)
    _hidden = Dense(units=uni_1, activation='sigmoid')(_input)
    _hidden = Dense(units=uni_2, activation='sigmoid')(_hidden)
    _hidden = Dense(units=uni_3)(_hidden)
    _output = Activation(out_activation)(_hidden)

    model = Model(inputs=_input, outputs=_output)
    # print(model.summary())

    model.set_weights(load_weight('sample_weight.pkl'))
    # print(model.get_weights())

    return model


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = preprocess(x_train, 60000, 784)
    x_test = preprocess(x_test, 10000, 784)

    model = set_model(shape = (784,),
                      uni_1 = 50,
                      uni_2 = 100,
                      uni_3 = 10,
                      out_activation = 'softmax')
    
    prediction = model.predict(x_test)
    prediction = np.argmax(prediction, axis=1)

    print( f'accuracy: {np.sum(y_test == prediction) / len(y_test)}')

if __name__ == '__main__':
    main()
