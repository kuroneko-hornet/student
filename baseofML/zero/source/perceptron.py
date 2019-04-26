from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np


def predict(input_array, model):
    '''
    Args: model, gatename
    '''
    X = input_array
    pre_y = model.predict(X)
    # pre_y[pre_y <= 0] = False
    # pre_y[pre_y > 0] = True
    return pre_y


# Sequential model
def seq():
    model = Sequential()
    model.add( Dense(input_dim=2, units=1) )
    # [weight, bias]
    model.set_weights( [np.array([[0.5], [0.5]]), np.array([-0.7])] )
    # test
    print( get_test_res(model) )

# functional API
def func():
    _input = Input(shape=(2,))
    _output = Dense(units=1)(_input)
    model = Model(inputs=_input, outputs=_output)
    model.set_weights( [np.array([[0.5], [0.5]]), np.array([-0.7])] )

    print( get_test_res(model) )

if __name__ == '__main__':
    seq()
    func()