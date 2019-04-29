from keras import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt


def load_preprocess_data(params):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # preprocess
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float') /255 
    x_test = x_test.astype('float') /255 

    y_train = to_categorical(y_train, num_classes=params['output_dim'])
    y_test = to_categorical(y_test, num_classes=params['output_dim'])

    return x_train, y_train, x_test, y_test


def set_params():
    params = {
        'img_shape' : (28 * 28, ),
        'hidden_dim' : 100,
        'output_dim' : 10,
        'batch_size' : 100,
        'learning_rate' : 0.1,
        'epochs' : 17
    }
    return params


def set_model(params):
    _input = Input(shape=params['img_shape'])
    _hidden = Dense(params['hidden_dim'], activation='sigmoid')(_input)
    _output = Dense(params['output_dim'], activation='softmax')(_hidden)
    model = Model(inputs=_input, outputs=_output)
    return model


def train(model, params, x_train, y_train, x_test, y_test):
    model.compile(\
        optimizer = SGD(lr=params['learning_rate']),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
    result = model.fit(\
        x = x_train, y = y_train,
        batch_size = params['batch_size'],
        epochs = params['epochs'],
        verbose = 1,
        validation_data = (x_test, y_test))
    return result


def show_result(history, epochs):
    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure()
    plt.plot(range(1, epochs+1), loss, marker='.', label='train')
    plt.plot(range(1, epochs+1), val_loss, marker='.', label='test')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss.jpg')


def main():
    params = set_params()
    model = set_model(params)
    # model.summary()
    x_train, y_train, x_test, y_test = \
        load_preprocess_data(params)
    _result = train(model, params, x_train, y_train, x_test, y_test)
    show_result(_result.history, params['epochs'])


if __name__ == '__main__':
    main()