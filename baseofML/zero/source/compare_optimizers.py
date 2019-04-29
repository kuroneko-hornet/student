from keras import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam
import matplotlib.pyplot as plt
from keras.initializers import he_normal, he_uniform


def load_preprocess_data(params):
    '''
        args: params
        output: x_train, y_train, x_test, y_test
    '''
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
    ''' no args'''
    params = {
        'activation' : 'relu',
        'batch_size' : 100,
        'epochs' : 20,
        'hidden_dim' : 100,
        'img_shape' : (28 * 28, ),
        'kernel_initializer' : 'he_normal'
        'layer_num' : 5,
        'learning_rate' : 0.01,
        'output_dim' : 10,
    }
    return params


def set_model(params):
    ''' Args: params'''
    _input = Input(shape=params['img_shape'])
    _hidden = _input #inputが書き変わるので注意
    for _ in range(params['layer_num']-1):
        _hidden = Dense(\
            units = params['hidden_dim'],
            activation = params['activation'],
            kernel_initializer = params['he_normal'])(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Dropout(0.2)(_hidden)
    _output = Dense(params['output_dim'], activation='softmax')(_hidden)
    model = Model(inputs=_input, outputs=_output)
    return model


def train(model, params, opt, x_train, y_train, x_test, y_test):
    ''' Args: model, params, opt, x_train, y_train, x_test, y_test'''
    model.compile(\
        optimizer = opt,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
    result = model.fit(\
        x = x_train, y = y_train,
        batch_size = params['batch_size'],
        epochs = params['epochs'],
        verbose = 1,
        validation_data = (x_test, y_test))
    return result


def show_result(history, epochs, opt_name):
    ''' Args: history, epochs, opt_name'''
    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure()
    plt.plot(range(1, epochs+1), loss, marker='.', label='train')
    plt.plot(range(1, epochs+1), val_loss, marker='.', label='test')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(f'loss/dropout_{opt_name}_he_normal.jpg')


def main():

    optimizers = {
        'sgd':SGD(),
        'momentum':SGD(momentum=0.9),
        'adagrad':Adagrad(),
        'adam':Adam()
        }
    
    for opt_name, opt in optimizers.items():
        params = set_params()
        model = set_model(params)
        # model.summary()
        x_train, y_train, x_test, y_test = \
            load_preprocess_data(params)

        _result = train(model, params, opt, x_train, y_train, x_test, y_test)
        show_result(_result.history, params['epochs'], opt_name)


if __name__ == '__main__':
    main()

