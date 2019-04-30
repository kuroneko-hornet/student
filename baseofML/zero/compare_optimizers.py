from keras import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam
import matplotlib.pyplot as plt
from keras.initializers import he_normal, he_uniform


def load_preprocess_data(params):
    '''
        args: params
        output: x_train, y_train, x_test, y_test
    '''
    use_fashipon = True
    if use_fashipon: 
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
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
        'batch_size' : 1000,
        'epochs' : 3,
        'hidden_dim' : 100,
        'img_shape' : (28 * 28, ),
        'kernel_initializer' : 'he_normal',
        'layer_num' : 6,
        'learning_rate' : 0.01,
        'output_dim' : 10,
    }
    return params


def set_optimizer():
    optimizers = {
        'sgd':SGD(),
        'momentum':SGD(momentum=0.9),
        'adagrad':Adagrad(),
        'adam':Adam()
        }
    return optimizers


def set_model(params):
    ''' Args: params'''
    _input = Input(shape=params['img_shape'])
    _hidden = _input #inputが書き変わるので注意
    for _ in range(params['layer_num']-1):
        _hidden = Dense(\
            units = params['hidden_dim'],
            activation = params['activation'],
            kernel_initializer = params['kernel_initializer'])(_hidden)
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


def build_result_file(figure_name, result_histories, epochs):
    ''' Args: history, epochs'''
    h_size = len(result_histories)
    f1 = plt.figure(figsize=(5,h_size*2.5))

    plt_num = 1
    for k, v in result_histories.items():
        d_test = v[f'{figure_name}']
        d_train = v[f'val_{figure_name}']
        plt.subplot(h_size, 1,plt_num)
        plt.plot(range(1, epochs+1), d_train, marker='.', label=f'train')
        plt.plot(range(1, epochs+1), d_test, marker='.', label=f'test')

        plt.title(k)
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.tight_layout()

        plt_num += 1

    plt.savefig(f'{figure_name}/test.jpg')


def main():

    optimizers = set_optimizer()
    params = set_params()
    x_train, y_train, x_test, y_test = \
        load_preprocess_data(params)

    result_histories = {}
    for opt_name, opt in optimizers.items():
        model = set_model(params)
        # model.summary()
        _result = train(model, params, opt, x_train, y_train, x_test, y_test)
        result_histories[opt_name] = _result.history

    build_result_file('loss', result_histories, params['epochs'])
    build_result_file('acc', result_histories, params['epochs'])


if __name__ == '__main__':
    main()

