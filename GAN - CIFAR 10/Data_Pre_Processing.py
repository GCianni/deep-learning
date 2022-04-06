from keras.datasets.cifar10 import load_data
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint


def load_full_data():
    (trainX, trainy), (testX, testy) = load_data()
    return (trainX, trainy), (testX, testy)


def load_real_samples():
    (trainX, _), (_, _) = load_data()
    # cConvert to float and scale.
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5  # Generator uses tanh activation so rescale
    # original images to -1 to 1 to match the output of generator.
    return X


def generate_real_samples(dataset, n_samples):
    # choose random images
    ix = randint(0, dataset.shape[0], n_samples)
    # select the random images and assign it to X
    X = dataset[ix]
    # generate class labels and assign to y
    y = ones((n_samples, 1))  # Label=1 indicating they are real
    return X, y


# generate n_samples number of latent vectors as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict using generator to generate fake samples.
    X = generator.predict(x_input)
    # Class labels will be 0 as these samples are fake.
    y = zeros((n_samples, 1))  # Label=0 indicating they are fake
    return X, y
