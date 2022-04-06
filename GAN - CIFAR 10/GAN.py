from numpy import ones
from Data_Pre_Processing import generate_fake_samples, generate_latent_points, generate_real_samples
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout


# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.
def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))  # 16x16x128
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))  # 8x8x128
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())  # shape of 8192
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))  # shape of 1 since it's a binary classifier

    # compile model
    opt = adam_v2.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def define_generator(latent_dim):  # latent_dim is the dimension of the latent vector (e.g., 100) -> random vector
    model = Sequential()
    # We will reshape input latent vector into 8x8 image as a starting point.
    # So n_nodes for the Dense layer can be 128x8x8 so when we reshape the output
    # it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
    n_nodes = 128 * 8 * 8  # 8192 nodes
    model.add(Dense(n_nodes, input_dim=latent_dim))  # Dense layer so we can work with 1D latent vector
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))  # 8x8x128 dataset from the latent vector.
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # 16x16x128
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # 32x32x128
    model.add(LeakyReLU(alpha=0.2))
    # generate
    model.add(Conv2D(3, (8, 8), activation='tanh', padding='same'))  # 32x32x3
    return model  # Model not compiled as it is not directly trained like the discriminator.
    # Generator is trained via GAN combined model.


def define_gan(generator, discriminator):
    discriminator.trainable = False  # Discriminator is trained separately. So set to not trainable.
    # connect generator and discriminator
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # compile model
    opt = adam_v2.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)  # the discriminator model is updated for a half batch of real samples
    # and a half batch of fake samples, combined a single batch.
    # manually enumerate epochs and bacthes.
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # Train the discriminator on real and fake images, separately (half batch each)
            # Research showed that separate training is more effective.
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            ##train_on_batch allows you to update weights based on a collection
            # of samples you provide
            # Let us just capture loss and ignore accuracy value (2nd output below)
            d_loss_real, _ = d_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)

            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            # This is where the generator is trying to trick discriminator into believing
            # the generated image is true (hence value of 1 for y)
            y_gan = ones((n_batch, 1))

            # Generator is part of combined model where it got directly linked with the discriminator
            # Train the generator with latent_dim as x and 1 as y.
            # Again, 1 as the output as it is adversarial and if generator did a great
            # job of folling the discriminator then the output would be 1 (true)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
    # save the generator model
    # g_model.save('cifar_generator_2epochs.h5')
