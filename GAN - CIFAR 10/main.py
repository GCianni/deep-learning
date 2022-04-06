import tensorflow as tf
import GAN
import Data_Pre_Processing
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # size of the latent space
    latent_dim = 100
    # create the discriminator
    discriminator = GAN.define_discriminator()
    # create the generator
    generator = GAN.define_generator(latent_dim)
    # create the gan
    gan_model = GAN.define_gan(generator, discriminator)
    # load image data
    dataset = Data_Pre_Processing.load_real_samples()
    
    # train model
    GAN.train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=10)


    # Plot generated images
    def show_plot(examples, n):
        for i in range(n * n):
            plt.subplot(n, n, 1 + i)
            plt.axis('off')
            plt.imshow(examples[i, :, :, :])
        plt.show()

    # load model
    model = load_model('cifar_generator_2Aepochs.h5')  # Model trained for 100 epochs
    # generate images
    latent_points = Data_Pre_Processing.generate_latent_points(100, 25)  # Latent dim and n_samples
    # generate images
    X = model.predict(latent_points)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    X = (X*255).astype(np.uint8)
    # plot the result
    show_plot(X, 1)
