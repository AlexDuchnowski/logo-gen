import tensorflow as tf
from callback import GANMonitor
from model import WGAN, get_critic_model, get_generator_model
from preprocess import make_input_generator

BATCH_SIZE = 10
LATENT_DIM = 128

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

c_model = get_critic_model()
c_model.summary()
g_model = get_generator_model(LATENT_DIM)
g_model.summary()

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# Instantiate the customer `GANMonitor` Keras callback.
cbk = GANMonitor(num_img=3, latent_dim=LATENT_DIM)

# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=c_model,
    generator=g_model,
    latent_dim=LATENT_DIM,
    discriminator_extra_steps=3,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Run preprocessing steps.
gen = make_input_generator('LLD-logo.hdf5', BATCH_SIZE, epochs=1)

# Start training the model.
# THIS WON'T WORK AS IS BECAUSE THE GENERATOR YIELDS 3 SEPARATE THINGS
wgan.fit(x=gen, callbacks=[cbk])