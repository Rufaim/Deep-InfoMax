import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as pyplot
from models import DeepInfoMax, BasicEncoder64x64, MutualInfoType


EPOCHS = 300
BUFFER_SIZE = 10000
BATCH_SIZE = 16
ENCODING_DIM = 128
MUTUAL_INFO_TYPE = MutualInfoType.DV
ALPHA = 1.
BETA = 0.
GAMMA = 0.1
LEARNING_RATE = 3e-4
LOG_DIR = "logs"
LOG_POSTFIX = "prior_global"



@tf.function
def normalize(image):
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1

def save_image(img,imgpath):
    f = pyplot.figure(figsize=(20, 20))
    pyplot.imshow(img * 0.5 + 0.5)
    pyplot.axis('off')
    f.savefig(imgpath)
    pyplot.close(f)

def train(model, data, epochs, summary_writer):
    global_mi = tf.keras.metrics.Mean()
    local_mi = tf.keras.metrics.Mean()
    encoder_loss = tf.keras.metrics.Mean()
    disc_loss = tf.keras.metrics.Mean()

    step = 0
    step = tf.cast(step, tf.int64)
    for epoch in range(epochs):
        for inp in data:
            gmi, lmi, le, ld = model.train_step(inp)
            global_mi(gmi)
            local_mi(lmi)
            encoder_loss(le)
            disc_loss(ld)

            with summary_writer.as_default():
                tf.summary.scalar('raw/encoder loss', le, step=step)
                tf.summary.scalar('raw/discriminator loss', ld, step=step)
                tf.summary.scalar('raw/global mutual info', gmi, step=step)
                tf.summary.scalar('raw/local mutual info', lmi, step=step)

            step += 1

        with summary_writer.as_default():
            tf.summary.scalar('smooth/encoder loss', encoder_loss.result(), step=epoch)
            tf.summary.scalar('smooth/discriminator loss', disc_loss.result(), step=epoch)
            tf.summary.scalar('smooth/global mutual info', global_mi.result(), step=epoch)
            tf.summary.scalar('smooth/local mutual info', local_mi.result(), step=epoch)

        global_mi.reset_states()
        local_mi.reset_states()
        encoder_loss.reset_states()
        disc_loss.reset_states()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.data.Dataset.from_tensor_slices(x_train) \
            .map(normalize,num_parallel_calls=-1)    \
            .shuffle(BUFFER_SIZE) \
            .batch(BATCH_SIZE)
x_test = tf.data.Dataset.from_tensor_slices(x_test) \
            .map(normalize,num_parallel_calls=-1) \
            .batch(1)



init = tf.keras.initializers.glorot_uniform()
global_dicriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init),
    tf.keras.layers.Dense(1, kernel_initializer=init)
])

local_discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(512,1,activation=tf.nn.relu, kernel_initializer=init),
    tf.keras.layers.Conv2D(512,1,activation=tf.nn.relu, kernel_initializer=init),
    tf.keras.layers.Conv2D(1,1, kernel_initializer=init)
])

dicriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=init),
    tf.keras.layers.Dense(1, kernel_initializer=init)
])

encoder = BasicEncoder64x64(ENCODING_DIM)
model = DeepInfoMax(encoder,dicriminator,global_dicriminator,local_discriminator,ALPHA,BETA,GAMMA,
                    MUTUAL_INFO_TYPE,LEARNING_RATE)


date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(LOG_DIR, f"deep_infomax_{date}_{LOG_POSTFIX}")
summary_writer = tf.summary.create_file_writer(logdir)

train(model, x_train, EPOCHS,summary_writer)

### Testing

test_examples = []
for image in x_test.take(20):
    o, f = encoder(image)
    test_examples.append((image[0],o[0],f[0]))

test_images = []
test_global_scores = []
test_local_scores = []
for image in x_test:
    o, f = encoder(image)
    global_scores = [tf.reduce_mean(tf.abs(te[1] - o)).numpy() for te in test_examples]
    local_cores = [tf.reduce_mean(tf.abs(te[2] - f)).numpy() for te in test_examples]
    test_images.append(image[0])
    test_global_scores.append(global_scores)
    test_local_scores.append(local_cores)

test_global_scores_argsorted = np.argsort(np.array(test_global_scores).T)
test_local_scores_argsorted = np.argsort(np.array(test_local_scores).T)

for i in range(len(test_examples)):
    img = test_examples[i][0]
    top = np.concatenate(([img] + [np.ones_like(img) for i in range(10)]), axis=1)
    middle_global = np.concatenate([test_images[i] for i in test_global_scores_argsorted[i,:11]], axis=1)
    bottom_global = np.concatenate([test_images[i] for i in test_global_scores_argsorted[i,-11:]], axis=1)
    img_global = np.concatenate((top, middle_global, bottom_global), axis=0)
    middle_local = np.concatenate([test_images[i] for i in test_local_scores_argsorted[i,:11]], axis=1)
    bottom_local = np.concatenate([test_images[i] for i in test_local_scores_argsorted[i,-11:]], axis=1)
    img_local = np.concatenate((top, middle_local, bottom_local), axis=0)

    os.makedirs(os.path.join(logdir, "imgs"), exist_ok=True)
    save_image(img_global, os.path.join(logdir, f"imgs/test_{i}_global.png"))
    save_image(img_local, os.path.join(logdir, f"imgs/test_{i}_local.png"))

