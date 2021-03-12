import tensorflow as tf
from .mututal_info import MutualInfoType, get_mutual_info_scorer

class DeepInfoMax(object):
    def __init__(self,encoder, discriminator, global_info_discriminator, local_info_discriminator,
                 alpha=1.0, beta=1.0, gamma=0.1, mutual_info_type=MutualInfoType.DV,
                 learning_rate=1e-4):
        self.encoder = encoder
        self.discriminator = discriminator
        self.global_info_discriminator = global_info_discriminator
        self.local_info_discriminator = local_info_discriminator
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mutual_info = get_mutual_info_scorer(mutual_info_type)
        self.global_info_discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.local_info_discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def train_step(self, input):
        with tf.GradientTape(persistent=True) as tape:
            out, features = self.encoder(input)

            # Concat'n'convolve architeture

            batch_size = tf.shape(out)[:1]
            feature_flatten = tf.reshape(features, tf.concat([batch_size, tf.expand_dims(tf.reduce_prod(tf.shape(features)[1:]),0)],axis=-1))

            global_pos_input = tf.concat([out,feature_flatten],axis=-1)
            global_neg_input = tf.concat([out,tf.concat([feature_flatten[1:],feature_flatten[:1]],axis=0)],axis=-1)

            global_pos_out = self.global_info_discriminator(global_pos_input)
            global_neg_out = self.global_info_discriminator(global_neg_input)

            sh_o = tf.shape(out)
            sh_o = tf.concat([sh_o[:1],tf.ones(tf.shape(tf.shape(features)[1:-1]),dtype=tf.int32),sh_o[1:]],axis=-1)
            out = tf.reshape(out, sh_o)
            sh_o = tf.concat([[1], tf.shape(features)[1:-1], [1]], axis=-1)
            out = tf.tile(out,sh_o)

            local_pos_input = tf.concat([out, features],axis=-1)
            local_neg_input = tf.concat([out, tf.concat([features[1:],features[:1]],axis=0)], axis=-1)
            local_den = tf.cast(tf.reduce_prod(tf.shape(features)[1:3]), dtype=self.encoder.dtype)

            local_pos_out = self.local_info_discriminator(local_pos_input)
            local_neg_out = self.local_info_discriminator(local_neg_input)

            priory_pos = self.discriminator(out)
            priory_neg = self.discriminator(tf.random.uniform(tf.shape(out)))

            disc_loss_enc = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(priory_neg), logits=priory_neg)
            disc_loss_enc = tf.math.reduce_mean(disc_loss_enc)

            disc_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(priory_pos), logits=priory_pos)
            disc_loss_real = tf.math.reduce_mean(disc_loss_real)
            disc_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(priory_neg), logits=priory_neg)
            disc_loss_fake = tf.math.reduce_mean(disc_loss_fake)

            global_neg_mi = self.mutual_info(global_pos_out, global_neg_out)
            local_neg_mi = self.mutual_info(local_pos_out, local_neg_out)

            loss_encoder = self.alpha * global_neg_mi + \
                   self.beta * local_neg_mi / local_den + \
                   self.gamma * disc_loss_enc
            loss_discriminator = disc_loss_real + disc_loss_fake

        global_info_discriminator_grads = tape.gradient(loss_encoder, self.global_info_discriminator.trainable_variables)
        self.global_info_discriminator_optimizer.apply_gradients(zip(global_info_discriminator_grads, self.global_info_discriminator.trainable_variables))

        local_info_discriminator_grads = tape.gradient(loss_encoder, self.local_info_discriminator.trainable_variables)
        self.local_info_discriminator_optimizer.apply_gradients(zip(local_info_discriminator_grads, self.local_info_discriminator.trainable_variables))

        encoder_grads = tape.gradient(loss_encoder, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_grads,self.encoder.trainable_variables))

        discriminator_grads = tape.gradient(loss_discriminator, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads,self.discriminator.trainable_variables))
        return -global_neg_mi, -local_neg_mi, loss_encoder, loss_discriminator

