import tensorflow as tf
from enum import Enum


# Note all functions below represent negative scores.
# Thus minimization of a score means maximization of the corresponding entropy

class MutualInfoType(Enum):
    JS = "Jensen-Shannon"
    DV = "Donsker-Varadhan"


def get_mutual_info_scorer(type: MutualInfoType):
    if type is MutualInfoType.JS:
        return neg_jensen_shannon
    if type is MutualInfoType.DV:
        return neg_donsker_varadhan
    raise RuntimeError("Info scorer is not implemented")

@tf.function
def neg_jensen_shannon(pos_examples,neg_examples):
    Ej = tf.math.reduce_mean(pos_examples)
    Em = tf.math.reduce_mean(tf.math.exp(neg_examples))
    Em = tf.math.log(Em)
    return Em - Ej

@tf.function
def neg_donsker_varadhan(pos_examples,neg_examples):
    Ep = tf.math.reduce_mean(tf.nn.softplus(pos_examples))
    Ep_ = tf.math.reduce_mean(tf.nn.softplus(-neg_examples))
    return Ep_ + Ep

