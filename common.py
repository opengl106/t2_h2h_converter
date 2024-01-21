import tensorflow as tf
from tensorflow import keras


def masked_loss(label, logit):
    loss_indexwise = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=keras.losses.Reduction.NONE,
    )
    unmasked = loss_indexwise(label, logit)

    # Mask zeros where length of expected label sequences is variable.
    mask = tf.cast(label != 0, dtype=unmasked.dtype)
    loss = unmasked * mask

    # Probability normalization
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(label, logit):
    pred = tf.cast(tf.argmax(logit, axis=-1), dtype=label.dtype)
    unmasked = tf.cast(pred == label, dtype=tf.int32)

    mask = tf.cast(label != 0, dtype=tf.int32)
    hits = unmasked * mask

    return tf.reduce_sum(hits) / tf.reduce_sum(mask)
