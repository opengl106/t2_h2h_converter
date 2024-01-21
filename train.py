# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Training.
'''

import codecs
import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data_load import load_vocab, load_data
from hyperparams import Hyperparams as hp

# Load vocab
hangul2idx, idx2hangul, hanja2idx, idx2hanja = load_vocab()

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

class H2HModel():
    """
    A tensorflow 2 version of the original model.
    """
    def __init__(self, model=None):
        if model:
            self.model = model
            return None

        self.model = keras.Sequential([
            keras.layers.Input(shape=[hp.maxlen], name="hangul_sent", dtype=tf.int32),
            # Mask zeros in the variable length input sequences.
            keras.layers.Embedding(len(hangul2idx), hp.hidden_units, mask_zero=True),
            keras.layers.Bidirectional(
                layer=keras.layers.GRU(
                    hp.hidden_units,
                    return_sequences=True,
                    return_state=False,
                ),
                merge_mode='concat',
            ),
            keras.layers.Dense(len(hanja2idx)),
        ])

        optimizer = keras.optimizers.Adam(
            learning_rate=hp.learning_rate,
        )
        metrics = [
            masked_accuracy,
            masked_loss,
        ]
        self.model.compile(
            optimizer=optimizer,
            loss=masked_loss,
            metrics=metrics,
        )

        return None

    def predict(self, data, dtype=tf.int32, batch_size=hp.batch_size):
        logit = self.model.predict(data, batch_size=batch_size)
        pred = tf.cast(tf.argmax(logit, axis=-1), dtype=dtype)
        return pred


if __name__ == '__main__':
    # Data loading
    X_train, Y_train = load_data(mode="train")
    train_data_x = tf.data.Dataset.from_tensor_slices(X_train)
    train_data_y = tf.data.Dataset.from_tensor_slices(Y_train)
    dataset = tf.data.Dataset.zip((train_data_x, train_data_y)).batch(hp.batch_size)

    # Model loading
    if os.path.exists(hp.logdir + "/model.keras"):
        print(f"Loading dataset from {hp.logdir}/model.keras")
        model = keras.models.load_model(hp.logdir + "/model.keras", custom_objects={
            'masked_accuracy': masked_accuracy,
            'masked_loss': masked_loss,
        })
        keras.backend.set_value(model.optimizer.learning_rate, hp.learning_rate)
        m = H2HModel(model)
    else:
        m = H2HModel()

    # Training
    checkpoint_path = hp.logdir + "/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    history = m.model.fit(
        dataset,
        epochs=hp.num_epochs,
        callbacks=[cp_callback],
    )

    if not os.path.exists(hp.logdir):
        os.mkdir(hp.logdir)
    m.model.save(hp.logdir + "/model.keras")

    # Logging
    hist_df = pd.DataFrame(history.history)
    hist_json_file = hp.logdir + "/history.json"
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    hist_csv_file = hp.logdir + "/history.csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Evaluating
    x_val, y_val = load_data(mode="val")
    val_data_x = tf.data.Dataset.from_tensor_slices(x_val)
    val_data_y = tf.data.Dataset.from_tensor_slices(y_val)
    preds = m.predict(x_val)
    pred_data = tf.data.Dataset.from_tensor_slices(preds)
    with codecs.open(hp.logdir + "/eval.txt", 'w', 'utf-8') as fout:
        for xx, yy, pred in tf.data.Dataset.zip(val_data_x, val_data_y, pred_data): # sentence-wise
            inputs, expected, got = [], [], []
            for xxx, yyy, ppp in zip(xx, yy, pred):  # character-wise
                if int(xxx)==0: break
                inputs.append(idx2hangul[int(xxx)])
                expected.append(idx2hanja[int(yyy)] if int(yyy)!=1 else idx2hangul[int(xxx)])
                got.append(idx2hanja[int(ppp)] if int(ppp) != 1 else idx2hangul[int(xxx)])
            fout.write(u"* Input   : {}\n".format("".join(inputs)))
            fout.write(u"* Expected: {}\n".format("".join(expected)))
            fout.write(u"* Got     : {}\n".format("".join(got)))
            fout.write("\n")
