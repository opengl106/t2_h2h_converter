import tensorflow as tf
from tensorflow import keras
from typing import List

from common import masked_accuracy, masked_loss
from data_load import load_vocab_list, load_vocab
from hyperparams import Hyperparams as hp

hanguls, hanjas = load_vocab_list()
hangul2idx, idx2hangul, hanja2idx, idx2hanja = load_vocab()

l_idx2hanja = keras.layers.StringLookup(
    vocabulary=hanjas,
    mask_token='', oov_token='[UNK]',
    invert=True
)
l_idx2hangul = keras.layers.StringLookup(
    vocabulary=hanguls,
    mask_token='', oov_token='[UNK]',
    invert=True
)

class H2HPredictor():
    """
    A easy-to-import wrapper around the original model.
    """
    def __init__(self) -> None:
        self.model = keras.models.load_model(hp.logdir + "/model.keras", custom_objects={
            'masked_accuracy': masked_accuracy,
            'masked_loss': masked_loss,
        })

    def __call__(self, text: str) -> str:
        # call model on single string.
        hangul_indice = [hangul2idx.get(char, 1) for char in text] + (hp.maxlen - len(text)) * [0]
        input_tensor = tf.reshape(tf.stack(hangul_indice), [1, hp.maxlen])
        logit = self.model(input_tensor)
        hanja_indice = tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32)[0]
        hanja = "".join([
            (idx2hanja[int(index)] if int(index) != 1 else text[i])
            for i, index in enumerate(hanja_indice)
            if int(hangul_indice[i]) != 0
        ])
        return hanja

    def convert(self, article: List[str]) -> List[str]:
        # predict on article consist of multiple strings.
        hangul_indices = [
            [hangul2idx.get(char, 1) for char in line] + (hp.maxlen - len(line)) * [0]
            for line in article
        ]
        input_tensor = tf.convert_to_tensor(
            [tf.stack(indice) for indice in hangul_indices]
        )
        logit = self.model.predict(input_tensor, batch_size=hp.batch_size)
        hanja_indices = tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32)
        mask = tf.cast(hanja_indices - 1 != 0, dtype=tf.int32)
        revmask = 1 - mask
        hanjas = l_idx2hanja(hanja_indices * mask)
        hanguls = l_idx2hangul(hangul_indices * revmask)
        array = tf.strings.join(tf.transpose(tf.strings.join([hanjas, hanguls])))
        return [item.decode() for item in array.numpy()]
