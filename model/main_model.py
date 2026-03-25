"""
Seq2seq model with LSTM encoder-decoder and attention for PubMed summarization.
"""

import logging

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Embedding,
    Attention,
    Concatenate,
)

from model.config import MAX_ARTICLE_LEN, MAX_SUMMARY_LEN, EMBEDDING_DIM, LATENT_DIM

logger = logging.getLogger(__name__)


def build_model(vocab_size: int) -> Model:
    """
    Build encoder-decoder with attention.
    - Encoder: article sequence -> LSTM -> (outputs, state_h, state_c)
    - Decoder: abstract sequence (teacher forcing) -> LSTM(initial_state=encoder state)
    - Attention over encoder outputs, then concat with decoder output -> Dense(vocab_size).
    """
    # Encoder
    encoder_inputs = Input(shape=(MAX_ARTICLE_LEN,), name="encoder_input")
    enc_emb = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="encoder_embedding")(
        encoder_inputs
    )
    encoder_lstm = LSTM(
        LATENT_DIM,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
    )
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    # Decoder (teacher forcing)
    decoder_inputs = Input(shape=(MAX_SUMMARY_LEN,), name="decoder_input")
    dec_emb = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="decoder_embedding")(
        decoder_inputs
    )
    decoder_lstm = LSTM(
        LATENT_DIM,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Attention: query=decoder, value=encoder
    attention_layer = Attention(name="attention")
    attention_output = attention_layer(
        [decoder_outputs, encoder_outputs]
    )

    # Concatenate decoder output and attention, then project to vocab
    concat = Concatenate(name="concat")([decoder_outputs, attention_output])
    outputs = Dense(vocab_size, activation="softmax", name="output")(concat)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs, name="seq2seq_attention")
    logger.info("Model built: vocab_size=%d, encoder_len=%d, decoder_len=%d", vocab_size, MAX_ARTICLE_LEN, MAX_SUMMARY_LEN)
    return model
