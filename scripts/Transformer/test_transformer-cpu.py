
import numpy as np

from transformer_cpu import (
    Multi_Head_Attention, 
    Feed_Forward, 
    Positional_Encoding, 
    Normalization, 
    EncoderLayer, 
    Encoder, 
    DecoderLayer, 
    Decoder, 
    Transformer
)

# 1: Multi_Head_Attention
def test_multi_head_attention():
    d_model = 8  # Dimensionality of embeddings
    num_heads = 2  # Number of attention heads
    seq_length = 5  # Sequence length
    batch_size = 2  # Batch size

    mha = Multi_Head_Attention(d_model, num_heads)

    Q = np.random.rand(batch_size, seq_length, d_model)
    K = np.random.rand(batch_size, seq_length, d_model)
    V = np.random.rand(batch_size, seq_length, d_model)

    attention_output = mha.forward(Q, K, V)
    
    print("Multi-Head Attention Output:\n", attention_output)

# 2: Feed_Forward
def test_feed_forward():
    d_model = 8  # Dimensionality of input/output
    d_ff = 16  # Dimensionality of feed-forward hidden layer
    batch_size = 2
    seq_length = 5

    ff = Feed_Forward(d_model, d_ff)

    x = np.random.rand(batch_size, seq_length, d_model)

    ff_output = ff.forward(x)
    
    print("Feed Forward Network Output:\n", ff_output)

# 3: Positional_Encoding
def test_positional_encoding():
    d_model = 8  # Dimensionality of embeddings
    max_seq_length = 10  # Maximum sequence length
    batch_size = 2
    seq_length = 5

    pe = Positional_Encoding(d_model, max_seq_length)

    x = np.random.rand(batch_size, seq_length, d_model)

    pos_encoded_output = pe.forward(x)

    print("Positional Encoding Output:\n", pos_encoded_output)

# 4: Normalization
def test_normalization():
    d_model = 8  # Dimensionality of input/output
    batch_size = 2
    seq_length = 5

    norm = Normalization(d_model)

    x = np.random.rand(batch_size, seq_length, d_model)

    norm_output = norm.forward(x)

    print("Normalization Output:\n", norm_output)

# 5: EncoderLayer
def test_encoder_layer():
    d_model = 8  # Dimensionality of embeddings
    num_heads = 2  # Number of attention heads
    d_ff = 16  # Feed-forward network dimensionality
    max_seq_length = 10  # Maximum sequence length
    batch_size = 2
    seq_length = 5

    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, max_seq_length)

    x = np.random.rand(batch_size, seq_length, d_model)

    encoder_output = encoder_layer.forward(x)

    print("Encoder Layer Output:\n", encoder_output)

# 6: Encoder
def test_encoder():
    d_model = 8
    num_heads = 2
    d_ff = 16
    num_layers = 3
    max_seq_length = 10
    batch_size = 2
    seq_length = 5

    encoder = Encoder(d_model, num_heads, d_ff, num_layers, max_seq_length)

    x = np.random.rand(batch_size, seq_length, d_model)

    encoder_output = encoder.forward(x)

    print("Encoder Output:\n", encoder_output)

# 7: DecoderLayer
def test_decoder_layer():
    d_model = 8
    num_heads = 2
    d_ff = 16
    max_seq_length = 10
    batch_size = 2
    seq_length = 5

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff, max_seq_length)

    x = np.random.rand(batch_size, seq_length, d_model)
    encoder_output = np.random.rand(batch_size, seq_length, d_model)

    decoder_output = decoder_layer.forward(x, encoder_output)

    print("Decoder Layer Output:\n", decoder_output)

# 8: Decoder
def test_decoder():
    d_model = 8
    num_heads = 2
    d_ff = 16
    num_layers = 3
    max_seq_length = 10
    batch_size = 2
    seq_length = 5

    decoder = Decoder(d_model, num_heads, d_ff, num_layers, max_seq_length)

    x = np.random.rand(batch_size, seq_length, d_model)
    encoder_output = np.random.rand(batch_size, seq_length, d_model)

    decoder_output = decoder.forward(x, encoder_output)

    print("Decoder Output:\n", decoder_output)

# 9: Transformer
def test_transformer():
    d_model = 8  # Dimensionality of embeddings
    num_heads = 2  # Number of attention heads
    d_ff = 16  # Feed-forward network dimensionality
    num_layers = 3  # Number of layers in the encoder and decoder
    max_seq_length = 10  # Maximum sequence length
    batch_size = 2  # Batch size
    seq_length = 5  # Input sequence length

    transformer = Transformer(d_model, num_heads, d_ff, num_layers, max_seq_length)

    encoder_input = np.random.rand(batch_size, seq_length, d_model)
    decoder_input = np.random.rand(batch_size, seq_length, d_model)

    transformer_output = transformer.forward(encoder_input, decoder_input)

    print("Transformer Output:\n", transformer_output)


if __name__ == "__main__":
    # test_multi_head_attention()
    # test_feed_forward()
    # test_positional_encoding()
    # test_normalization()
    # test_encoder_layer()
    # test_encoder()
    # test_decoder_layer()
    # test_decoder()
    test_transformer()
