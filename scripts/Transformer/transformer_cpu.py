import numpy as np

############################## 1. Blocks ##############################
# TODO: KV Cache
class Multi_Head_Attention:
    def __init__(self, d_model, num_heads):
        """       
        Args:
            d_model (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight Matrices for Queries (Q), Keys (K), Values (V), and Output (O)
        self.W_Q = self.initialize_weight(d_model, self.d_k * num_heads)
        self.W_K = self.initialize_weight(d_model, self.d_k * num_heads)
        self.W_V = self.initialize_weight(d_model, self.d_k * num_heads)
        self.W_O = self.initialize_weight(self.d_k * num_heads, d_model)

    # Use Xavier Initialization with Normal Distribution for Weight Matrices
    def initialize_weight(self, n_in, n_out):
        stddev = np.sqrt(2 / (n_in+n_out))
        weight = np.random.normal(0, stddev, (n_in, n_out))

        return weight       

    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]  # d_k is the last dimension of Q (i.e., dimensionality of keys/queries)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # Shape: (batch_size, num_heads, seq_length, seq_length)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Shape: (batch_size, num_heads, seq_length, seq_length)

        attention_output = np.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_length, d_k)
        # print("Shape of attention_output after applying attention weights:", attention_output.shape)

        return attention_output

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k) # Shape before transpose: (batch_size, num_heads, seq_length, d_k)
        return x.transpose(0, 2, 1, 3) # Shape after transpose: (batch_size, seq_length, num_heads, d_k)

    def concat_heads(self, x, batch_size):
        # print("Shape of x before transpose in concat_heads:", x.shape)
        # Shape: (batch_size, num_heads, seq_length, d_k) -> (batch_size, seq_length, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # print("Shape of x after transpose:", x.shape)
        # Reshape to (batch_size, seq_length, d_model) where d_model = num_heads * d_k
        x = x.reshape(batch_size, -1, self.d_model)
        # print("Shape of x after reshape:", x.shape)

        return x

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        Q = np.dot(Q, self.W_Q)
        Q = self.split_heads(Q, batch_size)
        # print("Shape of Q after split_heads:", Q.shape)

        K = np.dot(K, self.W_K)
        K = self.split_heads(K, batch_size)
        # print("Shape of K after split_heads:", K.shape)

        V = np.dot(V, self.W_V)
        V = self.split_heads(V, batch_size)
        # print("Shape of V after split_heads:", V.shape)

        attention_output = self.attention(Q, K, V, mask)
        # print("Shape of attention_output:", attention_output.shape)

        # Concatenate heads
        attention_output = self.concat_heads(attention_output, batch_size)
        # print("Shape of attention_output after concat_heads:", attention_output.shape)

        # Final linear projection
        output = np.dot(attention_output, self.W_O)
        return output
    
class Feed_Forward:
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model (int): The input and output dimensionality (usually the same).
            d_ff (int): The hidden layer dimensionality (usually larger than d_model).
        """
        self.d_model = d_model
        self.d_ff = d_ff

        self.W_1 = self.initialize_weight(d_model, d_ff)
        self.b_1 = np.zeros((1, d_ff))

        self.W_2 = self.initialize_weight(d_ff, d_model)
        self.b_2 = np.zeros((1, d_model))

    # Use Xavier Initialization with Normal Distribution for Weight Matrices
    def initialize_weight(self, n_in, n_out):
        stddev = np.sqrt(2 / (n_in+n_out))
        weight = np.random.normal(0, stddev, (n_in, n_out))

        return weight 

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        hidden = np.dot(x, self.W_1) + self.b_1
        hidden = self.relu(hidden)

        output = np.dot(hidden, self.W_2) + self.b_2
        return output


class Positional_Encoding:
    def __init__(self, d_model, max_seq_length):
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Initialize the Positional Encoding Matrix
        self.positional_encoding = self.create_positional_encodeing(max_seq_length, d_model)

    def create_positional_encodeing(self, max_seq_length, d_model):
        positional_encoding = np.zeros((max_seq_length, d_model))

        for pos in range(max_seq_length):
            for i in range(0, d_model, 2): # Step over dimensions 2 at a time (even indices for sine, odd for cosine)
                if i + 1 < d_model:
                    positional_encoding[pos, i+1] = np.cos(pos / (10000**((i+1)/d_model)))

        return positional_encoding

    def forward(self, x):
        batch_size, seq_length, d_model = x.shape

        assert seq_length <= self.max_seq_length, "Input sequence length exceeds maximum sequence length."

        return x + self.positional_encoding[:seq_length, :]

############################## 2. Layers ##############################

############### 2.1 Layer Normalization ###############

class Normalization:
    def __init__(self, d_model, epsilon=1e-6):
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        norm_x = (x-mean) / (std+self.epsilon)

        return self.gamma * norm_x + self.beta

############### 2.2 Encoder ###############

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, max_seq_length):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.multi_head_attention = Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = Feed_Forward(d_model, d_ff)

        self.layer_norm_1 = Normalization(d_model)
        self.layer_norm_2 = Normalization(d_model)

    def forward(self, x, mask=None):
        attention_output = self.multi_head_attention.forward(x, x, x, mask)
        x = self.layer_norm_1.forward(x+attention_output)

        ff_output = self.feed_forward.forward(x)
        output = self.layer_norm_2.forward(x+ff_output)

        return output

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, max_seq_length):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.positional_encoding = Positional_Encoding(d_model, max_seq_length)
        self.multi_head_attention = Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = Feed_Forward(d_model, d_ff)

        self.layer_norm_1 = Normalization(d_model)
        self.layer_norm_2 = Normalization(d_model)

    def forward(self, x, mask=None):
        x = self.positional_encoding.forward(x)
        attention_output = self.multi_head_attention.forward(x, x, x, mask)
        x = self.layer_norm_1.forward(x+attention_output)

        ff_output = self.feed_forward.forward(x)
        output = self.layer_norm_2.forward(x+ff_output)

        return output

class Encoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_seq_length):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.positional_encoding = Positional_Encoding(d_model, max_seq_length)

        self.layers = [EncoderLayer(d_model, num_heads, d_ff, max_seq_length) for _ in range(num_layers)]

    def forward(self, x, mask=None):
        x = self.positional_encoding.forward(x)

        for layer in self.layers:
            x = layer.forward(x, mask)

        return x


############### 2.3 Decoder ###############

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, max_seq_length):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.masked_multi_head_attention = Multi_Head_Attention(d_model, num_heads)
        self.encoder_decoder_attention = Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = Feed_Forward(d_model, d_ff)

        self.layer_norm_1 = Normalization(d_model)
        self.layer_norm_2 = Normalization(d_model)
        self.layer_norm_3 = Normalization(d_model)

    def forward(self, x, encoder_output, mask=None, encoder_decoder_mask=None):
        attention_output = self.masked_multi_head_attention.forward(x, x, x, mask)
        x = self.layer_norm_1.forward(x + attention_output)

        encoder_attention_output = self.encoder_decoder_attention.forward(x, encoder_output, encoder_output, encoder_decoder_mask)
        x = self.layer_norm_2.forward(x + encoder_attention_output)

        ff_output = self.feed_forward.forward(x)
        output = self.layer_norm_3.forward(x + ff_output)

        return output

class Decoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_seq_length):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.positional_encoding = Positional_Encoding(d_model, max_seq_length)

        self.layers = [DecoderLayer(d_model, num_heads, d_ff, max_seq_length) for _ in range(num_layers)]

    def forward(self, x, encoder_output, mask=None, encoder_decoder_mask=None):
        x = self.positional_encoding.forward(x)

        for layer in self.layers:
            x = layer.forward(x, encoder_output, mask, encoder_decoder_mask)

        return x

############################## 3. Transformer ##############################

class Transformer:
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_seq_length):
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, max_seq_length)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, max_seq_length)

        # Final linear layer to map decoder output to vocabulary size
        self.W_out = self.initialize_weight(d_model, d_model)

    def initialize_weight(self, n_in, n_out):
        stddev = np.sqrt(2 / (n_in+n_out))
        weight = np.random.normal(0, stddev, (n_in, n_out))

        return weight 
    
    def forward(self, encoder_input, decoder_input, encoder_mask=None, decoder_mask=None, encoder_decoder_mask=None):
        encoder_output = self.encoder.forward(encoder_input, encoder_mask)
        
        decoder_output = self.decoder.forward(decoder_input, encoder_output, decoder_mask, encoder_decoder_mask)

        output = np.dot(decoder_output, self.W_out)

        return output

