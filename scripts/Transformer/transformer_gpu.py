import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################## 1. Blocks ##############################

# Multi-Head Attention Block
class Multi_Head_Attention(nn.Module):  
    def __init__(self, d_model, num_heads):
        super(Multi_Head_Attention, self).__init__()  
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Parameter(self.initialize_weight(d_model, self.d_k * num_heads))
        self.W_K = nn.Parameter(self.initialize_weight(d_model, self.d_k * num_heads))
        self.W_V = nn.Parameter(self.initialize_weight(d_model, self.d_k * num_heads))
        self.W_O = nn.Parameter(self.initialize_weight(self.d_k * num_heads, d_model))

    def initialize_weight(self, n_in, n_out):
        return torch.nn.init.xavier_normal_(torch.empty(n_in, n_out))

    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]  

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device))

        if mask is not None:
            mask = mask.to(Q.device) 
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)  # Reshape
        return x.permute(0, 2, 1, 3)  # Transpose

    def concat_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()  # Transpose
        x = x.view(batch_size, -1, self.d_model)  # Reshape
        return x

    def forward(self, Q, K, V, mask=None):
        Q = Q.to(self.W_Q.device)
        K = K.to(self.W_Q.device)
        V = V.to(self.W_Q.device)
        if mask is not None:
            mask = mask.to(self.W_Q.device)

        batch_size = Q.size(0)

        Q = torch.matmul(Q, self.W_Q)
        Q = self.split_heads(Q, batch_size)

        K = torch.matmul(K, self.W_K)
        K = self.split_heads(K, batch_size)

        V = torch.matmul(V, self.W_V)
        V = self.split_heads(V, batch_size)

        attention_output = self.attention(Q, K, V, mask)

        attention_output = self.concat_heads(attention_output, batch_size)

        output = torch.matmul(attention_output, self.W_O)

        return output


# Feed Forward Block
class Feed_Forward(nn.Module): 
    def __init__(self, d_model, d_ff):
        super(Feed_Forward, self).__init__()  

        self.W_1 = nn.Parameter(self.initialize_weight(d_model, d_ff))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))  
        
        self.W_2 = nn.Parameter(self.initialize_weight(d_ff, d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model)) 

    def initialize_weight(self, n_in, n_out):
        return torch.nn.init.xavier_normal_(torch.empty(n_in, n_out))

    def forward(self, x):
        x = x.to(self.W_1.device)

        hidden = torch.matmul(x, self.W_1) + self.b_1.to(x.device)  # Move bias to the correct device
        hidden = F.relu(hidden)

        output = torch.matmul(hidden, self.W_2) + self.b_2.to(x.device)  # Move bias to the correct device

        return output


# Positional Encoding Block
class Positional_Encoding(nn.Module): 
    def __init__(self, d_model, max_seq_length):
        super(Positional_Encoding, self).__init__() 
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        self.register_buffer('positional_encoding', positional_encoding)

    def create_positional_encoding(self, max_seq_length, d_model):
        positional_encoding = torch.zeros((max_seq_length, d_model))

        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                positional_encoding[pos, i] = torch.sin(torch.tensor(pos) / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    positional_encoding[pos, i + 1] = torch.cos(torch.tensor(pos) / (10000 ** ((2 * (i + 1)) / d_model)))

        return positional_encoding

    def forward(self, x):
        seq_length = x.size(1) 

        assert seq_length <= self.max_seq_length, "Input sequence length exceeds maximum sequence length."

        return x + self.positional_encoding[:seq_length, :].unsqueeze(0).to(x.device)



############################## 2. Layers ##############################

# Layer Normalization
class Normalization(nn.Module):  
    def __init__(self, d_model, epsilon=1e-6):
        super(Normalization, self).__init__() 
        self.d_model = d_model
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(d_model))  
        self.beta = nn.Parameter(torch.zeros(d_model)) 

    def forward(self, x):
        x = x.to(self.gamma.device)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        norm_x = (x - mean) / (std + self.epsilon)

        output = self.gamma.to(x.device) * norm_x + self.beta.to(x.device)

        return output


# Encoder Layer
class EncoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, max_seq_length):
        super(EncoderLayer, self).__init__()  
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.multi_head_attention = Multi_Head_Attention(d_model, num_heads)  
        self.feed_forward = Feed_Forward(d_model, d_ff) 

        self.layer_norm_1 = Normalization(d_model)
        self.layer_norm_2 = Normalization(d_model)

    def forward(self, x, mask=None):
        x = x.to(self.layer_norm_1.gamma.device)

        attention_output = self.multi_head_attention.forward(x, x, x, mask)
        attention_output = attention_output.to(x.device)
        x = self.layer_norm_1.forward(x + attention_output)
        ff_output = self.feed_forward.forward(x)
        ff_output = ff_output.to(x.device)
        output = self.layer_norm_2.forward(x + ff_output)

        return output


# Encoder Block
class Encoder(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_seq_length):
        super(Encoder, self).__init__() 
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.positional_encoding = Positional_Encoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, max_seq_length) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.positional_encoding.forward(x)

        for layer in self.layers:
            x = layer.forward(x, mask)

        return x


# Decoder Layer
class DecoderLayer(nn.Module):  
    def __init__(self, d_model, num_heads, d_ff, max_seq_length):
        super(DecoderLayer, self).__init__() 
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
        x = x.to(self.layer_norm_1.gamma.device)
        encoder_output = encoder_output.to(x.device)

        attention_output = self.masked_multi_head_attention.forward(x, x, x, mask)
        attention_output = attention_output.to(x.device)
        x = self.layer_norm_1.forward(x + attention_output)

        encoder_attention_output = self.encoder_decoder_attention.forward(x, encoder_output, encoder_output, encoder_decoder_mask)
        encoder_attention_output = encoder_attention_output.to(x.device)
        x = self.layer_norm_2.forward(x + encoder_attention_output)

        ff_output = self.feed_forward.forward(x)
        ff_output = ff_output.to(x.device)

        output = self.layer_norm_3.forward(x + ff_output)

        return output


# Decoder Block
class Decoder(nn.Module):  # Inherit from nn.Module
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_seq_length):
        super(Decoder, self).__init__()  # Initialize nn.Module

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.positional_encoding = Positional_Encoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, max_seq_length) 
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, mask=None, encoder_decoder_mask=None):
        # Add positional encoding to the input tensor
        x = self.positional_encoding.forward(x)

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer.forward(x, encoder_output, mask, encoder_decoder_mask)

        return x



############################## 3. Transformer ##############################

class Transformer(nn.Module):  # Inherit from nn.Module
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_seq_length):
        super(Transformer, self).__init__()  # Initialize nn.Module

        # Initialize Encoder and Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, max_seq_length)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, max_seq_length)

        # Final linear layer to map the decoder output to the vocabulary size (or output dimension)
        self.W_out = self.initialize_weight(d_model, d_model)

    # Xavier initialization for the weight matrix
    def initialize_weight(self, n_in, n_out):
        return torch.nn.init.xavier_normal_(torch.empty(n_in, n_out, device=device))

    def forward(self, encoder_input, decoder_input, encoder_mask=None, decoder_mask=None, encoder_decoder_mask=None):
        encoder_output = self.encoder.forward(encoder_input, encoder_mask)
        decoder_output = self.decoder.forward(decoder_input, encoder_output, decoder_mask, encoder_decoder_mask)

        W_out = self.W_out.to(decoder_output.device)

        # Final linear projection
        output = torch.matmul(decoder_output, W_out)

        return output


