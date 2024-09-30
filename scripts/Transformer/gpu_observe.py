import torch
from joblib import Parallel, delayed
from transformer_gpu import Transformer

def test_transformer(transformer, encoder_input, decoder_input):
    transformer_output = transformer.forward(encoder_input, decoder_input)
    return transformer_output

def monitor_gpu():
    print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")

def process_round(transformer, encoder_input, decoder_input, round_num):
    print(f"Round {round_num + 1}/{num_rounds}")
    test_transformer(transformer, encoder_input, decoder_input)
    monitor_gpu()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    d_model = 4096  # Dimensionality of embeddings
    num_heads = 32  # Number of attention heads
    d_ff = 16384  # Feed-forward network dimensionality
    num_layers = 24  # Number of layers in the encoder and decoder
    max_seq_length = 2048  # Maximum sequence length
    batch_size = 8  # Batch size
    seq_length = 256  # Input sequence length

    transformer = Transformer(d_model, num_heads, d_ff, num_layers, max_seq_length)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        transformer = torch.nn.DataParallel(transformer)

    transformer = transformer.cuda()

    encoder_input = torch.rand(batch_size, seq_length, d_model).cuda()
    decoder_input = torch.rand(batch_size, seq_length, d_model).cuda()

    num_rounds = 100

    Parallel(n_jobs=16)(delayed(process_round)(transformer, encoder_input, decoder_input, round) for round in range(num_rounds))
