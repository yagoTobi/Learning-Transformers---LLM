import torch 
import torch.nn as nn
import math

# - Wow so this is actually using the reference paper and we're following through each component. 

class InputEmbeddings(nn.Module):
    # Constructor of the model takes in the vocab size for d_model - Transform to the vector
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # Vector of dimension d - In this case 512
        self.vocab_size = vocab_size
        self.embedding = nn.embedding(vocab_size, d_model) # Wait, wtf, that's it??
    # PyTorch provides the same vector every time to map between numbers and vector 512 -> d_model

    def forward(self, x): 
        # In the embedding layers, we multiply those weights by sqrt d_model
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Part II - Building the positional encodings
class PositionalEncoding(nn.Module):
    # So for the positional encoder, we're going to pass the embedding size
    # The sequence length and the dropout rate
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # Max length of the sentence
        self.dropout = dropout # To avoid overfitting

        # Create a positional encoding matrix - Matrix of shape (seq_len, d_model)
        # Seq length which covers the sentence length, and d model to match the embedding size
        # We must use the formulas taken from the paper to recreate the vector of size 512
        pe = torch.zeros(seq_len, d_model) # Initialise the matrix with zeros
        # There are two different formulas for the positional encoding - One for even words, one for odd words
        
        # Create a 1D tensor from 0 to seq_len - 1 and specify their type as float
        # unsqueeze(1) adds a dimension to the tensor
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (Seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Special functions which convey the position

        # sine function for even positions
        pe[:,0::2] = torch.sin(position * div_term)
        # cosine function for odd positions
        pe[:1,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the buffer to the model - Save the buffer 
        self.register_buffer('pe', pe) 

        # So now, we have our formulas saved for the structure 

    def forward(self, x): # Defines the forward pass of the network
        # Adding the positional encoding vector to the input embedding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Slices the positional encoder such that if fits the shape of x. 
        return self.dropout(x) # Add dropout to the positional encoding to prevent overfitting
    

class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float = 10**-6) -> None:
        # x_normalised = x - mean / sqrt(variance + epsilon)
        # Regularisation parameters
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative parameter 
        self.bias = nn.Parameter(torch.zeros(1)) # Additive parameter

    def forward(self,x): # Remember we need to calculate the mean and variance for each layer
        mean = x.mean(dim = -1, keepDim)