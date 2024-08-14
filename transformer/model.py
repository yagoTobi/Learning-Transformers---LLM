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
        mean = x.mean(dim = -1, keepDim=True)
        std = x.std(dim = -1, keepDim=True) # dim = -1 means the last dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.bias # Formula for normalisation with parameters
    
class FeedForwardBlock(nn.Module):
    # Two linear transformations with a ReLU activation function
    # FFN(x) = max(0, xW1 + b1)W2 + b2
    # They use different parameters from layer to layer, with input and output of d_model 512
    # and the inner layer of d_ff 2048
    # Structure: d_model -> d_ff -> d_model
    # Detail: (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model)
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # First layer - w1, b1 (bias already included by Torch)
        self.dropout = nn.Dropout(dropout) # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model) # Second layer - w2, b2

    def forward(self, x): # Forward pass of the network
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # The syntax appears to be almost too simple?

# Time for the Big Boy - The Multi-Head Attention Block
class MultiHeadAttentionBlock(nn.Module):
    # MHA requires that we take the input matrix of seq by d_model
    # then duplicate it into three matrices. 
    # Multiply by the weight matrices to learn and then split the output into the heads 
    # Finally, concatenate the heads and multiply by the output weight matrix
    # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V <- Multi Head
    # The output of the MHA is the same size as the input - MH(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O
    def __init__(self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # The number of heads we require
        # ! We need to ensure that the dimension of the head is divisible by the number of heads
        assert d_model % h == 0, "Number of heads must be divisible by the model dimension"
        self.d_k = d_model // h
        
        # ? Weight matrices for Q, K and V - Output must be seq by d_model
        # ? We declare it a linear as it will be refined
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # * Output weight matrix -> Concatenated heads Tensor (seq, h * d_k) (Number of heads * dimension of the split)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod #Call the function without having an instance of the MHA class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # Get the dimension of the key
        # * Formula is Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # ? Before applying the softmax, we need to apply the mask if present
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) # Replace the masked values with a large negative number 
        
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # * Return the output of the MHA and the attention scores for visualization 
        return (attention_scores @ value), attention_scores 


    def forward(self, q, k, v, mask): 
        #*-Remember that in the decoder, we have the masked MHA, that's why we can include it here 
        #*-So that they don't learn the next word interaction
        # ! Step 1: Apply the linear transformations to the query, key and value tensors
        query = self.w_q(q) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        key = self.w_k(k)   # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        value = self.w_v(v) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        
        # ! Step 2: Split the query, key and value tensors into self.h heads
        # ? Reshaping and transposing the query tensor to prepare it for the split
        # * query.shape[0] is the batch size
        # * query.shape[1] is the sequence length
        # * self.h is the number of heads
        # * self.d_k is the dimension of the split (d_model // h)
        # ? This transformation alone would leave us with: (Batch, Seq, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        # Imagine we have a batch of sentences, and we want to apply different attention patters to each sentence. 
        # By having the dimension come before the sequence length, we are saying: For each sentence, apply all attention patterns simultaneously
        # Instead of for each attention pattern, process all sentences sequentially. So we're enabling parallel processing. 
        # ! So we must pass from (Batch_size, Seq_length, Num_Heads, head_dim) to (Batch_Size, Num_Heads, Seq_length, head_dim)
        query = query.transpose(1,2) # * Each head needs to see the full sentence. We split up the embedding. 
        # ? The same transformation is applied to the key and value tensors
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # ! Step 3: Apply the scaled dot-product attention mechanism to the query, key and value tensors + formula
        # So we now have the heads calculated for the query, key and value
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # ! Step 4: Concatenate the heads and apply the output weight matrix
        # ? We need to reshape the output tensor to concatenate the heads 
        # ? (Batch, h, seq, d_k) --> (Batch, seq, h, d_k) --> (Batch, seq, d_model)

        # A tensor is contiguous if it's stored in a single block of memory. When we transpose
        # the tensor, the result might not be contiguous so, we ensure that it will be stored by using that method. 
        # Then we reshape the tensor to keep the batch size dimension, we infer the dimension (sequence length, and then we concatenate all heads)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h * self.d_k)

        # ? Apply the output weight matrix to the concatenated heads
        return self.w_o(x)