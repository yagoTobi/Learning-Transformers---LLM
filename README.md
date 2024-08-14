Building a Transformer using PyTorch following Umar Jamil Videos

# Building the Transformer

- Training 
- Inferencing 
- Visualising the attention scores

We will build a translation model. 

Dataset: 
Opus Books - We're going to choose English to Spanish as I'm Spanish so I can verify the translation

1. Build the input embeddings
    Converts the original sentence into a vector of 512 dimensions. We transform the sentence into an input ID, which then passess to an embedding

2. Building the positional encodings
    Once we have our embeddings, we wish to convey the information of the word order by adding another vector to the embeddings layer. We add the positional encoding as the multi-headed attention is permutation invariant (meaning that the order does not affect the result) by default. 

> They add a dropout layer, but why? Adding a dropout to the encodings might be a bit counterintuitive since we're talking about positions. But by doing this, we're forcing the model to not rely too much on exact positional information. It will help the model to generalise better to sequences of different lengths. 

3. Layer Normalization
    The layer normalization serves to scale down the data, and keep the variance as constant as possible. We include two parameters in order to improve the models learning. One additive and the other multiplicative. 

4. Feed Forward Block 
    The feed forward block is actually two interconnected layers with a dropout layer in the middle, and a ReLu activation function. 
    It allows the model to perform more complex transformation on the data than only self-attention. As we're aiming to capture non-linear transformations, we need a ReLu, and its using this which adds sparsity to the network. ReLu doesn't saturate for positive values, eliminating vanishing gradient, like tanH or sigmoid. 

5. The Big Boy - Multi-Head Attention
    Takes the input of the encoder and duplicate it by 3 - Query, Key and Value. We then multiply each of these by the weight matrices W_Q, W_K and W_V. We need to think of Q as what's used to ask questions, K as what's used to answer questions and V to give information. 

    The W_Q, W_K, W_V matrices are at first randomly initialised and then tuned during the training process with back-prop

    This allows the model to transform the same input information in different ways for different purposes within the attenion mechanism. 
    Each head will have access to the full sentence but for different interpretations or aspects
    
# Training the transformer model on Corpus Dictionary Data