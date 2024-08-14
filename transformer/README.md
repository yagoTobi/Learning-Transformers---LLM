Building a Transformer using PyTorch following Umar Jamil Videos

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
    
