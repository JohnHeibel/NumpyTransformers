![[Attachments/Pasted image 20240412120937.png]]
# Architecture
- Tokenizer
	- Convert text into tokens
- Embedding layer
	- Converts tokens and positions of tokens into vectors
- Transformer layer
	- Do repeated transformations on the vector representations
	- Alternating attention and feedforward layers
	- Can be either encoder or decoder
		- In Attention Is All You Need, both are used
		- GPT models only use decoders
		- BERT only uses encoders
- Add and normalize
- Feedforward
- Masked Multi-Head Attention
- Un-embedding layer (optional)
	- if needed, convert final vectors back to probability distribution of tokens
# Tokenizer
- Encode text strings into "tokens" that are more easily represented in modeling
- 3 types tokenization, word level, subword level, and character level
	- "This" "is" "word" "level" "tokenization"
	- "This "is" "sub""word" "lev""el" "tok""eniz""ation"
	- "T""h""i""s" "i""s"...
	- These are not the real outputs from a tokenizer, just a demonstration. 
- Tokenizers have all sorts of architectural decisions and tradeoffs
	- Practically all current generation LLMs use subword tokenization
	- Byte Pair Encoding is one of the most popular tokenizers
	- Tokenization is extremely important to model generation, and its also capable of causing many errors
		- SolidGoldMagikarp is one infamous example
		- ![[Attachments/Pasted image 20240412125747.png|350]]
	- Another limitation of tokenization is that sometimes a character may not be in the vocabulary. You may have a weird unicode character that is not detected or was a malformed character in your training data. 
		- Sometimes, but not all the time, tokenizers implement byte fallback to reduce this issue
			- If a token is 
- Finally these tokens are converted to integers for simpler calculations
- These definition of tokenization only applies to language. Transformers work for many other modalities, like images, audio, and even time series. They all have their own methods of tokenizing pieces or patches of data.
# Embedding Layer
- ## Embedding
	- The first part of our Transformer that is trained
	- Now that we have our input string tokenized and converted into individual tokens, we want to convert these into (usually) higher-dimensional vectors
		- Why do we want that?
			- Its because that the closer two vectors are, the more "similar" they are
			- Queen is closer to King in embedding space than the token dog
			- This distance relationship can be used to have different words be valued more
	- Through the training process, our embedding layer is one of the sections that is changed
		- This increases the accuracy of our embedding space
		- The better trained our embeddings, the more accurate the similarities of tokens
	- After its trained, tokens are "projected" into embedding space through a map.
	- This projection is just a linear transformation
- ## Positional Encoding
	- All of these embeddings have to be concatenated into one value in some sort of way
		- Usually just done by addition
			- However, this will lead to use losing the value of the order of the tokens
		- These two sentences would have the same value if we added their embeddings, despite drastically different meanings
			- I just saw someone that killed it
			- I just killed someone that saw it
	- We need to somehow give an order to these tokens
	- All we do is add a unique vector to each token in order
		- token 1 + vector 1
		-  token 2 + vector 2
		- And so on
	- Through the training process, the transformer will learn that vector 1 is the first token, vector 2 is the second...
	- The actual formula used is
		- $PE_{pos,2i}=\sin(\frac{pos}{10000^{2i/d_{model}}})$
		- $PE_{pos,2i+1}=\cos(\frac{pos}{10000^{2i/d_{model}}})$
		- where pos is the position and i is the dimension
	- We use both sin and cos instead of just one function, as otherwise our values would repeat
	- This formula lets us have a unique value for every position that is bounded, so its easier for the model to learn
# Attention
- The most important part of a transformer network
	- This is the main mechanism introduced in the original paper Attention Is All You Need
- Core modeling capabilities come from these attention blocks
- Their main purpose is to give each token the possibility affect every other token
- Imagine the sequence of words "Adam is cool and he is an accountant"
	- Say you wanted to rank which words are important to the word he
		- "Adam is cool and <u>he</u> is an accountant"
		- Just looking at the nearby words, they aren't that useful
			- and, is
		- The words that actually matter are accountant, cool, Adam
			- We want to have an operation that changes the importance of each word in our input
- We essentialy want to model attention as it works in the human brain, where we focus on 
- ## Scaled Dot-Product Attention
	- ### Queries, Keys, and Values
		- Every input sequence is represented through 3 values, Q, K, V
		- These are described as the Query and Key Value pairs
			- 
		- X is the entire tokenized input sequence that has been converted into embeddings, and had the positional encoding added. All of these embeddings are then added together into one value, X
		- Queries
			- $Q = XW^Q$
		- Keys
			- $K = XW^K$
		- Values
			- $V = XW^V$
		- In each of these values, W is the weight, or the learned matrix by which we transform each value. 
		- These are some of the other values modified during the training process
	
		![[Attachments/Pasted image 20240415121150.png|400]]
		- We first perform a dot product multiplication on Q and a transposed version of K
		- Then we scale $QK^T$ by $\frac{1}{\sqrt{d_k}}$
			- where d is our dimensionality of our keys, values, and queries
			- This is done to prevent values from getting too small or too large, which causes problems when doing softmax
		- We then normalize the values using softmax
			- This turns our values into a probability distribution, and makes sure it all adds up to 1
			- $\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$
			- ![[Attachments/Pasted image 20240415132834.png|300]]
		- Finally, we multiply this value returned from softmax by V, scaling each vector by its corresponding score
		- $\text{Attention}(Q,K,V) = \text{softmax}(\dfrac{QK^T}{\sqrt{d_k}})V$
		- This vector returned from Attention is supposed to be a rich more more dynamic representation of the input
	- 
- ## Multi-Head Attention
	- Attention is very useful, but it is still limited. We only have one matrix for each of the QKV values that we can train, so there is only so much information we can extract from a single input.
	- We fix this problem by simply running many different attention mechanisms for a single QKV set. 
	- ![[Attachments/Pasted image 20240415122617.png|400]]
	- We have h different attention blocks, which all have been been trained differently to capture different aspects of the input
	- We perform multiple different Scaled Dot-Product Attentions in parallel
	- We then add all of those values together
	- Finally we multiply it by $W^O$, a trained matrix value
	- $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1 ... \text{head}_h)W^0$
	- $\text{head}_i = \text{Attention}(QW^i, KW^i, VW^i)$
	- For each of these heads, we reduce the dimensionality, so it costs about the same as a full attention head, but is much more effective.
- ## Masked Multi-Head Attention
	- 
	- ![[Attachments/Pasted image 20240415141940.png|400]]
	- 
- Overall, Self-Attention has a $O(n^2\cdot d)$ time complexity per layer, with sequential operations being $O(n)$
		- This is slower for large inputs
		- Recurrent neural networks have a complexity per layer of $O(n \cdot d^2)$ and a sequential Operation complexity of $O(n)$
		- The computational complexity for an input of size $n$ grows $n^2$ for transformers, and $n$ for recurrent models
		- ![[Attachments/Pasted image 20240415130716.png]]
# Add and Normalize
- 
# Feedforward Network
- Each layer of both encoders and decoders contains a fully connected Feedforward network
- This contains two linear transformations with a non-linearity in-between
- This feedforward network is applied to each position independently, so we can parallelilize the process
- The purpose is to add another layer of transformation that can be trained, and to introduce non-linearities. 
- ![[Attachments/Pasted image 20240415141013.png]]

# Transformer Layer
- In this project, we will be implementing an Encoder-Decoder transformer for machine translation tasks
- In this setup, the Encoder block will encode the initial sentence, and then the decoder produces the translated sentence from that encoding
## Encoder
- ![[Attachments/Pasted image 20240415134109.png]]
- We now have all of the knowledge we need to make the encoder block
- For the encoder block, we need to implement out attention blocks and our feed forward networks







https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021

https://arxiv.org/pdf/1706.03762.pdf
