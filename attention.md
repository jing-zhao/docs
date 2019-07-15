# Transformer and Self Attention
A reading summary of [https://jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer).

**_Keys to Understand_**
> **_Encoder_**
> - Encoder encodes **ALL** input terms in parallel (thru Self-Attention); So only one pass of encoding process is needed;
> - **Query**, **Key**, **Value** vectors (calculated by input vector * corresponding trained matrix) lie at the heart of **_Self-Attention_**;

> **_Decoder_**
> - Decoder only decodes for **ONE** output position at a time; So multiple passes of decoding process is needed (one pass per output position);
> - **_Encoder-Decoder-Attention_** layer in Decoder only needs to calculate Query vector (so to only train Query matrix) and directly uses Key vectors and Value vectors from Encoder output;



## Encoder and Decoder
*Animations are embedded from the original post throught the doc.*

**Figure 1**
![](https://jalammar.github.io/images/t/transformer_decoding_1.gif)

**Figure 2**
![](https://jalammar.github.io/images/t/Transformer_decoder.png)

> **_TLDR:_**
> - Enoder and Decoder blocks are all identical to their kinds, not only w.r.t block composition but also w.r.t function (e.g: All Encoders encode whole input, and all Decoders decode for one particular output position at one pass). So **# of Encoder/Decoders are just H-params**.
> - Encoders run only **ONE** pass while Decoders run **N** passes (N = # of output positions);
> - Figure 2 illustrates simplied view of Encoder/Decoder composition, see Detailed Designs for comprehensive view;
> - Self-Attention layer in Decoder is similar to Self-Attention layer in Encoder, except that 1) Input is vector instead of matrix (Decoder processes one position at at time while Encoder encodes all positions); and 2) For each pass, it can only "*see*" outputs emitted by prev passes (results of prev output positions);
> - Encoder-Decoder Attention is a specialized Self-Attention layer where only Query vector is calculated by the layer while **Key vectors and Value vectors** are results from Encoders (see Self Attention for Q, K, V explaination);

## Self Attention
Self Attention is the **_SOUL_** of Transformer :-)


**Figure 3: an illustrative example**

![](https://jalammar.github.io/images/t/self-attention-output.png)

**_Figure 3 Explained:_** 
> Each input position has corresponding **_vectors_** of:
> - **_x_**: output from prev layer or embedding for first layer;
> - **_q_**: calculated by **_x_** * **_Query Matrix_**(see Q, K, V Explained);
> - **_k_**: calculated by **_x_** * **_Key Matrix_**(see Q, K, V Explained);
> - **_v_**: calculated by **_x_** * **_Value Matrix_**(see Q, K, V Explained); 

> Figure 3 illustrates how to calculate **_z1_** as output for **_x1_**
> - **_scores_**: to represent **_relevance_** between **_x1_** and each **_x[i]_** (including itself), done by **_q1_** * **_k[i]_** (then regulize and normalize, the Divide by... and Softmax);
> - **_z1_**: equals weighted sum of **_v[i]_**, **_score[i]_** as weight;

> Calculate **_z[i]_** for each input position;

The **_SOUL_** of Self Attention itself has **_Query_**, **_Key_**, **_Value_** :-)

**_Q, K, V Explained:_** 
> Intuition of each **_vector_** for position **_x_**:
> * **_q_**: think of it as a mask vector on **_Key_** space to **_query_** for relevant other **_x_**;
> * **_k_**: actual representation of this **_x_** on **_Key_** space;
> * **_v_**: semantic representation of this **_x_**;

> Why scoring by Q and K?

> **_Scores_** represents **_contextual relevance_** between **_x1_** and each **_x[i]_**, rather than semantic similarity; Consider example: "_The **animal** didn't cross the **street** because **it** was too **tired**_" where "**_it_**" has no coherently strong semantic relationship (word embedding wise) with "**_tired_**", however, by some learned dimisions on Key space, for example, a dimision with latent concept of "**_pos-adj_**", by masking "**_pos-adj_**" dimision on query vector of "**_it_**" and setting proper value of the same dimision on key vector of "**_tired_**", now scoring between these two is elevated; Same applies to "**_it_**" against "**_animal_**" and "**_street_**";

**_Some words on FeedForward after Self Attention:_** 
> Each **_z[i]_** from Self Attention will be fed to a FF block after Self-Attention. Note that different **_z[i]_** runs thru different FF blocks and they don't depend/interact with each other in FF stage;

> **_Why FeedForward?_**
> Like most FF layers in modern NNs, FF blocks allow nonlinearity combination of features. Take the same example in Q,K,V, **_z[i]_** for "**_it_**" after Self-Attention may now have positive values set on latent dimensions, e.g: "animal-like", "street-like", "tired-like". By FF, weight for ("street-like" * "tired-like") tend to be zero or negative while weight for ("animal-like" * "tired-like") tend to be positive; Finally, "**_it_**" tends to be set on "animal-like"  and unset on "street-like".

## Encoder-Decoder Attention
> **_TLDR_**
> Think it of as a special Self Attention layer, where **_x_** represents output to be emitted at a particular position. Other **_x[i]_**, in this case, are encoded input at each positino. Therefore, only query vec of **_x_** needs to be calculated in Encoder-Decoder Attention, whereas key and value vecs of **_x[i]_** are simply results from Encoders.

## Some Detailed Designs
Below are some details w.r.t practical matters, link to original post for reference.

### Multi-headed attentions
Instead of one **_z[i]_** per **_x[i]_**, in practice, there will be K output per **_x_** from Self-Attention, where K is the # of attention heads. This is said to improve model performance.
![](https://jalammar.github.io/images/t/transformer_attention_heads_z.png)

### Positional Encoding
What's summarized so far doesn't take input sequence into account. And in practice, position is encoded together with raw embedding as input.
![](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)

### The Residuals
![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)
