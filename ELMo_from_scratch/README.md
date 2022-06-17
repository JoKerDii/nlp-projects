# Sentiment Analysis Projects

## 1. IMDB Sentiment Analysis with ELMo

### Goal

For this sentiment analysis task we used IMDB dataset which is publicly available [here](http://ai.stanford.edu/~amaas/data/sentiment/). We aimed to classify movie reviews as positive and negative. The work can be split into three parts:

1. Build a language model to train a basic ELMo. Instead of using Character Embeddings, we used word embeddings.
2. Use the generated ELMo embeddings to performn sentiment analysis on IMDB dataset.
3. Evaluate the model with trained ELMo embedding with other two models without trained embedding and with a word2vec embedding.

### Results

|          | Model with embeddings from scratch | Model with word2vec embeddings | Model with trained ELMo embeddings |
| -------- | ---------------------------------- | ------------------------------ | ---------------------------------- |
| Accuracy | 0.8722                             | 0.8609                         | 0.8177                             |

### Discussion

The embeddings trained from scratch (baseline) performed surprisingly well, almost as good as the pretrained word2vec model. It is probably because the task is relatively easy. It is likely that the performance of the models would differ much more on a more sophisticated multi-class classification problem. Moreover, because we are dealing only with movie reviews, words have very specific connotations in that context. For example, the words "flop", "bomb", and "turkey" most likely mean "a bad movie" when appearing in a movie review. By training our own word embeddings we are able to focus on these idiomatic usages whereas the pretrained word2vec embeddings must represent a combination of all possible meanings of these words across all contexts. So if we were to use word2vec embeddings and keep the embeddings trainable and fine tune these original embeddings to our specific domain of movie reviews would likely give even better results.

The ELMo embedding model's performance is disappointing. One reason is that we have to limit teh amount of training data due to memory constraints. The max length argument to the IMDB data decreases and so the performance of the final suffers. Setting `maxlen` to be smaller reduces a large number of reviews that exceed this length. An under-trained ELMo model simply does not produce very useful contextual embeddings.