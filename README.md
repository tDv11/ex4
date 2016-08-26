# ex4

Crowdflower Search Results Relevance .

The preprocessing was all about word replacement, spelling correction and synonym replacement for deep learning.
we started with building a model using ensemble selection, by creating a library that is intended to be as diverse as possible to
capitalize on a large number of unique learning approaches.
to decode we calculated the precentage of any relevant level, by that we rank the prediction in order:
0-10 was set to 1.
10-25 was set to 2
25-50 was set to 3
and 50-100 to 4.
we mostly relied on numpy and SVD, and for the training part, on keras.
