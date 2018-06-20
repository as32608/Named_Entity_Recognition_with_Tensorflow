# Named_Entity_Recognition_with_Tensorflow
Named Entity Recognition with bidirectional LSTM + CRF architecture on word level and character level. \
Course project at university to anonymize german textual data and comply with GDPR.

## Prerequisites
- Tensorflow
- numpy
- pandas 
- nltk
- sklearn

## Dataset
GermanEval2014 dataset.
It was sampled from German Wikipedia and News Corpora as a collection of citations and covers over 31,000 sentences corresponding to over 590,000 tokens.
https://sites.google.com/site/germeval2014ner/data

## Code 
I uploaded three **.py** files and one **.ipynb** file. The .py files contain the network implementation and utilities. The Jupyter Notebook is a demo of how to apply the model.

## Architecture
 + Word level bidirectional LSTM 
 + Char level bidirectional LSTM
 + Bahdanau Attention
 + Dropout everywhere
 + Stacked fully connected layers
 + CRF or cross entropy loss.


Even though the model was trained on german data, the architecture should apply to any language. \
It scores around **state-of-the-art results**.
