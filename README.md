# Tweet Sentiment Extraction - Serving the app with Flask

Repo with code for training your own bert model fine tuned on imdb dataset to serve it on a webapp using flask.

## Demo

![](bert-flask.gif)

## Getting Started

Clone repository, install requirements, download datasets and pretrained bert model.

### Prerequisites

Must have a GPU for training and inference. You can train on CPU but this will very likely burn up your machine :).
Inference can be done through CPU, yet will not be very efficient.

Download following datasets and put then in input file (see config.py for paths I used):
-https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
-https://www.kaggle.com/abhishek/bert-base-uncased

## Requirements

See requiments.txt.

## Acknowledgments

Thanks to Abhishek Thakur (x3 GM on Kaggle) for providing awesome tutorials. 
