# Pain Point & Needs detection Prototype

This repository contains experimental approaches for detecting **pain points**, **needs** and **depth levels** from sales conversation transcripts

## Requirements

Install dependencies

**bash**
`pip install torch transformers scikit-learn sentence-transformers`


## How to run

Zero-shot

`python zero-shot.py`

Embedding-Based Classification

`python embedding.py`

DistillBERT Fine Tuning

`python lw-bert.py`
