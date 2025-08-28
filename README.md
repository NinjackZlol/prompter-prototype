# Pain Point & Needs detection Prototype

This repository contains experimental approaches for detecting **pain points**, **needs** and **depth levels** from sales conversation transcripts

## Requirements

Install dependencies

**bash**
`pip install transformers sentence-transformers scikit-learn datasets torch transformers[torch]`


## How to run

Zero-shot

`python zero-shot.py`

Embedding-Based Classification

`python embedding.py`

DistillBERT Fine Tuning

`python lw-bert.py`

##Documentation
Research Doc https://docs.google.com/document/d/10EdMDjwxjT-GTeV57Qpq41_AW2BR8tp6xo8NeaiZjGc/edit?usp=sharing
