# DSC 180B Final Pipeline
---
## How to run

```
usage: python run.py [-h] [-p] [-s] [-r]

optional arguments:
  -h, --help       show help message and exit
  -p, --pages      obtain list of pages to analyze from Wikipedia
  -s, --sentiment  run sentiment analysis on pages from list
  -r, --results    obtain stats and visuals from sentiment analysis
  -t, --test       runs test suite
```

## Purpose
This program collects wikipedia data from URLs to analyze the sentiment of articles over time.

## Config Formats
The configuration .json files in the config folder can be used to change the program operation
### Pages
* language:      English, Spanish, or Chinese
* targets:       Wikipedia categories from which to analyze articles
* skip_cats:     Wikipedia categories to skip due to abundance of unnecessary articles
* output:        data file to write list of articles to
### Sentiment
* language:      English, Spanish, or Chinese
* infile:        data file to read in from
* outfile:       data file to write to
### Results
* language:      English, Spanish, or Chinese
* infile:        data file to read in from
* outfile:       data file to write to
### Test - Pages
* language:      English, Spanish, or Chinese
* targets:       Wikipedia categories from which to analyze articles
* skip_cats:     Wikipedia categories to skip due to abundance of unnecessary articles
* output:        data file to write list of articles to
### Test - Sentiment
* language:      English, Spanish, or Chinese
* infile:        data file to read in from
* outfile:       data file to write to
### Test - Results
* language:      English, Spanish, or Chinese
* infile:        data file to read in from
* outfile:       data file to write to

---
Yuanbo Shi

Henry Lozada

Parth Patel

Emma Logomasini

UCSD Winter 2021
