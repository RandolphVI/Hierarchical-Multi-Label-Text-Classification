# Hierarchical Multi-Label Text Classification

[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Hierarchical-Multi-Label-Text-Classification.svg?branch=master)](https://travis-ci.org/RandolphVI/Hierarchical-Multi-Label-Text-Classification)[![Codacy Badge](https://api.codacy.com/project/badge/Grade/80fe0da5f16146219a5d0a66f8c8ed70)](https://www.codacy.com/manual/chinawolfman/Hierarchical-Multi-Label-Text-Classification?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Hierarchical-Multi-Label-Text-Classification&amp;utm_campaign=Badge_Grade)[![License](https://img.shields.io/github/license/RandolphVI/Hierarchical-Multi-Label-Text-Classification.svg)](https://www.apache.org/licenses/LICENSE-2.0) 

This repository is my research project, and it is accepted by CIKM'19. The [paper](https://dl.acm.org/citation.cfm?id=3357384.3357885) is already published.

The main objective of the project is to solve the hierarchical multi-label text classification (**HMTC**) problem. Different from the multi-label text classification, HMTC assigns each instance (object) into multiple categories and these categories are stored in a hierarchy structure, is a fundamental but challenging task of numerous applications.

## Requirements

- Python 3.6
- Tensorflow 1.15.0
- Tensorboard 1.15.0
- Numpy 1.16.2
- Gensim 3.5.0

## Introduction

Many real-world applications organize data in a hierarchical structure, where classes are specialized into subclasses or grouped into superclasses. For example, an electronic document (e.g. web-pages, digital libraries, patents and e-mails) is associated with multiple categories and all these categories are stored hierarchically in a **tree** or **Direct Acyclic Graph (DAG)**. 

It provides an elegant way to show the characteristics of data and a multi-dimensional perspective to tackle the classification problem via hierarchy structure. 

![](https://farm8.staticflickr.com/7806/31717892987_e2e851eaaf_o.png)

The Figure shows an example of predefined labels in hierarchical multi-label classification of documents in patent texts. 

- Documents are shown as colored rectangles, labels as rounded rectangles. 
- Circles in the rounded rectangles indicate that the corresponding document has been assigned the label. 
- Arrows indicate a hierarchical structure between labels.

## Project

The project structure is below:

```text
.
├── HARNN
│   ├── test_harnn.py
│   ├── text_harnn.py
│   ├── train_harnn.py
│   └── visualization.py
├── data
│   ├── word2vec_100.model.* [Need Download]
│   ├── Test_sample.json
│   ├── Train_sample.json
│   └── Validation_sample.json
└── utils
│   ├── checkmate.py
│   ├── data_helpers.py
│   └── param_parser.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Data

You can download the [Patent Dataset](https://drive.google.com/open?id=1So3unr5p_vlYq31gE0Ly07Z2XTvD5QlM) used in the paper. And the [Word2vec model file](https://drive.google.com/open?id=1cu5sjts9x7eOcKw-ngXwFKpDMVItaivk) (dim=100) is also uploaded. **Make sure they are under the `/data` folder.**

(As for **Education Dataset**, they may attract copyright protection under China law. Thus, there is no details of dataset.)

### Text Segment

1. You can use `nltk` package if you are going to deal with the English text data.

2. You can use `jieba` package if you are going to deal with the Chinese text data.

### Data Format

See data format in `data` folder which including the data sample files. For example:

```json
{"id": "3930316", "title": ["sighting", "firearm"], "abstract": ["rear", "sight", "firearm", "ha", "peephole", "device", "formed", "hollow", "tube", "end", "closed", "peephole",], "section": [5], "subsection": [104], "group": [512], "subgroup": [6535], "labels": [5, 113, 649, 7333]}
```

- **"id"**: just the id.
- **"title" & "abstract"**: it's the word segment (after cleaning stopwords).
- **"section"**: it's the first level category index.
- **"subsection"**: it's the second level category index.
- **"group"**: it's the third level category index.
- **"subgroup"**: it's the fourth level category index.
- **"labels"**: it's the total category which add the index offset. (I will explain that later)

### How to construct the data?

Use the <u>figure in Introduction</u> as example, now I will explain how to construct the label index. 

**Step 1:** Figure has 3 categories, you should index this 3 categories first, like:

```
{"Chemistry": 1, "Physics": 2, "Electricity": 3}
```

**Step 2**: You index the next level, like:

```
{"Inorganic Chemistry": 1, "Organic Chemistry": 2, "Nuclear Physics": 3, "Material analysis": 4, "XXX": 5, ..., "XXX": N}
```

Note: ***N*** is the total number of your subcategories.

**Step 3**: You index the third level, like:

```
{"Steroids": 1, "Peptides": 2, "Heterocyclic Compounds": 3, ..., "XXX": M}
```

Note: ***M*** is the total number of your level-3 categories.

**Step 4**: If you have the fourth level or deeper level, index them.

**Step 5**: Now Suppose you have one record:

```
{"id": "1", "title": ["tokens"], "abstract": ["tokens"], "section": [1, 2], "subsection": [1, 2, 3, 4], "group": [1, 2, 3, 4], "labels": [1, 2, 1+N, 2+N, 3+N, 4+N, 1+N+M, 2+N+M, 3+N+M, 4+N+M]}
```

**Assume that your total category number of level-2 is 100 (*N*=100), of level-3 is 500 (*M*=500). *N* & *M* is the offset for the `labels` attribute.**

the record should be construed as:

```json
{"id": "1", "hashtags": ["token"], "section": [1, 2], "subsection": [1, 2, 3, 4], "group": [1, 2, 3, 4], "labels": [1, 2, 101, 102, 103, 104, 601, 602, 603, 604]}
```

This repository can be used in other datasets (text classification) in two ways:
1. Modify your datasets into the same format of [the sample](https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification/tree/master/data).
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depend on what your data and task are.

### Pre-trained Word Vectors

You can pre-training your word vectors(based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Usage

See [Usage](https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification/blob/master/Usage.md).

## Network Structure

![](https://live.staticflickr.com/65535/48647692206_2e5e6e7f13_o.png)

## Reference

**If you want to follow the paper or utilize the code, please note the following info in your work:** 

```bibtex
@inproceedings{huang2019hierarchical,
  author    = {Wei Huang and
               Enhong Chen and
               Qi Liu and
               Yuying Chen and
               Zai Huang and
               Yang Liu and
               Zhou Zhao and
               Dan Zhang and
               Shijin Wang},
  title     = {Hierarchical Multi-label Text Classification: An Attention-based Recurrent Network Approach},
  booktitle = {Proceedings of the 28th {ACM} {CIKM} International Conference on Information and Knowledge Management, {CIKM} 2019, Beijing, CHINA, Nov 3-7, 2019},
  pages     = {1051--1060},
  year      = {2019},
}
```
---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
