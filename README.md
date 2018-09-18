# ContextualSLU: Multi-Turn Spoken/Natural Language Understanding


*A Keras implementation of the models described in [Chen et al. (2016)] (https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_ContextualSLU.pdf).*

This model implements a memory network architecture for multi-turn understanding, 
where the history utterances are encoded as vectors and stored into memory cells for the current utterance's attention to improve slot tagging.

## Content
* [Requirements](#requirements)
* [Getting Started](#getting-started)
* [Model Running](#model-running)
* [Contact](#contact)
* [Reference](#reference)

## Requirements
1. Python
2. Numpy `pip install numpy`
3. Keras and associated Theano or TensorFlow `pip install keras`
4. H5py `pip install h5py`

## Dataset
1. Train/Test: word sequences with IOB slot tags and the indicator of the dialogue start point (1: starting point; 0: otherwise) `data/cortana.communication.5.[train/dev/test].iob`


## Getting Started
You can train and test JointSLU with the following commands:

```shell
  git clone --recursive https://github.com/halloTheCoder/ContextualSLUChatBot.git
  cd ContextualSLUChatBot
```

## Model Running
Points to consider while running
- Give pretrained-embeddings as ![ConceptNet Numberbatch][https://github.com/commonsense/conceptnet-numberbatch], other will lead to error.
- Change data directories in `sequence_tagger.py`, by default **dataset without acts**. (NOTE :: Don't use **dataset folder as acts is not handled in code for now**)

```shell
	python sequence_tagger.py

```

## Contact
* Email - akash.chandra8d@gmail.com


## Reference

Main papers to be cited
```
@Inproceedings{chen2016end,
  author    = {Chen, Yun-Nung and Hakkani-Tur, Dilek and Tur, Gokhan and Gao, Jianfeng and Deng, Li},
  title     = {End-to-End Memory Networks with Knowledge Carryover for Multi-Turn Spoken Language Understanding},
  booktitle = {Proceedings of Interspeech},
  year      = {2016}
}


