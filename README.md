#  Transformer DSM Attention - : A Pytorch Implementation

This is a PyTorch implementation of peper "End-to-End Dialogue Generation Using a Single Encoder and a Decoder Cascade With a Multidimension Attention Mechanism,"


A novel multi sequence to sequence framework utilizes the **sequence of Recurrent structure**.

The project support training and translation with trained model now.


# Requirement
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy


# Usage

# pytorch-chatbot
Seq2Seq chatbot implement using PyTorch  
Feature: Seq2Seq + Beam Search

## Requirements
- Python3
- Pytorch 0.3

## Corpus
- [DailyDialog](http://www.aclweb.org/anthology/I17-1099)
- [Ubuntu](https://arxiv.org/abs/1506.08909)
- [Cornell Movie](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
## Usage
### Training
```python
python chatbot_Cascade.py nb_decoder 3 train model
```
### Test
```python
python chatbot_Cascade.py test nb_decoder 3 ./ckpt model beam 4
```

## Reference
- [End-to-End Dialogue Generation Using a Single Encoder and a Decoder Cascade With a Multidimension Attention Mechanism,](https://ieeexplore.ieee.org/abstract/document/9723498)
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)
- [seq2seq-translation.ipynb](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
- [Pytorch Documentation](https://pytorch.org/docs/0.3.0/)