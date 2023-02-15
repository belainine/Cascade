#  Cascade Model - : A Pytorch Implementation
Is a model encoder-decoder, the encoder represents a single
RNR and decoder is a sequence of RNRs that progressively build a prediction context by trusting each other and each managing a word position in the dialog utterances. The final context is thus based as much on the words as the statements preceding the answer to be predicted.

This is a PyTorch implementation of peper "End-to-End Dialogue Generation Using a Single Encoder and a Decoder Cascade With a Multidimension Attention Mechanism,"


A novel multi sequence to sequence framework utilizes the **sequence of Recurrent structure**.

The project support training and dailog generation with trained model now.

The project support training and dialog generation now.
<p align="center">
<img src="https://github.com/belainine/Cascade/blob/main/Cascade1.jpg" width="600">
  
# Requirement
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy
- [GoogleNews-vectors-negative300.bin.gz](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)
- nltk
- jiwer
- tensorboardX
- matplotlib
- gensim

# Usage

# pytorch-chatbot
CASCADE chatbot implement using PyTorch  
Feature: CASCADE model + Beam Search

## Corpus
- [DailyDialog](http://www.aclweb.org/anthology/I17-1099)
- [Ubuntu](https://arxiv.org/abs/1506.08909)
- [Cornell Movie](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
## Usage
### Training
```python
python chatbot_Cascade.py nb_decoder 3 -train model Cornell_Movie
```
### Test
```python
python chatbot_Cascade.py -log -test nb_decoder 3 ./ckpt model beam 4 Cornell_Movie_test
```
  ## Citation
```
@ARTICLE{9723498,
  author={Belainine, Billal and Sadat, Fatiha and Boukadoum, Mounir},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={End-to-End Dialogue Generation Using a Single Encoder and a Decoder Cascade With a Multidimension Attention Mechanism}, 
  year={2022},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TNNLS.2022.3151347}
  }
```

## Reference
- [End-to-End Dialogue Generation Using a Single Encoder and a Decoder Cascade With a Multidimension Attention Mechanism,](https://ieeexplore.ieee.org/abstract/document/9723498)
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)
- [seq2seq-translation.ipynb](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
- [Pytorch Documentation](https://pytorch.org/docs/0.3.0/)
