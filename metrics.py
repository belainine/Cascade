from random import randint
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import argparse
from jiwer import wer
import sys
def greedy_match(r1,r2, w2v):
    res1 = greedy_score(r1,r2, w2v)
    res2 = greedy_score(r1,r2, w2v)
    res_sum = (res1 + res2)/2.0

    return np.mean(res_sum), 1.96*np.std(res_sum)/float(len(res_sum)), np.std(res_sum)


def greedy_score(r1,r2, w2v):

    dim = w2v.vector_size # embedding dimensions

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X= np.zeros((dim,))
        y_count = 0
        x_count = 0
        o = 0.0
        Y = np.zeros((dim,1))
        for tok in tokens2:
            if tok in w2v:
                Y = np.hstack((Y,(w2v[tok].reshape((dim,1)))))
                y_count += 1

        for tok in tokens1:
            if tok in w2v:
                tmp  = w2v[tok].reshape((1,dim)).dot(Y)
                o += np.max(tmp)
                x_count += 1

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0)
            continue

        o /= float(x_count)
        scores.append(o)


    return np.asarray(scores)


def extrema_score(r1,r2, w2v):


    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X= []
        for tok in tokens1:
            if tok in w2v:
                X.append(w2v[tok])
        Y = []
        for tok in tokens2:
            if tok in w2v:
                Y.append(w2v[tok])

        # if none of the words have embeddings in ground truth, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X,0)  # get abs of min
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)   # get extrema

        ymax = np.max(Y, 0)
        ymin = np.min(Y,0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)


def average(r1,r2, w2v):

    dim = w2v.vector_size # dimension of embeddings

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        X= np.zeros((dim,))
        for tok in tokens1:
            if tok in w2v:
                X+=w2v[tok]
        Y = np.zeros((dim,))
        for tok in tokens2:
            if tok in w2v:
                Y += w2v[tok]

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        X = np.array(X)/np.linalg.norm(X)
        Y = np.array(Y)/np.linalg.norm(Y)
        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)
def evaluationMetricsWord(decoded_words_referances, decoded_words_candidates,w2v):
    
    r1=decoded_words_referances
    r2=decoded_words_candidates
    r = average(r1, r2,  w2v)
    print("Embedding Average Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))

    r = greedy_match(r1, r2, w2v)
    print("Greedy Matching Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))

    r = extrema_score(r1,r2, w2v)
    print("Extrema Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))
    
    error = wer(r1,r2)
    print("Word error rate Score: %f " %(error))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth', help="ground truth text file, one example per line", default='data/embeddings/test1.txt')
    parser.add_argument('predicted', help="predicted text file, one example per line", default='data/embeddings/test1.txt')
    parser.add_argument('embeddings', help="embeddings bin file", default='data/embeddings/GoogleNews-vectors-negative300.bin.gz')
    sys.argv[1:]=['data/embeddings/test1.txt','data/embeddings/test1.txt','data/embeddings/GoogleNews-vectors-negative300.bin.gz']

    args = parser.parse_args()
    
    print("loading embeddings file...")
    w2v = KeyedVectors.load_word2vec_format(args.embeddings, binary=True)
    f1 = open(args.ground_truth, 'r')
    f2 = open(args.predicted, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()
    evaluationMetricsWord(r1, r2,w2v)
    f1.close()
    f2.close()