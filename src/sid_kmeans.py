#! /usr/bin/env python

# Extracts K-means cluster and SID features as described by Yantcheva et al. (2016).
# This code is based on the code used for experiments by Sirts et al. (2017).

# References:
# ---
# Yancheva, M., & Rudzicz, F. (2016, August). Vector-space topic models for detecting Alzheimer’s disease. 
# In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 2337-2346).
# ---
# Sirts, K., Piguet, O., & Johnson, M. (2017, August). Idea density for predicting Alzheimer’s disease from transcribed speech. 
# In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017) (pp. 322-332). Chicago

# Created by: Kairit Sirts
# Created at: 26.08.2020
# Contact: kairit.sirts@gmail.com


import sys
from sklearn.cluster import KMeans
import numpy as np
import json
from scipy._lib._util import _asarray_validated
import argparse
import os
from collections import Counter
from torchtext.vocab import Vocab
import jsonlines

# Implementation adopted from scipy.cluster.vq.whiten
def whiten(obs, check_finite=True, std_dev=None):
    obs = _asarray_validated(obs, check_finite=check_finite)
    if std_dev is None:
        std_dev = np.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    print(zero_std_mask)
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning)
    return (std_dev, obs / std_dev)


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', help="Path to input file, a json lines file.")
    parser.add_argument('-O', '--output', help="Path to output feature file")
    parser.add_argument('-C', '--cluster', action="store_true", help="Compute K-means clustering features")
    parser.add_argument('-S', '--sid', action="store_true", help="Compute SID feature based on K-means clustering")
    parser.add_argument('-D', '--mode', default="train", choices=["train", "predict"], help="Compute the features in train or predict mode")
    parser.add_argument('--id', default="")
    parser.add_argument('-M', '--model', default='/tmp/model')
    return parser.parse_args()


def read_words(fn):
    words = Counter()
    with open(fn) as f:
        for line in f:
            data = json.loads(line.strip())
            words.update(data['text'].split())
    return words

def main(args):
    if args.mode == 'train':
        train(args)
    else:
        predict(args)

def train(args):
    # Construct vocabulary
    words = read_words(args.input)
    vocab = Vocab(words, specials=(), vectors='glove.6B.50d')
    
    # Prepare clustering data
    embeddings = vocab.vectors
    std_dev, data = whiten(embeddings)

    # Perform K-means
    kmeans = KMeans(n_clusters=10, tol=1e-5)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    clusters = kmeans.labels_
    
    # Calculate parameters necessary for computing features
    d = np.linalg.norm(data - centers[clusters], axis=1)
    mu = []
    std = []
    for i in range(10):
        mask = clusters == i
        mu.append(np.mean(d[mask], axis=0))
        std.append(np.std(d[mask], axis=0))

    mu = np.array(mu)
    std = np.array(std)

    d_norm = (d - mu[clusters]) / std[clusters]

    # Open output file
    if args.output:
        out_fn = args.output + args.id
        of = open(out_fn, 'w')
    else:
        of = sys.stdout

    # Print file header
    header = ["id", "label", "num"]
    if args.cluster:
        header += ["C" + str(i) for i in range(10)]
    if args.sid:
        header.append("sid")
    print('\t'.join(header), file=of)

    # Calculate and print features
    with jsonlines.open(args.input) as f:
        for item in f:
            feats = [item["id"], item["label"], item["num"]]
            dist = [[] for i in range(10)]
            tokens = item["text"].strip().split()
            icus = 0
            for word in tokens:
                ind = vocab[word]
                cluster = clusters[ind]
                dist[cluster].append(d_norm[ind])
                if d_norm[ind] < 3.0:
                    icus += 1

            if args.cluster:
                feats += [str(np.mean(vec)) if len(vec) else '0.0' for vec in dist]
            if args.sid:
                sid = 1.0 * icus / len(tokens)
                feats.append(str(sid))

            print('\t'.join(feats), file=of)

    # Finally, close the outfile
    of.close()


    params = {
              'std_dev': std_dev.tolist(),
              'centers': centers.tolist(), 
              'mu': mu.tolist(), 
              'std': std.tolist(), 
            }

    model_fn = args.model + args.id + '.json'
    with open(model_fn, 'w') as f:
        json.dump(params, f, indent=2)


def predict(args):
    model_fn = args.model + args.id + '.json'
    with open(model_fn) as f:
        model = json.load(f)

    # Construct vocabulary
    words = read_words(args.input)
    vocab = Vocab(words, specials=(), vectors='glove.6B.50d')

    # Prepare clustering data
    embeddings = vocab.vectors
    std_dev = np.array(model['std_dev'])
    _, data = whiten(embeddings, std_dev=std_dev)  

    # Construct the K-means model
    kmeans = KMeans(n_clusters=10)
    kmeans._n_threads = None  # Otherwise the predict method does not work when loading model parameters from file
    centers = np.array(model['centers'])
    kmeans.cluster_centers_ = centers
    clusters = kmeans.predict(data)

    # Calculate parameters necessary for computing features
    d = np.linalg.norm(data - centers[clusters], axis=1)
    mu = np.array(model['mu'])
    std = np.array(model['std'])
    d_norm = (d - mu[clusters]) / std[clusters]

    # Open output file
    if args.output:
        out_fn = args.output + args.id
        of = open(out_fn, 'w')
    else:
        of = sys.stdout

    # Print file header
    header = ["id", "label", "num"]
    if args.cluster:
        header += ["C" + str(i) for i in range(10)]
    if args.sid:
        header.append("sid")
    print('\t'.join(header), file=of)

    # Calculate and print features
    with jsonlines.open(args.input) as f:
        for item in f:
            feats = [item["id"], item["label"], item["num"]]
            dist = [[] for i in range(10)]
            tokens = item["text"].strip().split()
            icus = 0
            for word in tokens:
                ind = vocab[word]
                cluster = clusters[ind]
                dist[cluster].append(d_norm[ind])
                if d_norm[ind] < 3.0:
                    icus += 1

            if args.cluster:
                feats += [str(np.mean(vec)) if len(vec) else '0.0' for vec in dist]
            if args.sid:
                sid = 1.0 * icus / len(tokens)
                feats.append(str(sid))

            print('\t'.join(feats), file=of)

    # Finally, close the outfile
    of.close()

if __name__=='__main__':
    args = get_parsed_args()
    main(args)
