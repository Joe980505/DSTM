import math
from math import log, exp
import random
import scipy.stats
from scipy.stats import poisson, dirichlet
from scipy import optimize
import copy
from copy import deepcopy
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import gensim
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from gensim import utils,matutils
from DTMpart import SJDTM_para
from gensim import utils,matutils
inp_V=1858
#sKL
def sKL(Phi,time,topici,topicj):
    """
    Input:
    Phi:topic-word distribution,K*V dimension
    """
    skl=0
    for v in range(inp_V):
        probi = Phi[time][v][topici]
        probj = Phi[time][v][topicj]
        tmp = 0.5 * probi * np.log(probi / probj) + probj * np.log(probj / probi)
        skl=skl+tmp
    return skl

def sKL_sum(Phi,num_topics,time):
    skl_sum = 0.0
    for t in range(time):
        for topic_i in range(num_topics):
            for topic_j in range(num_topics):
                if topic_i==topic_j:
                    pass
                else:
                    skl_sum += sKL(Phi, t,topic_i, topic_j)
    return(skl_sum)

#PMI
def calculate_pmi(Phi,time_slice,N,corpus,time_slice_start):
    '''
    Input: 
    Phi: topic-word distribution,K*V dimension
    N:The top N high frequency words for each topic
    '''
    pmi_sum=0
    for t in range(len(time_slice)):
        word=np.array(list(map(list,zip(*Phi[t]))))
        tmp_corpus = corpus[time_slice_start[t]:time_slice_start[t+1]]
        important_word = np.apply_along_axis(lambda x: np.argsort(-x)[:N], 1,word)  #top N high frequency words，K*N dimension
        K = word.shape[0]
        pmik = np.zeros(K)
        for k in range(K):
            pmikk = 0
            for i in range(N-1):      #choose the first term
                for j in range(i+1, N): #choose the second term
                    D_i = 0
                    D_j = 0
                    D_ij = 0
                    for doc in tmp_corpus:     #each document
                        words = [word for word, freq in doc]
                        if important_word[k, i] in words:
                            D_i += 1
                            if important_word[k, j] in words:
                                D_j += 1
                                D_ij += 1
                        elif important_word[k, j] in words:
                            D_j += 1
                        
                    if D_ij != 0:
                        pmikk += np.log(D_ij) - np.log(D_i) - np.log(D_j) + np.log(len(tmp_corpus))
                
            pmik[k] = 2*pmikk/(N*(N-1))
            pmi_sumt=sum(pmik)
            pmi_sum+=pmi_sumt
    
        
    return pmi_sum


def calculate_coherence_static(phi, corpus_sample, N_word):
    """
    calculate_coherence score in each time slice
    :param phi: topic-word distribution，K*V dimensions
    :param corpus_diff_words: list,order number of different words in each document
    :param N_word: The top N high frequency words for each topic
    :return: coherence score
    """
    word_list = []
    corpus_diff_words = [list(map(lambda x:x[0],corpus_sample[i])) for i in range(len(corpus_sample))]
    for i in range(phi.shape[0]):
        word_list.append(np.argsort(phi[i,:],)[-N_word:])
    coherence = np.zeros(phi.shape[0])
    for k in range(phi.shape[0]):
        for i in range(N_word-1):
            for j in range(i+1):
                count_gongxian = 0
                count_single = 0
                for n in range(len(corpus_diff_words)):
                    count_gongxian += int(set([word_list[k][i+1],word_list[k][j]]).issubset(set(corpus_diff_words[n])))
                    count_single += int(set([word_list[k][j]]).issubset(set(corpus_diff_words[n])))
                if(count_single==0): count_single=1
                coherence[k] += np.log((count_gongxian+1)/count_single)
    result=sum(coherence)
    return result


def calculate_coherence(Phi, corpus_sample, num_topics,N_word,time_slice,time_slice_start):
    """
    coherence score for all of the time slice 
    :param phi: topic-word distribution，K*V dimensions
    :param corpus_diff_words: list,order number of different words in each document
    :param N_word: The top N high frequency words for each topic
    :return: coherence score
    """
    T=len(time_slice)
    coherence = np.zeros(T)
    for t in range(T):
        phi=np.array(list(map(list,zip(*Phi[t]))))
        corpus = corpus_sample[time_slice_start[t]:time_slice_start[t+1]]
        coherence[t]=calculate_coherence_static(phi, corpus, 10)
    result=sum(coherence)
    return result
