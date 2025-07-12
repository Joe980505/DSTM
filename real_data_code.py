import pickle
import numpy as np
import math
from math import log, exp
import random
import scipy.stats
from scipy.stats import poisson, dirichlet
from scipy import optimize
from scipy.special import digamma, gammaln
from scipy.optimize import minimize
import copy
from copy import deepcopy
import pandas as pd
import random
import math
from lifelines import CoxPHFitter
import gensim
from gensim.corpora import Dictionary
from lifelines.utils import to_long_format
from lifelines.utils import add_covariate_to_timeline
from lifelines import CoxTimeVaryingFitter
from DTMpart import DTM_para #For topic-word distribution update
import json
import time
import matplotlib.pyplot as plt
import gensim
from gensim.corpora import Dictionary
from gensim import utils,matutils
from real_data_evalution_metrics import sKL,sKL_sum,calculate_pmi,calculate_coherence_static,calculate_coherence
data_text=pd.read_csv('text.csv',encoding='utf-8-sig')
data_survival=pd.read_csv('time_and_event.csv',encoding='utf-8-sig')
time_slice = data_text.data_dt.value_counts().sort_index().values.tolist()
dict_sample = gensim.corpora.Dictionary(data_text.cut)
dict_sample.filter_extremes(no_below=3) 
corpus_sample= = [dictionary_sample.doc2bow(doc) for doc in data_text.cut]
MT=len(np.unique(data_text.user_id))
time=data_survival['time']
event=data_survival['event']
T=20
K=10

def __init__topic(corpus=None,id2word=None,time_slice=None,
                 alphas=0.1,num_topics=K,initialize='gensim',
                 beta_variance_chain=0.05):
    if corpus is None and id2word is None:
           raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )
        
    if id2word is None:
        logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
        id2word = utils.dict_from_corpus(corpus)
        vocab_len = len(id2word)
    elif id2word:
        vocab_len = len(id2word)
    else:
        vocab_len = 0
       
    if corpus is not None:
        try:
            corpus_len = len(corpus)
        except TypeError:
            logger.warning("input corpus stream has no len(); counting documents")
    num_time_slices = len(time_slice)
    alphas = np.full(num_topics, alphas)
    beta_mean_init=0
    beta_init = [[[random.gauss(beta_mean_init, beta_variance_chain) \
                        for y in range(num_topics)] \
                            for x in range(vocab_len)]]
    for t in range(T-1):
        beta_init.append([[random.gauss(y, beta_variance_chain) for y in x] for x in beta_init[t]])
    beta_init=copy.deepcopy(beta_init)
    for t in range(num_time_slices):
        for k in range(num_topics):
            s = sum([exp(x[k]) for x in beta_init[t]])
            for word in range(vocab_len):
                beta_init[t][word][k] = exp(beta_init[t][word][k])/s
    beta_init=np.array(beta_init)#一个主题下不同词的概率之和为1
    time_slice_start = [sum(time_slice[0:u:1]) for u,k in enumerate(time_slice)]
    time_slice_start.append(sum(time_slice))
    time = []
    n = 0
    for t in time_slice:
        time.extend([n]*t)
        n += 1
    phi = []
    theta = []
    bound_pre_corpus_doc = []
    bound_pre_corpus = 0
    for i_doc, (doc, t) in enumerate(zip(corpus, time)):#重复len(corpus)次
        tmp=np.random.rand(num_topics)
        phi_doc = [np.exp(tmp)/np.sum(np.exp(tmp)) for _ in range(len(doc))]
        theta_doc = [sum(count for word_id, count in corpus[0])/num_topics+alphas] * num_topics
        bound_pre_corpus_doc.append(bound_pre_corpus)
        phi.append(phi_doc) #phi[d][n][k]
        init_z=np.zeros((max(time_slice),num_time_slices,num_topics))#M*T*K
        theta.append(theta_doc)#theta[d][k]，the dimension of d is \sum M_t t=1,2,...,T
    return alphas,theta,phi,time,beta_init,time_slice_start

init_para=__init__topic(corpus=corpus_sample,id2word=dict_sample,time_slice=time_slice,
                 alphas=1/K,num_topics=K,initialize='gensim',beta_variance_chain=1.5)#The beta_variance_chain is determined based on the data distribution.
var_topic=[]
for k in range(K):
    tmp=list()
    for i in range(time_slice[0]):
        tmp.append(0)
        for j in range(len(time_slice)):
            tmp.append(init_para[2][i+j*time_slice[0]][0][k])
    var_topic.append(tmp)
id_num=list(np.unique(data_text.user_id))
dataframe=pd.DataFrame({'id':pd.Series(id_num),'duration':pd.Series(time),'event':pd.Series(event)})
id2=[element for element in list(np.unique(data_text.user_id)) for _ in range(T+1)]
time1=list(np.unique(data_text.user_id))*MT
base_df = to_long_format(dataframe, duration_col="duration")
cv=pd.DataFrame({'id':pd.Series(id2),'time':pd.Series(time1)})
for i in range(K):
    name='var'+str(i+1)
    cv[name]=pd.Series(var_topic[i])
df=add_covariate_to_timeline(base_df, cv,duration_col="time", id_col="id", event_col="event")
ctv = CoxTimeVaryingFitter(penalizer=0.1)
ctv.fit(df, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=False)
init_Cox=ctv.params_
# update coefficients
#E step
def update_gamma(num_topics,alphas,phi,corpus,base_df):
    #update the variational parameter of theta :gamma
    corpus_len = len(corpus)
    n_list = [len(x) for x in corpus]
    gamma = np.stack([np.copy(alphas) for _ in range(corpus_len)], axis=0)
    for m in range(corpus_len):
        n_per = n_list[m]
        for n in range(n_per):
            word_freq = corpus[m][n][1]
            for k in range(num_topics):
                gamma[m][k] += phi[m][n][k] * word_freq
    return gamma
def update_phi(num_topics,gamma,corpus,beta,time_slice,lamb):
    dig = np.zeros(num_topics)
    n_list = [len(x) for x in corpus]
    n_max = max(n_list)
    corpus_len = len(corpus)
    time_slice_start = [sum(time_slice[0:u:1]) for u,k in enumerate(time_slice)]
    time_slice_start.append(sum(time_slice))
    phi_word = np.full(shape=[corpus_len,n_max,num_topics],fill_value=0,dtype=float)
    t=0
    for m in range(corpus_len):
        if (m>=time_slice_start[t+1]):t=t+1
        tmp=m-MT*t
        n_per = n_list[m]
        for n in range(n_per):
            word_id = corpus[m][n][0]
            word_freq = corpus[m][n][1]
            beta_est=np.zeros(num_topics)
            for k in range(num_topics):
                dig[k] = digamma(gamma[m][k])
                beta_est[k]=beta[t][word_id][k]
            dig_sum=digamma(np.sum(gamma[m]))
            
            if event[tmp]=='True':
                phi_est=np.zeros(num_topics)
                for i in range(num_topics):
                    lamb=lamb[i]
                    dig=dig[i]
                    beta_est=beta_est[i]
                    fun= lambda x : (lamb/n_per+dig-dig_sum+np.log(beta_est)-np.log(x))*x-np.log(x)
                    para =[[0,10]]
                    ret = so.differential_evolution(fun, para)
                    phi_est.append(ret.x[0])
                print('phi_est:',phi_est)
                phi_est=[np.exp(x)/sum(np.exp(phi_est)) for x in phi_est]
                phi_word[m][n]=phi_est
            else :  
                phi_est=np.zeros(num_topics)
                for k in range(num_topics):
                    phi_est[k]=beta_est[k]*np.exp(dig[k]-dig_sum)
                phi_tmp=[x/sum(phi_est) for x in phi_est]
                phi_word[m][n]=phi_tmp
    return phi_word
#M step for update beta and coefficients
def update_coef(phi,time_slice,corpus_len,num_topics,time_slice_start,base_df):
    num_time_slice=len(time_slice)
    corpus_len=len(corpus_sample)
    bar_phi=np.zeros((corpus_len,num_topics))
    time_slice_start = [sum(time_slice[0:u:1]) for u,k in enumerate(time_slice)]
    time_slice_start.append(sum(time_slice))
    for m in range(corpus_len):
        tmp1=np.mean(phi[m],axis=0)
        bar_phi[m]=tmp1
    id2=list()
    for i in range(1,MT+1):
        for j in range(T+1):
            id2.append(i)
    var_topic=[]
    for k in range(K):
        tmp=list()
        for i in range(time_slice[0]):
            tmp.append(0)
            for j in range(len(time_slice)):
                tmp.append(bar_phi[i+j*time_slice[0]][k])
        var_topic.append(tmp)
    cv=pd.DataFrame({'id':pd.Series(id2),'time':pd.Series(time1)})
    for i in range(K):
        name='var'+str(i+2)
        cv[name]=pd.Series(var_topic[i])
    df_est=add_covariate_to_timeline(base_df, cv,duration_col="time", id_col="id", event_col="event")
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(df_est, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=False)
    para_Cox=ctv.params_v
    return para_Cox

def update_beta(phi,corpus,time_slice,id2word,num_topics,beta):
    beta_estimated = beta
    V=len(id2word)
    n_tw_all = [0] * num_topics
    n_t_all = [0] * num_topics
    n_list = [len(x) for x in corpus]
    n_max = max(n_list)
    corpus_len = len(corpus)
    time_slice_start = [sum(time_slice[0:u:1]) for u,k in enumerate(time_slice)]
    time_slice_start.append(sum(time_slice))
    t=0
    for k in range(num_topics):
        n_tw_k = [[0] * len(time_slice) for _ in range(V)]#K*V*(0-T)
        for m in range(corpus_len):
            if (m>=time_slice_start[t+1]):t=t+1
            n_per = n_list[m]
            for n in range(n_per):
                word_id = corpus[m][n][0]
                word_freq = corpus[m][n][1]
                n_tw_k[word_id][t]+= word_freq * phi[m][n][k] 
        n_t_k = []
        for t in range(len(n_tw_k[0])):
            n_t_k.append(sum([x[t] for x in n_tw_k]))
            n_tw_all[k] = deepcopy(n_tw_k)#K*V*T
            n_t_all[k] = deepcopy(n_t_k)
    dtm_model_part=DTM_para(id2word=id2word,time_slice=time_slice,topic_suffstats=np.array(n_tw_all),num_topics= num_topics,obs_variance=0.05,chain_variance=1)
    bound=dtm_model_part.fit_lda_seq_topics(np.array(n_tw_all))
    for k in range(num_topics):
        for t in range(len(time_slice)):
            topwords_tmp = dtm_model_part.print_topic(k, t, top_terms=V)#V是词汇表总数
            for wordtoken,prob in topwords_tmp:
                word = id2word.token2id[wordtoken]
                beta_estimated[t][word][k] = prob
    return bound,beta_estimated

def compute_SDTM_lhood(num_topics,corpus,time_slice,beta_estimated,gammas,phi,alphas):
        """Compute the log likelihood bound.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.

        """
        time=0
        corpus_len=len(corpus)
        time_slice=np.cumsum(time_slice)
        sum_lhood=0
        for m in range(corpus_len):
            if m > time_slice[time]:
                time += 1
            beta=beta_estimated[time]
            lhood=[0]*num_topics
            gamma=gammas[m]
            gamma_sum = np.sum(gamma)
            lhoods = gammaln(np.sum(alphas)) - gammaln(gamma_sum)
            digsum = digamma(gamma_sum)

            for k in range(num_topics):
                e_log_theta_k = digamma(gamma[k]) - digsum
                lhood_term = (alphas[k] - gamma[k]) * e_log_theta_k + gammaln(gamma[k]) - gammaln(alphas[k])
                n = 0
                for word_id, count in corpus[m]:
                    if phi[m][n][k] > 0:
                        lhood_term += \
                            count * phi[m][n][k] * (e_log_theta_k + beta[word_id][k] - np.log(phi[m][n][k]))
                    n += 1
                lhood[k] = lhood_term
            sum_lhood+=np.sum(lhood)

        return sum_lhood

def compute_Cox_lhood(num_topics,corpus,vocab_len,time_slice,phi,para_Cox):
    Cox_bound=0
    bound_tmp=list()
    for m in range(time_slice[0]):
        if event[m]=='True': 
            for t in range(len(time_slice)):
                phi_=phi[corpus_len*t+m]
                phi_bar=np.mean(phi,axis=0)
                term1=np.sum(para_Cox[1:]*phi_bar)
                term2=vocab_len*np.sum(np.log(num_topics-1+np.exp(para_Cox[1:]/vocab_len)))+np.sum(var1[m]*para_Cox[0])
                numbers=base_df['stop']
                value=numbers[m]
                tmp=np.array(numbers[numbers > value].index)
                term3=0
                for i in range(tmp):
                    tmp2=np.log(np.cumsum(phi[corpus_len*t+i]*np.exp(np.sum(var1[i]*para_Cox[1:]))))
                    term3+=tmp2
                term=term1+term2+term3
                bound_tmp.append(term)
    Cox_bound=np.sum(bound_tmp)
    return Cox_bound

def fit_SDTM(corpus,dict_sample,time_slice,num_topics,base_df,em_min_iter,em_max_iter):
    #init_para
    alphas=init_para[0]
    phi=init_para[2]
    beta=init_para[4]
    time_slice_start=init_para[5]
    lambda_est=lambda_init
    corpus_len=len(corpus)
    vocab_len=len(dict_sample)
    LDASQE_EM_THRESHOLD = 1e-4
    #E step for update gamma and phi
    iter_=0
    bound=0
    while iter_ < em_min_iter or ((convergence > LDASQE_EM_THRESHOLD) and iter_ <= em_max_iter):
        bound_pre = bound
        print('正在进行第{}次迭代'.format(iter_))
        gamma = update_gamma(num_topics,alphas,phi,corpus,base_df)
        phi=update_phi(num_topics,gamma,corpus,beta,time_slice,lambda_est)
    #M step for update coefi and beta
        para_Cox=update_coef(phi,time_slice,corpus_len,num_topics,time_slice_start,base_df)
        lambda_est=para_Cox[1:]
        dtm_bound,beta=update_beta(phi,corpus,time_slice,dict_sample,num_topics,beta)
        Cox_bound=compute_Cox_lhood(num_topics,corpus,vocab_len,time_slice,phi,para_Cox)
        SDTM_bound=compute_SDTM_lhood(num_topics,corpus,time_slice,beta,gamma,phi,alphas)
        bound=Cox_bound+SDTM_bound+dtm_bound
        convergence = np.fabs((bound - bound_pre) / bound_pre)
        print('bound result:',bound)
        iter_+=1
    return gamma,phi,para_Cox,beta

num_topics=5
min_iter_em=2
max_iter_em=50
results=fit_SDTM(corpus_sample,dict_sample,time_slice,num_topics,base_df,min_iter_em,max_iter_em)
beta=results[3]
sample_time_id = [sum(time_slice[:i+1]) for i,x in enumerate(time_slice)]
time_slice_start = [sum(time_slice[0:u:1]) for u,k in enumerate(time_slice)]
time_slice_start.append(sum(time_slice))
SKL_DSTM=sKL_sum(beta,10,len(time_slice))
coherence_DSTM = calculate_coherence(np.array(beta), corpus_sample, 10,10,time_slice,time_slice_start)
SKL_DSTM=sKL_sum(beta,5,len(pre_time_slice))
print('SKL_DSTM:',SKL_DSTM)
print('PMI_DSTM:',PMI_DSTM)
print('COH_DSTM:',COH_DSTM)