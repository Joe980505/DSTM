1. realdata_twtm_input.pkl has the following data: the lead corpus and lag corpus ,time slice for lead and lag corpus,number of shared,lead-specific and lag-specific topics, max lag,id2word and init topic-word distribution.Details are shown in real_data_code.py;
2. real_data_code.py is the main program of SJDTM for analysing real datasets;
3.DTMpart.py is used for updating topic-word distribution beta in real_data_code.py;
4.real_data_evalution_metrics.py  has the evalution_matrics to measure the performance of SJDTM.
Users can directly run real_data_code.py in realdata folder.
