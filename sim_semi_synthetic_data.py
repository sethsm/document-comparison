###############################################################################
###############################################################################
#####                                                                     ##### 
#####                     Semi-synthetic data settings                    #####
#####                                                                     ##### 
###############################################################################
###############################################################################

import sys
import time
import scipy.io
import scipy.sparse
import scipy.stats
import numpy as np

# main_dir = '/home/centos/TopicModel/Simulation'
# sys.path.insert(0, main_dir + '/py_code/LOVE')
# sys.path.insert(0, main_dir + '/py_code/Arora')
# sys.path.insert(0, main_dir + '/py_code')
# sys.path.insert(0, main_dir + '/py_code/Sp_Top')

import general_sim_setting as sim
import plotting as myplot

class Params:

    def __init__(self, filename):
        self.log_prefix=None
        self.checkpoint_prefix=None
        self.seed = int(time.time())

        for l in file(filename):
            if l == "\n" or l[0] == "#":
                continue
            l = l.strip()
            l = l.split('=')
            if l[0] == "log_prefix":
                self.log_prefix = l[1]
            elif l[0] == "max_threads":
                self.max_threads = int(l[1])
            elif l[0] == "eps":
                self.eps = float(l[1])
            elif l[0] == "checkpoint_prefix":
                self.checkpoint_prefix = l[1]
            elif l[0] == "new_dim":
                self.new_dim = int(l[1])
            elif l[0] == "seed":
                self.seed = int(l[1])
            elif l[0] == "anchor_thresh":
                self.anchor_thresh = int(l[1])
            elif l[0] == "top_words":
                self.top_words = int(l[1])
                
                
# Combine the old result with new result
                
def comb_result(old_result, new_result, ind = 6):
    old_result['l1-loss'][:,ind] = new_result['l1-loss'][:,0]
    old_result['l1-loss-std'][:,ind] = new_result['l1-loss-std'][:,0]
    old_result['inf1-loss'][:,ind] = new_result['inf1-loss'][:,0]
    old_result['inf1-loss-std'][:,ind] = new_result['inf1-loss-std'][:,0]
    old_result['time'][:,ind] = new_result['time'][:,0]
    return old_result


###############################################################################
####                       NIPs dataset   
###############################################################################

#### load in the NIPS data matrix (after removing rare words and stopping words)
#### the rare words are those appearing in less than 150 documents 
    
#vocab = file(main_dir + "/Simulation/py_code/Arora/vocab.nips.txt.trunc").read().strip().split()
#M = scipy.io.loadmat(main_dir + "/Simulation/py_code/Arora/sparse_nips.txt.trunc.mat")['M']
#X = M.toarray().transpose().astype(int)   ##   [# of doc,  # of words]
#p = X.shape[1]

#Ns = np.sum(X, 1).astype(float)   # 847.11
#X_tran = X.transpose()

#import matplotlib.pyplot as plt
#plt.hist(Ns)


#### Remove documents with length <= cut_off
#cut_off = 150
#X = X[Ns >= cut_off, ]

K = 100
# K = 120

#model = lda.LDA(n_topics=K, n_iter=1000, random_state=1)
#model.fit(X) 
#true_A = model.topic_word_.transpose()
out_dir = main_dir + "/py_code/Arora/LDA_A_K100"
# out_dir = main_dir + "/py_code/Arora/Top_K_120"
#scipy.io.savemat(out_dir, {'A' : true_A}, oned_as='column')

true_A = scipy.io.loadmat(out_dir)['A']
   
params = Params(main_dir + "/py_code/Arora/settings.example")
params.max_threads = 0
params.anchor_thresh = 0

ns = [2000, 4000, 6000, 8000, 10000]
N = [850]

#option, param = ["log-normal", 0.2]
#option, param, anchor_homo = ["unif_pure", 0.3, True]
#option, param = ["diri", 0.03]
#option, param = ["unif", int(float(K)/2)]


settings = [['diri', 0.03], ["diri", 0.3], ["log-normal", 0.1], ["log-normal", 0.3]]
methods = ["LOVE-fast", "LOVE-sparse", "Recover_L2-100", "Recover_KL-100", "Sparse-100", "LDA-100"]

for setting in settings:
    option, param = setting

    res = sim.sim_semi_syn_parallel(N, ns, true_A, option, param, params, 
                                sim_rep = 25, methods = methods)
    
    outfile_ns = main_dir + "/results/syn-Nips-m0" + option + "-" + str(param) 
    scipy.io.savemat(outfile_ns, res)

    print "Finishing the case with no anchor word for setting:" + option + str(param) + "......\n"

    m = 1
    A_anchor = np.vstack((np.repeat(np.diag(np.max(true_A, 0)), m, 0), true_A))
    A_anchor = A_anchor / np.sum(A_anchor, 0)

    res_anchor_1 = sim.sim_semi_syn_parallel(N, ns, A_anchor, option, param, params, 
                                           sim_rep = 25, methods = methods)

    outfile_anchor_1 = main_dir + "/results/syn-Nips-m1" + option + "-" + str(param) 
    scipy.io.savemat(outfile_anchor_1, res_anchor_1)

    print "Finishing the case with one anchor word for setting:" + option + str(param) + "......\n"

    m = 5

    A_anchor = np.vstack((np.repeat(np.diag(np.max(true_A, 0)), m, 0), true_A))
    A_anchor = A_anchor / np.sum(A_anchor, 0)

    res_anchor_5 = sim.sim_semi_syn_parallel(N, ns, A_anchor, option, param, params, 
                                           sim_rep = 25, methods = methods)

    outfile_anchor_5 = main_dir + "/results/syn-Nips-m5" + option + "-" + str(param) 
    scipy.io.savemat(outfile_anchor_5, res_anchor_5)

    print "Finishing the case with five anchor words for setting:" + option + str(param) + "......\n"



sys.exit()
