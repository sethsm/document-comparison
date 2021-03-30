# ###############################################################################
# ###############################################################################
# #####                                                                     ##### 
# #####                     Semi-synthetic data settings                    #####
# #####                                                                     ##### 
# ###############################################################################
# ###############################################################################

# import sys
# import time
# import scipy.io
# import scipy.sparse
# import scipy.stats
# import numpy as np
# import os

# main_dir = '/home/centos/TopicModel'
# sys.path.insert(0, main_dir+'/Simulation/py_code/LOVE')
# sys.path.insert(0, main_dir+'/Simulation/py_code/Arora')
# sys.path.insert(0, main_dir+'/Simulation/py_code')
# sys.path.insert(0, main_dir+'/Simulation/py_code/Sp_Top')

# import general_sim_setting as sim
# import plotting as myplot
# import lda
# import LOVE

# class Params:

#     def __init__(self, filename):
#         self.log_prefix=None
#         self.checkpoint_prefix=None
#         self.seed = int(time.time())

#         for l in file(filename):
#             if l == "\n" or l[0] == "#":
#                 continue
#             l = l.strip()
#             l = l.split('=')
#             if l[0] == "log_prefix":
#                 self.log_prefix = l[1]
#             elif l[0] == "max_threads":
#                 self.max_threads = int(l[1])
#             elif l[0] == "eps":
#                 self.eps = float(l[1])
#             elif l[0] == "checkpoint_prefix":
#                 self.checkpoint_prefix = l[1]
#             elif l[0] == "new_dim":
#                 self.new_dim = int(l[1])
#             elif l[0] == "seed":
#                 self.seed = int(l[1])
#             elif l[0] == "anchor_thresh":
#                 self.anchor_thresh = int(l[1])
#             elif l[0] == "top_words":
#                 self.top_words = int(l[1])
                
# # Combine the old result with new result
                
# def comb_result(old_result, new_result, ind = 6):
#     old_result['l1-loss'][:,ind] = new_result['l1-loss'][:,0]
#     old_result['l1-loss-std'][:,ind] = new_result['l1-loss-std'][:,0]
#     old_result['inf1-loss'][:,ind] = new_result['inf1-loss'][:,0]
#     old_result['inf1-loss-std'][:,ind] = new_result['inf1-loss-std'][:,0]
#     old_result['time'][:,ind] = new_result['time'][:,0]
#     return old_result

# ###############################################################################
# ####                       New York Times dataset   
# ###############################################################################

# #### load in the NYT data matrix (after removing rare words and stopping words)

# #vocab = file("/Users/mike/Documents/Mike/Projects/Topic models/Simulation/py_code/Arora/vocab.nytimes.txt.trunc").read().strip().split()
# #data_dir = "/Users/mike/Documents/Mike/Projects/Topic models/Simulation/py_code/Arora/sparse_nytimes.txt.trunc.mat"
# #M = scipy.io.loadmat(data_dir)['M']
# #X = M.toarray().transpose().astype(int)   ##   [# of doc,  # of words]
# #p = X.shape[1]

# #Ns = np.sum(X, 1).astype(float)   # 209.21

# ###############################################################################
# ####                      Use Top to obtain the initial A
# ###############################################################################

# # Remove documents with length <= cut_off
# #cut_off = 10
# #X = X[Ns >= cut_off, ]

# #res = LOVE.LOVE(X.transpose(), delta = [15.5], average = False, 
# #                verbose = True, estA = True)

# #K, true_A = res['K'], res['A']

# # out_dir = main_dir + "/Simulation/py_code/Arora/NYT_Top_K_101"
# #scipy.io.savemat(out_dir, {'A' : true_A}, oned_as='column')

# # K = 101
# # true_A = scipy.io.loadmat(out_dir)['A']

# #nonzero_elements = [np.count_nonzero(true_A[j,]) for j in range(p)]
# #anchor_indices = [i for i, x in enumerate(nonzero_elements) if x == 1]   ## 200+ anchor words


# ###############################################################################
# ####                      Use LDA to obtain the initial A
# ###############################################################################

# K = 100
# #
# ##model = lda.LDA(n_topics=K, n_iter=1000, random_state=1)
# ##model.fit(X) 
# ##true_A = model.topic_word_.transpose()
# #                        
# out_dir = main_dir + "/Simulation/py_code/Arora/NYT_LDA_A_K100"
# ##scipy.io.savemat(out_dir, {'A' : true_A}, oned_as='column')
# #
# true_A = scipy.io.loadmat(out_dir)['A']
   
# ##myplot.plt.hist(np.sum(true_A <= 1e-6, 1))

# params = Params(main_dir + "/Simulation/py_code/Arora/settings.example")
# params.max_threads = 0
# params.anchor_thresh = 0

# ns = [30000, 40000, 50000, 60000, 70000]
# N = [300]


# #option, param = ["log-normal", 0.2]
# #option, param, anchor_homo = ["unif_pure", 0.3, True]
# #option, param = ["diri", 0.03]
# #option, param = ["unif", int(float(K)/2)]

# settings = [["diri", 0.03], ["diri", 0.3], ["log-normal", 0.1], ["log-normal", 0.3]]
# methods = ["LOVE-fast", "LOVE-sparse", "Recover_L2-100", "Recover_KL-100", "Sparse-100", "LDA-100"]

# for setting in settings:
#     option, param = setting

#     res = sim.sim_semi_syn_parallel(N, ns, true_A, option, param, params, 
#                                 sim_rep = 25, methods = methods)
    
#     outfile_ns = main_dir + "/Simulation/results/syn-NYT-m0" + option + "-" + str(param) 
#     scipy.io.savemat(outfile_ns, res)

#     print "Finishing the case with no anchor word for setting:" + option + str(param) + "......\n"

#     m = 1
#     A_anchor = np.vstack((np.repeat(np.diag(np.max(true_A, 0)), m, 0), true_A))
#     A_anchor = A_anchor / np.sum(A_anchor, 0)

#     res_anchor_1 = sim.sim_semi_syn_parallel(N, ns, A_anchor, option, param, params, 
#                                            sim_rep = 25, methods = methods)

#     outfile_anchor_1 = main_dir + "/Simulation/results/syn-NYT-m1" + option + "-" + str(param) 
#     scipy.io.savemat(outfile_anchor_1, res_anchor_1)

#     print "Finishing the case with one anchor word for setting:" + option + str(param) + "......\n"

#     m = 5

#     A_anchor = np.vstack((np.repeat(np.diag(np.max(true_A, 0)), m, 0), true_A))
#     A_anchor = A_anchor / np.sum(A_anchor, 0)

#     res_anchor_5 = sim.sim_semi_syn_parallel(N, ns, A_anchor, option, param, params, 
#                                            sim_rep = 25, methods = methods)

#     outfile_anchor_5 = main_dir + "/Simulation/results/syn-NYT-m5" + option + "-" + str(param) 
#     scipy.io.savemat(outfile_anchor_5, res_anchor_5)

#     print "Finishing the case with five anchor words for setting:" + option + str(param) + "......\n"



# sys.exit()
