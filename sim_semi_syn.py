import scipy.sparse
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import gensim

###############################################################################
#####                            PREPARE DATA                             #####
###############################################################################


docword = scipy.sparse.load_npz('data/kos_sparse_matrix.npz')
x = docword.toarray().transpose().astype(int) #x.shape = [# of docs, # of words]
p = x.shape[1]
print(x.shape)

doc_lens = np.sum(x,1)
#plot document length distribution
#plt.hist(doc_lens)

# Remove words that appear in less than cutoff number of docs
cutoff = 150
x = x[doc_lens >= cutoff, ]


###############################################################################
#####                            PREPARE DATA                             #####
###############################################################################



