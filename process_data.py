
import pandas as pd
import scipy.sparse
import numpy as np
import math


# load data into data frame
kos_docword = pd.read_csv('data/docword.kos.txt', sep = ' ', names = ['doc_id','word_id','count'], skiprows=3)

# pivot to correct format, then convert to numpy array (that contains NaNs)
kos_docword = kos_docword.pivot(index = 'word_id', columns = 'doc_id', values='count').to_numpy() 

# convert to sparse array
indices = np.nonzero(~np.isnan(kos_docword))
kos_docword = scipy.sparse.coo_matrix((kos_docword[indices], indices), shape=kos_docword.shape) 

# save sparse array
scipy.sparse.save_npz('data/sparse_matrix.npz', kos_docword)