from Paths import *
from Basics import preproccesing ,fahist ,segmentation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

b0_mask ,mask ,affine =preproccesing(In.img,save=False)
fahist(b0_mask ,mask ,affine,In.bval,In.bvec,In.t1)


'''
sns.set()
FA=DTImaps(In.img,In.bval,In.bvec,tracto=False)
plt.hist(np.ravel(FA))
sns.kdeplot(np.ravel(FA),shade=True)
plt.show()
'''
