from Paths import *
from Basics import DTImaps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
FA=DTImaps(In.img,In.bval,In.bvec,tracto=False)
plt.hist(np.ravel(FA))
sns.kdeplot(np.ravel(FA),shade=True)
plt.show()
