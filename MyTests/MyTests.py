import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

a = [50, 100, 150, 20, 10] #np.random.normal(loc=10, scale =3, size=100)
b = [110, 189, 1, 29, 1] #np.random.normal(loc=9, scale =3, size=100)

a = [50, 100, 150, 90, 111] #np.random.normal(loc=10, scale =3, size=100)
a = [50, 100, 151] #np.random.normal(loc=10, scale =3, size=100)
b = [50, 100, 150] #np.random.normal(loc=9, scale =3, size=100)

a = np.random.geometric(p=0.35, size=500)
b = np.random.normal(loc=10, scale =3, size=500)
b = np.random.geometric(p=0.35, size=500)

print("")
print("T_test: ", stats.ttest_ind(a,b))
print("A (mean, std): ", np.mean(a), np.std(a))
print("B (mean, std): ", np.mean(b), np.std(b))
