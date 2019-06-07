# coding: utf-8
def tournament(pop, t, offspr):    return [min(rs.choice(np.arange(pop), t, replace=False)) for _ in range(offspr)]
def count_in_tournament(p, t, o):
    tourn = tournament(p, t, o)
    counts = [tourn.count(c) for c in range(p)]
    return counts
    
def avg_c_in_t(p, t, o, x):
    cts = np.empty([x, p], dtype=float)
    for i in range(x):
        cts[i] = count_in_tournament(p, t, o)
    return cts.mean(0)
    
import numpy as no
import numpy as np
np.asarray([[1, 2], [3, 4]])
a = np.asarray([[1, 2], [3, 4]])
a.mean(0)
a.mean(1)
avg_c_in_t(5, 2, 10, 100)
rs = np.random.RandomState()
avg_c_in_t(5, 2, 10, 100)
avg_c_in_t(150, 5, 1000, 2)
avg_c_in_t(150, 5, 1000, 100)
from matplotlib import pyplot as plt
fig = plt.figure()
for i in range(10):
    plt.plot(avg_c_in_t(150, i, 1000, 50), label=i)
    
for i in range(2, 10):
    plt.plot(avg_c_in_t(150, i, 1000, 50), label=i)
    
    
plt.legend(loc='upper left')
plt.savefig('logs/tournament_sizes.png', format='png')
fig.close()
plt.close()
fig = plt.figure()
for i in range(3, 16, 2):
    plt.plot(avg_c_in_t(150, i, 1000, 100), label=i)
    
    
plt.legend(loc='upper right')
plt.savefig('logs/tournament_sizes.pdf', format='pdf')
plt.close()
fig = plt.figure()
for i in [3, 5, 7, 9, 11]:
    plt.plot(avg_c_in_t(150, i, 1000, 100), label=i)
    
    
plt.legend(loc='upper right')
plt.savefig('logs/tournament_sizes_2.pdf', format='pdf')
get_ipython().run_line_magic('save', 'tournament_sizes.py')
