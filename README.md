# ga-img-captioning

#### Installation

run pkill python before & after

conda:
mkl-service?

pip:
...

#### Running

Caution: when you change mode from nets to seeds or vice versa it is possible
that you will get some errors because of task caching by the redis relay. Just
restart the experiment or wait some iterations.

Caution: when continuing from population make sure to load infos and models
from same iteration!


#### Profiling
https://docs.python.org/3.7/library/profile.html

```
import pstats
from pstats import SortKey

w = pstats.Stats('profile_worker.txt')
w.sort_stats(SortKey.TIME).print_stats(10)
```