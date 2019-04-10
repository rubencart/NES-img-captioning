# ga-img-captioning

#### Installation

run pkill python before & after

conda:
mkl-service?

pip:
...

#### Running

Example json:
```
{
  "mode":  "nets",
  "config": {
    "eval_prob": 0.02,
    "noise_stdev": 0.01,
    "snapshot_freq": 5,
    "batch_size": 64,
    "patience": 20,
    "stdev_decr_divisor": 1.0
  },
  "dataset": "mnist",
  "net": "mnist",
  "population_size": 100,
  "truncation": 30,
  "num_elites": 1,
  // ONE OF:
  "from_population": {
    "infos": "logs/es_nets_88450/z_info_e1_i66-938.json",
    "models": "logs/es_nets_88450/z_parents_params_e1_i66-938_r0.42.tar"
  },
  // OR
  "from_single": "snapshots/sgd_69966/params_acc97.pth",
}

```

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