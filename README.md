# ga-img-captioning

#### References

Based on & took code from:
- https://github.com/ruotianluo/self-critical.pytorch
- https://github.com/uber-research/deep-neuroevolution
- https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66


#### Installation

run pkill python before & after

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#viewing-a-list-of-your-environments
conda:
mkl-service?

pip:
...

Needs ts from moreutils and tee (unix packages)
https://superuser.com/questions/1174408/can-you-prefix-each-line-written-with-tee-with-the-current-date-and-time
https://git-scm.com/book/en/v2/Git-Tools-Submodules

#### Distributed application

https://seba-1511.github.io/tutorials/intermediate/dist_tuto.html
https://pytorch.org/docs/stable/multiprocessing.html
https://stackoverflow.com/questions/48822463/how-to-use-pytorch-multiprocessing

https://bugs.python.org/issue33725
"The underlying problem is that macOS system frameworks (basically anything higher level than libc) are not save wrt fork(2) and fixing that appears to have no priority at all at Apple."

#### Running

Truncation should be smaller than population size!

Caution you need population_size * <size of one param.pth file> disk space! For a 3M param network with an 11MB param file this is ~12GB.

if num_val_batches not in config: val on entire val set

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

#### Useful commands

`ls dir/ | wc -l` : number of files in dir
`du -h` : disk usage of dir and subdirs

https://unix.stackexchange.com/questions/1314/how-to-set-default-file-permissions-for-all-folders-files-in-a-directory

