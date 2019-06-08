# ga-img-captioning

#### References

Based on & took code from:
- https://github.com/ruotianluo/self-critical.pytorch
- https://github.com/uber-research/deep-neuroevolution
- https://github.com/openai/evolution-strategies-starter/
- https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
- https://github.com/uber-research/safemutations


#### Installation

requirements: python 3.7, pytorch, torchvision, java, redis-server
tmux, tee

git clone ...
git submodule init
git submodule update

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#viewing-a-list-of-your-environments
conda create -n cp3env python=3.7
conda activate cp3env
conda install psutil
conda install -c pytorch pytorch torchvision
pip install -r requirements.txt

(you need gcc, otherwise run `sudo apt install gcc` and then `make distclean` and `make`)
redis:
curl -O http://download.redis.io/redis-stable.tar.gz
tar xzvf redis-stable.tar.gz
cd redis-stable

Needs ts from moreutils and tee (unix packages)
https://superuser.com/questions/1174408/can-you-prefix-each-line-written-with-tee-with-the-current-date-and-time
https://git-scm.com/book/en/v2/Git-Tools-Submodules

And you need data in .data/, mkdir logs, mkdir output

#### Distributed application

https://seba-1511.github.io/tutorials/intermediate/dist_tuto.html
https://pytorch.org/docs/stable/multiprocessing.html
https://stackoverflow.com/questions/48822463/how-to-use-pytorch-multiprocessing

https://bugs.python.org/issue33725
"The underlying problem is that macOS system frameworks (basically anything higher level than libc) are not save wrt fork(2) and fixing that appears to have no priority at all at Apple."

#### Running

```
screen -S redis
sudo su
echo never > /sys/kernel/mm/transparent_hugepage/enabled
exit
. src/scripts/local_run_redis.sh

screen -S ga
. src/scripts/local_run_exp.sh ga experiments/mscoco_ga.json 58
```

Caution order is important when you use safe mutations based on param type!

Truncation should be smaller than population size!
elite_cands should also be smaller than pop_size and truncation

Caution you need population_size * <size of one param.pth file> disk space! For a 3M param network with an 11MB param file this is ~12GB.

if num_val_batches not in config: val on entire val set

Caution resuming an experiment exactly is not always possible (state of optimizers in ES experiment for example)

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
    "stdev_divisor": 1.0
  },
  "dataset": "mnist",
  "net": "mnist",
  "population_size": 100,
  "truncation": 30,
  "num_elites": 1,
  // ONE OF:
  "from_infos": {
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

w = pstats.Stats('output/profile_worker.txt')
w.sort_stats(SortKey.TIME).print_stats(10)
```

#### Useful commands

`ls dir/ | wc -l` : number of files in dir
`du -h` : disk usage of dir and subdirs

https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file
https://stackoverflow.com/questions/876239/how-can-i-redirect-and-append-both-stdout-and-stderr-to-a-file-with-bash

https://unix.stackexchange.com/questions/1314/how-to-set-default-file-permissions-for-all-folders-files-in-a-directory

#### GENIUS (VSC) machines ####

https://vscentrum.be/cluster-doc/running-jobs/credit-system-basics#how-to-request-introduction-credits
https://www.vscentrum.be/cluster-doc/running-jobs/specifying-requirements
Add `#PBS -A ltutorial_liir` to put on LIIR?