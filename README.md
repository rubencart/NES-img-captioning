# Neural image captioning with (natural) evolution strategies

This repo implements two algorithms that train the weights of an LSTM network for MSCOCO image captioning: NIC-ES, an 
evolution strategy inspired by the one in LINK, and NIC-NES, a natural evolution strategy inspired by LINK.


### Installation

requirements: 
- (mini)conda
- python 3.7
- java
- tmux, tee, screen

```
git clone git@github.com:rubencart/es-img-captioning.git
cd es-img-captioning
git submodule init
git submodule update
cd cococaption && git checkout master && git pull
cd cider && git checkout master && git pull 
mkdir logs && mkdir output && mkdir data

conda create -n cp3env python=3.7       # don't change the name of the virtual env!
conda activate cp3env
conda install psutil
conda install -c pytorch pytorch torchvision
pip install -r requirements.txt

# (you need gcc, otherwise run `sudo apt install gcc`)
# install redis:
curl -O http://download.redis.io/redis-stable.tar.gz
tar xzvf redis-stable.tar.gz
cd redis-stable
make install
```

TODO: get data from https://www.dropbox.com/sh/65735ziv4w7ky7o/AAC5RFJqBUlGPIvYjEC8ENwAa?dl=0


### Starting an experiment

To run locally: first choose settings in `experiments/<exp>.json`, and then do the following.
```
screen -S redis
# next 3 steps can be skipped but are recommended if redis displays a warning about thp pages
sudo su
echo never > /sys/kernel/mm/transparent_hugepage/enabled
# when `WARNING: The TCP backlog setting of 511 cannot be enforced because /... is set to the lower value of 128.
sysctl -w net.core.somaxconn=65535
exit
. src/scripts/local_run_redis.sh

screen -S es
# <path_to_run_script> followed by arguments:
# <path_to_experiment_json>
# <nb of workers>: set to -1 to have the launched process be the only worker
# (optional) <id>: any chosen number to id outputfiles 
# (optional) <redis tcp port> to use: 6379 | 6380 | 6381 | 6382
. src/scripts/local_run_exp.sh experiments/mscoco_es.json 92 123
# or
. src/scripts/local_run_exp.sh experiments/mscoco_nes.json 92 123
```

This will start a screen that runs redis and another screen where the experiment runs. In the experiment screen the output of
the master process and worker processes is logged in a split-screen with tmux. Running an experiment creates a master outputfile
and a worker outputfile in `output/` like `<id>_master_outputfile.txt` where all logs are written to.
A directory in `logs/` is also created, which can be recognized by the pid of the started master process. Snapshots,
checkpoints and plots will be written to that dir. The parameters of the current population and all offspring are also stored
there. So an experiment creates:
```
./
    output/
        <id>_master_outputfile.txt
        <id>_worker_outputfile.txt
    logs/
        <name_depends_on_algo>_pid/     # eg nic_nes_mscoco_fc_caption_84267/
            snapshot/                   # every <snapshot_freq> a snapshot is saved to this dir
                plots
                experiment.json                 # copy
                z_info_e<nb>_i<nb>-<nb>.json    # checkpoint with scores and stats, experiments can be started from
                                                # checkpoint json's like these.
            models/
                best/       # stores best models so far
                offspring/  # stores population + offspring: only for NIC-ES
                current/    # stores current theta: only for NIC-NES
            optimizer/      # stores adam/sgd optimizer state: only for NIC-NES
```

NIC-ES stores the parents and the population on disk. This is an ... TODO

To run with qsub, choose settings in json files as below, choose PBS settings in `src/scripts/local_run_exp.pbs`, and then:
```
qsub -v "algo=es,port=6380,workers=36,exp=experiments/mscoco_es.json,id=110" src/scripts/local_run_exp.pbs
```

Make sure that when you start an experiment from a pretrained network (with `"from_single": "./path"`), 
the dimensions of the input pretrained network and the specified dimensions in the experiment json match.
More specifically, what has to match is:
- "input_encoding_size", "rnn_size", "fc_feat_size"
- the data on which it was trained
- "layer_n", "layer_n_affine": so whether layer norm (with/without affine transformation params) was used to pretrain
    the input network
- "vbn", "vbn_e", "vbn_affine": whether (virtual) batch norm was used (and how)

I recommend to not use layer/batch normalisation (was not tested a lot).


#### NIC-ES

Example json experiment file:
```
{
  "algorithm": "nic_es",                # nic_es | nic_nes
  "config": {
    "eval_prob": 0.006,                 # prob that worker will do evaluation run on validation set instead of
                                        # evolve/mutate run
    "noise_stdev": 0.002,
    "snapshot_freq": 10,                # every snapshot_freq a checkpoint and plots will be saved to logs/
    "batch_size": 128,
    
    "ref_batch_size": 0,                # batch size for reference batch for virtual batch norm

    "num_val_items": 5000,
                                        # use either patience or schedule, or none
    "patience": 0,                      # anneal noise_stdev and batch size if for patience iterations
                                        # no new elite gets one of the num_elites best scores so far
    "schedule_start": 200,              # nb of iteration to start the annealing schedule in
    "schedule_limit": 200,              # anneal every X iterations
    "stdev_divisor": 1.414,
    "bs_multiplier": 2                  
  },

  "policy_options": {
    "net": "fc_caption",                # name of network to be used. Use "mnist" for mnist experiment
    "fitness": "greedy",                # Fitness function used. Only for MSCOCO experiment, mnist
                                        # experiments automatically use XENT
                                        # greedy | sample | sc_loss | self_critical | greedy_logprob
    "vbn": false,                       # use virtual batch norm

    "model_options": {
      "safe_mutation_underflow": 0.2,   # if applicable, calculated sensitivities are capped (below),
                                        # to avoid dividing the mutation vector with very small values
      "safe_mutations": "SM-G-SUM",     # "SM-G-SUM" | "SM-PROPORTIONAL" | "SM-VECTOR" | ""
      "safe_mutation_vector": "",       # set to path if SM-VECTOR is used, eg "./data/sensitivity.pt"

      "vbn_e": false,                   # whether to also use batch norm after the embedding layers
                                        # (instead of only in the LSTM)
      "vbn_affine": false,              # use affine transformation (extra learnable params) with batch norm
      "layer_n": false,                 # layer norm
      "layer_n_affine": false,

      "input_encoding_size":  128,      # size of image and word embeddings
      "rnn_size": 128,                  # hidden state size of LSTM
      "fc_feat_size": 2048              # input image feature size
    }
  },

  "dataset": "mscoco",                  # mnist | mscoco

  "nb_offspring": 1000,                 # lambda
  "population_size": 50,                # mu (only for NIC-ES): make sure mu < lambda
  "selection": "uniform",               # selection of parents (only for NIC-ES): uniform | tournament
  "tournament_size": 0,

  "num_elites": 3,                      # E
  "num_elite_cands": 2,                 # C (only for NIC-ES)
  
  # - to start from randomly initialized pop: set to "_from_single" and "_from_infos"
  # - to start from one single pretrained NW: set to "from_single" and "_from_infos"
  # - starting from >1 pretrained NW: update coming soon
  # - to start from checkpoint saved by previous experiment: set to "_from_single" and "from_infos"
  # path to single .pth file with params of pretrained NW
  "from_single": "logs/logs_xent_fc_128/checkpt/model-best.pth",
  # path to checkpoint .json of previous experiment 
  "_from_infos": "logs/ga_mscoco_fc_caption_24173/snapshot/z_info_e6_i450-442.json",

  "caption_options": {
    "input_json": "data/cocotalk.json",         # path to json with captions
    "input_fc_dir": "data/cocobu_fc",           # path to input image features, set to data/cocotalk_fc for
                                                # resnet instead of faster rcnn features
    "input_label_h5": "data/cocotalk_label.h5"  # path to h5 file
  }
}
```

#### NIC-NES

Example experiment json file:
```
{
  "algorithm": "nic_nes",
  "config": {
    "eval_prob": 0.003,
    "noise_stdev": 0.005,
    "snapshot_freq": 10,
    "batch_size": 128,

    "val_batch_size": 256,
    "num_val_items": 5000,

    "patience": 0,
    "schedule_start": 200,
    "schedule_limit": 100,
    "stdev_divisor": 1.414,
    "bs_multiplier": 1.414,
    "stepsize_divisor": 1,              # annealing factor for the optimizer learning rate

    "ref_batch_size": 2,       
    "l2coeff": 0.0002,                  # L2 weight regularization coeff
    "single_batch": false               # whether NIC-NES evaluates all deltas on the same batch or not
  },

  "policy_options": {
    "net": "fc_caption",
    "fitness": "greedy_logprob",
    "vbn": false,

    "model_options": {
      "safe_mutations": "SM-G-SUM",
      "safe_mutation_vector": "",
      "safe_mutation_underflow": 0.1,

      "vbn_e": false,
      "vbn_affine": false,
      "layer_n": false,
      "layer_n_affine": false,

      "input_encoding_size":  128,
      "rnn_size": 128,
      "fc_feat_size": 2048
    }
  },

  "optimizer_options": {
    "type": "adam",                     # which optimizer to use: sgd | adam
    "args": {
      "stepsize": 0.002                 # learning rate, eta
    }
  },

  "dataset": "mscoco",

  "nb_offspring": 1000,
  "num_elites": 1,

  "from_single": "../instance_m_gpu_logs/vbn_5_xent_128_checkpt_c.695/model-best.pth",
  "_from_infos": "logs/es_mscoco_fc_caption_32313/snapshot/z_info_e1_i2-56643.json",

  "caption_options": {
    "input_json": "data/cocotalk.json",
    "input_fc_dir": "data/cocobu_fc",
    "input_label_h5": "data/cocotalk_label.h5"
  }
}
```

### Resuming an experiment

Resuming an experiment can be done by setting the `"from_infos"` option in the experiment json to the path in the 
repo dir of the checkpoint json you want to start from. Make sure to also change `"from_single"` to something like 
`"_from_single"`. E.g.:
```
"from_infos": "./logs/nic_nes_mscoco_fc_caption_84267/snapshot/z_info_e1_i1-885.json"
"_from_single": "..."
``` 

When NIC-ES is resumed all parents are copied, so apart from having to generate the offspring that was already generated
when execution was interrupted again, execution can resume as if it had never stopped. The same counts for NIC-NES.
All running stats are also copied so the future checkpoints are as if they were from 1 single (uninterrupted) run.

TODO also not taken: dataloader! so could be possible that one cycle through dataset is interrupted

When resuming an experiment part of the settings are taken from the checkpoint .json and part of the settings are 
taken from the used experiment .json, which can be confusing. Most importantly:
- mutation stdev
- batch size
- learning rate
- running stats like scores in all previous iterations, time elapsed,...
- iteration nb, how many bad iterations have happened (if patience),...
are used from the checkpoint. 

Settings like the mutation type, fitness, the patience or schedule, number of offspring, population size,... are taken
from the used experiment .json.

### Overview of code

TODO scripts/ to ../ !

The repo structure:
```
cider/                              # submodule to forked repo that computes cider scores
cococaption/                        # submodule to forked repo that computes diff. scores
data/                               # data folder (you should put the data in here)
experiments/                        # experiment json files
                                    # create your own/adjust the existing ones
logs/
output/
pretrained/                         # contains some pretrained models
redis_config/                       # config files for redis
src/                                # the actual code
    algorithm/
        nic_es/
            experiment.py           # NIC-ES specific subclass of tools.Experiment
            iteration.py            # NIC-ES specific subclass of tools.Iteration
            nic_es_master.py        # actual NIC-ES code
            nic_es_worker.py        # actual NIC-ES code, evolve + evaluate methods
            
        nic_nes/
            experiment.py           # idem
            iteration.py            # idem
            nic_nes_master.py       # idem
            nic_nes_worker.py       # idem
            optimizers.py           # sgd and adam optimizers used by NIC-NES
        
        tools/                      # supporting code
            experiment.py           # experiment base class with experiment-wide settings and dataloaders
            iteration.py            # iteration base class that keeps population, manages schedule/patience
            podium.py               # class that keeps best individuals so far
            setup.py                # setup tools for master and workers
            snapshot.py             # save snapshot/checkpoint to logs
            statistics.py           # keeps running statistics like time, scores,...
            utils.py                # bunch of util methods
            
        nets.py                     # neural net base class with evolve/init methods
        policies.py                 # policy base class with setup/load/save methods
        safe_mutations.py           # where sensitivity is calculated
    
    captioning/
        dataloader.py               # loads data from data/
        dataloaderraw.py            TODO????
        eval_utils.py               # used to calculate cider scores on validation set
        experiment.py               # captioning specific subclass of tools.Experiment
        fitness.py                  # different fitness functions, used by captioning/policies.py
        nets.py                     # definition of the LSTM and word embedding layers
        policies.py                 # implementation of rollout and validation_score methods
    
    classification/                 # cfr. captioning
    
    scripts/                        # launch scripts
        local_env_setup.sh          # launch miniconda env
        local_profile_cpu_exp.sh    # cpu profile run
        local_run_exp.pbs           # PBS launch script for qsub, contains settings like max runtime and nb procs
        local_run_exp.sh            # local experiment launch file
        local_run_redis.sh          # launch redis master & relay
    
    dist.py                         # redis communication stuff
    test.py                         # code for evaluation on test splits
    main.py                         # main python entry point (launches algorithms)
    
```

So to
- change NIC-(N)ES: adjust algorithm.nic_(n)es.
- create a new algorithm: create a master & worker, create a (or reuse an existing) Experiment & Iteration subclass.
- apply to new dataset: create a Policy subclass that implements the rollout and validation_score methods,
create a PolicyNet (algorithm.nets) subclass with the neural net definition, create an Experiment subclass
with the appropriate dataloaders.
- train a different NN architecture for MSCOCO with the existing algorithms: define a new CaptionModel subclass
in captioning.nets and register it in algorithm.policies.

### Pretrained models

TODO!

### Profiling

TODO?

First run for example (with redis already running): `. src/scripts/local_profile_cpu_exp.sh nic_nes experiments/mscoco_nes.json`.
Then, after execution is finished, go into a python environment and do the following. Consult the cProfile documentation
for more info.

```
import pstats
from pstats import SortKey

w = pstats.Stats('output/profile_worker.txt')
w.sort_stats(SortKey.TIME).print_stats(10)
m = pstats.Stats('output/profile_master.txt')
m.sort_stats(SortKey.TIME).print_stats(10)
```

### Remarks

Some other remarks:
- Order is important when you use safe mutations based on param type!
- For NIC-ES you need AT LEAST nb_offspring * size of one param.pth file disk space! 
For a 3M param network with an 11MB param file and nb_offspring=1000 this is ~12GB. 
Recommended is at least 2-3 times this space to have some margin.
- Sometimes when starting an experiment you might get errors because redis stores results from previous experiments, 
which might reach your master or workers. Try restarting the experiment when this happens. Sometimes restarting redis also
helps (this clears the cache).


### References

TODO LICENSE!!!!!!

Based on & took code from:
- https://github.com/ruotianluo/self-critical.pytorch
- https://github.com/uber-research/deep-neuroevolution
- https://github.com/openai/evolution-strategies-starter/
- https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66
- https://github.com/uber-research/safemutations
- https://github.com/vrama91/cider
- https://github.com/tylin/coco-caption

Thanks to everyone who worked on these projects. 