# tf_rlib

### Ideas 

- goal
    - This libs is target for fastest AI development within tf2, we only implement the most general algorithms here. Therefore, the flexibility is limited. All we need is manipulate the flags, then we could achieve the SOTA performance on most of common tasks.
- coding ideas
    - based on the facts that DL tasks are diferent, it's not necessary to use single api solve all the problems, for examples:
        - about loss: we won't use dice loss for classification.
        - about data: we won't use labeled data in self-supervised/unsupervised training.
        - about structure: we won't combine resblock, resnextblock, preactblock, pyramidnet into a new one.
    - this is a task-specific lib, every task has its own best practice model and runner, we sacrify some flexibility to ease the development efforts.
    - we use absl.flags to manage all hyper-perameters between all modules, then we could apply grid search through bash scripts easily.
    - this lib is runnable on both interactive mode (jupyter) and cmd mode.
    - we use tensorboard and tqdm to visulize the training progress.
    - APIs shared within lib:
        - basic layers/blocks, such as Conv, we use flag to control the op dimentions -> Conv1d, Conv2d, Conv3d share the same conv api!
        - basic utilization/visulization tools, such as progress bar, logging ...etc.
        - basic dataset apis, such as tensorflow_database as default.
    - APIs won't be shared within lib:
        - runner and model, such as classification task, we use:
            - pyramidnet as model.
            - mixup, lookahead, adamw in runner.

### Code WiKi

- run_formatter.sh: run yapf on all files
- exp: run expiriments here
- tf_rlib: 
    - __init__.py: manager for absl.flags and absl.app
    - blocks: apis for common combinations, based on tf_rlib.layers
        - block.py: a template to manage API in blocks.
        - basic.py: conv+bn+act
        - residual.py: ResBlock and ResBottleneck
    - layers: apis for dimentions-agnostic single operation
        - conv.py: conv for any dim
        - bn.py: bn for any dim
        - act.py: actication for conv only
        - pooling.py: pooling in conv and globalpooling apis.
    - models: apis for best practice network structure for each task.
        - model.py: a template to manage API in models.
        - pyramidnet.py: for classification
    - runners: apis for best practice runner for each task.
        - classification.py: TODO
    - utils:
        - ipython.py: apis related to notebook.
  
