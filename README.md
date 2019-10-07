# tf_rlib

### Ideas 

- goal
    - This libs is target for fastest AI development within tf2, we only implement the most general algorithms here. Therefore, the flexibility is limited. All we need is manipulate the flags, then we could achieve the SOTA performance on most of common tasks.
- coding ideas
    - based on the facts that DL tasks are diferent, if no research-orientedit's not necessary to use single api solve all the problems, for examples:
        - about loss: we won't use dice loss for classification.
        - about data: we won't use labeled data in self-supervised/unsupervised training.
        - about structure: we won't combine resblock, resnextblock, preactblock, pyramidnet into a new one.
    - we use absl.flags to manage all hyper-perameters between all modules.
    - this lib should be runnable on both interactive mode (jupyter) and cmd mode.
    - we use tensorboard and tqdm to visulize the training progress.

### Code WiKi

- run_formatter.sh: run yapf on all files
- exp: run expiriments here
- tf_rlib: 
    - __init__.py: manager for absl.flags and absl.app
    - blocks: apis for common combinations, based on tf_rlib.layers
        - basic.py: conv+bn+act
    - layers: apis for dimentions-agnostic single operation
        - conv.py: conv for any dim
        - bn.py: bn for any dim
    - models: apis for best practice network structure for each task.
        - classification.py: TODO
    - runners: apis for best practice runner for each task.
        - TODO
    - utils:
        - ipython.py: apis related to notebook.
  
