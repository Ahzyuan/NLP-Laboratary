import os
import torch.distributed as dist

local_rank = os.environ['LOCAL_RANK']
local_ws = os.environ['LOCAL_WORLD_SIZE']

# SPMD-INFO see https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md
# These are the parameters used to initialize the process group
env_dict = {
    key: os.environ[key] for key in ("RANK", 'LOCAL_RANK',"WORLD_SIZE","MASTER_ADDR", "MASTER_PORT")
}
print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

dist.init_process_group(backend="nccl")

print(
    f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
    + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
)

train_script(*training_args, **kargs)

# Tear down the process group
dist.destroy_process_group()
