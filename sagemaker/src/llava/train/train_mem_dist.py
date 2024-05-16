from llava.train.train import train
import os
import torch
from datetime import datetime
import deepspeed

if __name__ == "__main__":
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MODEL_NAME_OR_PATH = str(os.environ['MODEL_NAME_OR_PATH'])
    PERTRAIN_PATH = str(os.environ['PERTRAIN_PATH'])
    OUTPUT_DIR = str(os.environ['OUTPUT_DIR'])

    # torch.cuda.set_device(LOCAL_RANK)
    deepspeed.init_distributed(dist_backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)
    ############################
    '''
    if 0 == LOCAL_RANK:
        print("*****************start cp pretrain model*****************************")
        os.system("chmod +x ./s5cmd")
        os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH_LLM'], MODEL_NAME_OR_PATH))
        os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH_PERJ'], PERTRAIN_PATH))
        print(f'------rank {LOCAL_RANK} finished cp-------')

    '''

    torch.distributed.barrier()
    ############################
    train(attn_implementation="flash_attention_2")
    ############################
    if WORLD_RANK == 0:
        persistant_path = os.environ['OUTPUT_MODEL_S3_PATH'] + str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")) + '/'
        os.system("./s5cmd sync {0} {1}".format(OUTPUT_DIR, persistant_path))

    torch.distributed.barrier()
    ############################
