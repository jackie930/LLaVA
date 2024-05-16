import os


if __name__ == "__main__":
    os.system("pip install --upgrade pip")
    os.system("pip install -e .")
    os.system("pip install deepspeed==0.12.6")
    os.system("pip install ninja")

    #os.system("pip install transformers==4.31.0")
    os.system("pip install torch==2.0.1")
    os.system("pip install flash-attn==2.5.8 --no-build-isolation")
    os.system("pip uninstall -y apex")
    os.system("pip install wandb")

    os.system("chmod +x ./finetune_single.sh")
    os.system("chmod +x ./s5cmd")
    os.system("/bin/bash -c ./finetune_single.sh")

