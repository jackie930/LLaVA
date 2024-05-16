# Sagemaker llava training

# Prerequisites

- wandb account
- Jupyter Notebook (for testing)


# 实验环境准备
在Amazon SageMaker上开启一个notebook环境

# 准备数据
[todo]

将数据准备为如下格式
```plain
data_root
+-- imgs/
 -- data.json
```

# 准备训练使用的wandb账号

使用`pip install wandb`, 运行如下的命令, 会自动生成一个`secrets.env`文件, 将这个文件放在 `src`目录下, 这样训练时日志将自动同步到wandb的控制台
```python
import wandb

wandb.init()
wandb.login()
wandb.sagemaker_auth(path="scripts")
```

# Create sagemaker pipeline for training model

Run [sagemaker-a10.ipynb](/sagemaker-a10.ipynb)


