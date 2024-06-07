# Sagemaker llava training

# Prerequisites

- wandb account
- Jupyter Notebook (for testing)


# 实验环境准备
在Amazon SageMaker上开启一个notebook环境

# 准备数据
Run [preprocess.py](/sagemaker/data/preprocess.py)
```shell
!python data/preprocess.py \
--data_path {yourexcel} \
--output_folder {outputjson}
```

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
wandb.sagemaker_auth(path="src")
```

# Create sagemaker training job for training model

Run [sagemaker-a100-kl-v2.ipynb](/sagemaker/sagemaker-a100-kl-v2.ipynb)


# inference

## local inference

* 首先创建对应的推理环境, `cd sagemaker/src`

```shell
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

* 运行推理脚本
```shell
!python inference/local_infer_kl.py \
--test_json_file {testfile} \
--model_dir {model_dir}
```

| 模型                      |             机型 | 精度  |  成本(sagemaker) /2kw张   |      成本(ec2) /2kw张      |
|:------------------------|---------------:|:---:|:----------------------:|:-----------------------:|
| llava1.5-13b            | ml.g5.12xlarge | 84% | $7.09/h * 5334h = $38k | $5.672/h * 5334h = $30k |
| llava1.5-13b-4bit       |  ml.g5.2xlarge | 85% | $1.53/h * 5556= $8.5k  | $1.1212/h * 5556h = $6k |
| llava1.5-7b             |   ml.g5.xlarge | 82% |           NA           |           NA            |
| llava1.6-mistral 7b     |  ml.g5.4xlarge | 62% |           NA           |           NA            |
| llava1.6-vicuna13b-4bit |  ml.g5.4xlarge | 83% |           NA           |           NA            |


### inference speed up
* 4bit
```python
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
+   load_in_4bit=True
+   use_flash_attention_2=True
```

### inference: SageMaker endpoint

* inference llava1.5-13b on g5.12x, use huggingface container: Run [deploy_llava.ipynb](/sagemaker/deploy_source/deploy_llava.ipynb)
* inference llava1.5-13b-4bit on g5.2x, use djl container: Run [llava-sagemaker-endpoint.ipynb](/sagemaker/deploy_djl/llava-sagemaker-endpoint.ipynb)
