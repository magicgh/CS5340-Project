# CS5340 Project

### download from huggingface

```cmd
 huggingface-cli download --token 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 --resume-download MODEL_NAME_ON_HUGGINGFACE --local-dir /root/autodl-tmp/models_to_load/MODEL_NAME
```

## UQ_ICL

### Dependencies

This code is written in Python. To use it you will need:

- Numpy
- Scipy
- pandas
- Transformers
- PyTorch
- datasets

### Usage

#### Using Hugging Face Model

```cmd
huggingface-cli login --token "YOUR_HUGGING_FACE_TOKEN"
```

Remember to register the use of intended model ahead.

#### Data

The data can be downloaded from the file by datasets Python library.

#### Test Models

There are five datasets, you can test the results of different datasets with using the executable files (_cola.sh, ag_news.sh, financial.sh, ssh.sh, sentiment.sh_) provided.

Note that the parameter value ranges are hyper-parameters, and different range
may result different performance in different dataset, be sure to tune hyper-parameters carefully.

## Instruction Tuning

Dataset: [LIMA](https://arxiv.org/abs/2305.11206)  
Tutorial: [X-Accessory](https://llama2-accessory.readthedocs.io/en/latest/)