# sbert
Repo for sbert implementation for NLI with siamese BERT transformer architecture (see below).

![grafik](https://user-images.githubusercontent.com/9453363/141699750-f6ce924f-cad9-4742-b6bb-5de870468792.png)
## Prerequisites

### Conda environment

Install conda environment named "sbert" from yml ([more information in conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

```
$ conda env create -f environment.yml
```

Activate environment

```
$ conda activate sbert
```

### Dataset (raw data)

Download and unpack the raw data from:

```
https://sbert.net/datasets/AllNLI.tsv.gz
```

### Settings
Adjust settings for your preference.
```
{
    "general": {
        "job_id" : "Test_Run",
        "train_device": "gpu"
    },
    "network": {
        "architecture": "bert-base-cased",
        "tokenizer_name": "bert-base-cased",
        "pretrained": true,
        "epochs" : 2,
        "batch_size" : 16, 
        "num_target_classes": 3,
        "batch_norm": true,
        "learning_rate": 2e-5,
        "warmup_percent": 0.1,
        "sent_embedding_dim": 768
    },  
    "data" : {
        "base_dir": "C:/Users/fsc/Documents/Privat/sbert/",
        "training_data_path": "C:/Users/fsc/Documents/Privat/sbert/data_raw/train_subset.tsv"
    },
    "data_augmentation" : {
        
    },
    "prediction" : {
        "path_model_pt" : "best_model_weights",
        "path_data" : "C:/Users/fsc/Documents/Privat/sbert/data_raw/test_subset.tsv",
        "do_classification": true
    }   
}
```

### Training
Begin training by
```{python}
python train.py
```

### Prediction
Predict class label of two sentences with
```{python}
python prediction.py --sent1 "Children smiling and waving at camera" --sent2 "There are children present"
```
or predict a whole dataset with
```{python}
python prediction_from_loader.py
```
You can also retrieve the embeddings by setting *"do_classification": false* in settings.json
