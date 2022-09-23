# ML Gapfill

# Installation instructions
1. Create new conda environment\
    `conda create --name mlfill python=3.8` 

2. Activate environment\
    `conda activate mlfill`

3. Install requirements 
    `pip install -r requirements.txt`

4. Run tests
    `pytest src/tests/* --vv`

# Data preparation
Two scripts need to be run to download and prepare data. These scripts include downloading of reference genomes, constructing raw GSMMs, extracting media and metabolic features. Splitting of training and test data

**Preprare melanie data**
In the manuscript we train the models on melanies dataset.
```
chmod +x ./preprocessing_melanie_data.sh
./preprocessing_melanie_data
```

**Preprare komodo data**

```
chmod +x ./preprocessing_komodo_data.sh
./preprocessing_komodo_data.sh
```

# Training, autogapfilling and GSMM testing
We have different entry scripts for each run mode. Each run mode also has it's on config files. See `./configs/`

Configuration of each run mode is defined using hydra configs (https://hydra.cc). I strongly suggest taking at look at some tutorials on this first.

All runs are recorded with mlflow: `https://mlflow.org/docs/latest/index.html`. Run `mlflow ui` to see runs, their metrics, visualisations.

As a minimum, please set the working directories required for the following configs:
```

```

**Model training**

`python -u train_mlgapfill.py` - Trains a model on given data. See `configs/train_configs/` for different possible settings

`./scripts/model_comparison.sh` - Performs hyperparameter optimisation of different model types by grid search. Uses hydra config overrides to perform grid search.

run `mlflow ui` to see the results

**Autogapfill models**

`python -u run_automated_gapfilling.py` - Performs automated gapfilling on GSMMs using a given classification model (produced by `train_mlgapfill.py`). See `/configs/autogapfill_config/config.yaml`.

`classifier_run_id` refers to a finished classifier training run, used to predict viability and inform gapfillings.

**Testing FBA on dataset**

`python -u run_fba_test.py.py` - Uses GSMM as a classifier to predict viability of an organism for a given media. Evaluates model performance based on given observed data. These could be either raw GSMMs or autogapfilled GSMMs

# TODO
Write tests to check carveme and prodigal installations.