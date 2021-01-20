# JointBERT

This repository will contain the code and data download links to reproduce the experiments of the paper *Dual-Objective Fine-Tuning of BERT for Entity Matching* by Ralph Peeters and Christian Bizer

\*Currently under construction\*

* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

    Please keep in mind that the code is not optimized for portable or even non-workstation devices. Some of the scripts require large amounts of RAM (64GB+) and GPUs. It is advised to use a powerful workstation or server when experimenting with some of the large files.

    The code has only been used and tested on Linux (Manjaro, Ubuntu, CentOS) computers.

* **Building the conda environment**

    To build the exact conda environment used for the experiments, navigate to the project root folder where the file *jointbert.yml* is located and run ```conda env create -f jointbert.yml```
    
    Furthermore you need to install the project as a package. To do this, activate the jointbert environment with ```conda activate jointbert```, navigate to the root folder of the project, and run ```pip install -e .```

* **Downloading the raw data files/models**

    Navigate to the *src/data/* folder and run ```python download_datasets.py``` to automatically download the files into the correct locations.
    You can find the data at *data/raw/* and the trained models used for generating explanations at *src/productbert/saved/models*

* **Processing the data**

    To prepare the data for the experiments, run the following scripts in that order. Make sure to navigate to the respective folders first.
    
    1. *src/processing/preprocess/preprocess_corpus.py*
    2. *src/processing/preprocess/preprocess_ts_gs.py*
    3. *src/processing/process-bert/process_to_bert.py*
    4. *src/processing/process-magellan/process_to_magellan.py*
    5. *src/processing/process-wordcooc/process-to-wordcooc.py*
    
    optional: prepar data for Ditto input: *src/processing/process-ditto/process_to_ditto.py*
    
* **Running the baseline experiments**

    Run the following scripts to replicate the baseline experiments:
    * **Magellan**:
        Navigate to *src/models/magellan/* and run the script *run_magellan.py*
    * **Word Coocurrence**:
    Navigate to *src/models/wordcooc/* and run the script *run_wordcooc.py*
    * **Deepmatcher**:
    Navigate to *src/models/deepmatcher* and run any of the scripts *train_\*.py*.
    
    Result files can subsequently be found in the *reports* folder.

* **Running the BERT experiments**

    Navigate to *src/productbert/*
    This project is based on a <a target="_blank" href="https://github.com/victoresque/pytorch-template/">PyTorch template project</a> It is suggested to read the respective github readme to understand how to train models and possible input commands.
    * **Fine-Tuning**:
    The folder *src/productbert* contains bash scripts to run all of the experiments including the learning rate sweeps. Run any of the bash scripts titled *train_\*.sh* and append the id of the gpu you want to use, e.g. ```bash train_computers_small.sh 0```

    * **Evaluating a trained model on a test set**:
    This is done by providing a config containing configuration parameters for the test run. Additionally you need to provide the checkpoint that should be used for testing. An example would be ```python test.py --device 0 -c configs/BERT/config_computers_small_test.json --resume saved/models/pathto/model/model_best.pth```
    
    The results of the BERT experiments can then be found in *src/productbert/saved/log* and the respective model checkpoints in *src/productbert/saved/models*.
    
    **NOTE**: When adjusting the batch size in any of the configs, make sure to also adjust the number of accumulation steps, as the combination of both constitutes the actual batch size.

* **Explanations**

    Jupyter Notebooks used to generate explanations and corresponding swarmplots can be found at *src/productbert*. The swarmplot images can be found at *data/processed/explain_labeling/pictures*.

* **Ditto**

    The code for the Ditto framework can be found [here](https://github.com/megagonlabs/ditto)
--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience

PyTorch Project based on the [PyTorch template project](https://github.com/victoresque/pytorch-template/) by [Victor Huang](https://github.com/victoresque).