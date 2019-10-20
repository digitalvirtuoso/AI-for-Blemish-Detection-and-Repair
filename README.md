# AI for blemish detection and repair

ML modeling for cleaning old photos without masking. This was made to train to clean mold and water damage from old photographic scans. It utilizes a custom augmentation tool to generate synthetic data to train on.

[Here](https://docs.google.com/presentation/d/1RmXTLyjV0TNKgsKn4sWcl1HKKZCy7QQWfIEMb5M4ots/edit?usp=sharing) are the slides for project.

## Project:

- **src** : All source code for production within structured directory

- **data** :  Contains a small sample set of data from [Flickr Faces HQ](https://github.com/NVlabs/ffhq-dataset) to train and validate model

*Note: To train and validate model on sample set data, batchsize needs to change from 64 to 5 in base.py*

## Environment Installation

The following setup instructions is for if you want to clone the repo to run locally:

Set up conda environment with environment.txt:
```
conda create --name myenv --file environment.txt
conda activate myenv
```

To get streamlit to run, it will require a pip install:
```
pip install streamlit
```

## Model Training
- Place unprocessed data into the raw folder and run raw_prep.py to preprocess the data
- Once the data is preprocessed train the model by running base.py

*Note: Remove all sample data when using a new dataset*

## Run Inference

- Utilize inf.py in src to run inference on images:
- Find the commented out section to input model weights
- Place images into designated folders and run inference on the images!
- Standard inference time is 2 seconds for GPU and 7 seconds on CPU

