# AI for blemish detection and repair

ML modeling for cleaning old photos without masking. This was made to train to clean mold and water damage from old photographic scans. It utilizes a custom augmentation tool to generate synthetic data to train on.

[Here](https://docs.google.com/presentation/d/1RmXTLyjV0TNKgsKn4sWcl1HKKZCy7QQWfIEMb5M4ots/edit?usp=sharing) are the slides for project.

## Project:

- **src** : All source code for production within structured directory

- **data** :  Contains the following small sample data to train and validate model


## Environment Installation

The following setup instructions is for if you want to clone the repo to run locally:

```
conda create --name myenv --file evironment.txt
conda activate myenv
```

To get the streamlit to run, it will require a pip install:

```
pip install streamlit
```

  

## Run Inference

- Utilize the inf file in src to run inference on a single image:
- Find the commented out section to input your model weights, and input them there
- Place images into designated folders and run inference on your image!
- Standard inference time is 2 seconds for GPU and 7 seconds on CPU
