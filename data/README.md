**raw** - Unprocessed raw images - Place all raw dataset image files here

**preprocessed** - Directory for RGB black and white images ready ingest into the model - Run raw_prep.py to populate this dir with preprocessed data ready to train

**processed** - Directory for augmented files from the model - When running base.py the preprocessed images will get processed and stored here for training

**spot_dmg** - Damage template files - All augment damage template files are stored here and called upon when creating processed images for the model to use to train

**generated_images** - Images saved from the model generator to pre-train the Critic

**test_imgs** - Folder for files to inference - Place any files you wish to test inference on in this folder