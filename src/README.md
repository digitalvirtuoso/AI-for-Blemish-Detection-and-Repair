base.py - Calls the Model to train

model_utils - Defs needed by the base, inf, and sl_app files

wmtransformer - Class for Augmenting Images

sl_app - Base version of streamlit app file
    To run the app with your files, weights, and templates follow the commented sections of the file

inf - File for calling inference on a saved model
    To run inference on a trained model version, find the commented section and put your model name there
        learn_gen = create_gen_learner(data_gen).load('NEWMODELHERE') # Input model to load
        # Replace NEWMODELHERE with your saved model name to load your model