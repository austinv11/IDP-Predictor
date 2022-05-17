# IDP-Predictor

Tool to predict intrinsically disordered proteins.

# Requirements
- `pip install requirements.txt`
- For data generation: NCBI-BLAST+
- For making test/training sets: CD-HIT

# Files
- `hopfield_network.py`: Hopfield network architecture implementation.
- `idp_dataset.py`: A script to generate embeddings for the dataset and load it into pytorch, along with all the associated augmentations.
- `aaindex_feature_imporance.ipynb`: A jupyter notebook where I experiment in order to get the most important biochemical features.
- `basic_aa_features.csv`: A csv file with the selected AAindex biochemical features for each AA.
- `protein_feature_extractor.py`: A python script to extract AAindex, evolutionary, and secondary structure features.
- `make_training_set.py`: A python script to generate the training/testing set from disprot.

# Viewing Parquet Files
I wrote a script that converts parquet files to csv files:

`python3 ./parquet_to_csv.py path/to/parquet/file.parquet`
