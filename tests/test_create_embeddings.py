import os
import sys
import pandas as pd
from cvc.embbeding_wrapper import EmbeddingWrapper
from lab_notebooks.utils import TRANSFORMER, DEVICE

# Set up directories and paths
DATA_DIR = "data/custom_dataset"
input_path = os.path.join(DATA_DIR, "test_tcrs.csv")
output_path = os.path.join(DATA_DIR, "test_tcrs_embeddings.csv")

# Load data
tcrb_data = pd.read_csv(input_path)

# Create embeddings
try:
    embed_wrap = EmbeddingWrapper(TRANSFORMER, DEVICE, tcrb_data, batch_size=1024, method="mean", layers=[-1])
    tcrb_embeddings_df = pd.DataFrame(embed_wrap.embeddings)
    # Save embeddings to CSV
    tcrb_embeddings_df.to_csv(output_path)
    print(f"Embeddings saved successfully to {output_path}")
except Exception as e:
    sys.exit(f"Error during embedding creation: {str(e)}")

