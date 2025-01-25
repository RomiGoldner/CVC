import os
import sys
import argparse
import pandas as pd
from cvc.embbeding_wrapper import EmbeddingWrapper
from lab_notebooks.utils import TRANSFORMER, DEVICE

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for a dataset.")
    parser.add_argument("--input_path", required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_path", default=None, type=str, help="Path to save the output embeddings CSV file. Default is <input_path>/<input_file>_embeddings.csv.")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for embedding creation. Default is 1024.")
    parser.add_argument("--model_path", default=TRANSFORMER, type=str, help="Path to the model. Default is TRANSFORMER.")
    parser.add_argument("--device", default=DEVICE, type=str, help="Device to use for computation (cpu or cuda). Default is DEVICE.")

    args = parser.parse_args()

    # Set up paths
    input_path = args.input_path
    if not input_path.endswith(".csv"):
        sys.exit("Error: input_path must end with .csv")
    output_path = args.output_path or os.path.join(os.path.dirname(input_path), os.path.basename(input_path).replace(".csv", "_embeddings.csv"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    batch_size = args.batch_size
    model_path = args.model_path
    device = args.device

    # Load data
    try:
        tcrb_data = pd.read_csv(input_path)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {input_path}")
    except Exception as e:
        sys.exit(f"Error loading input file: {str(e)}")

    # Create embeddings
    try:
        embed_wrap = EmbeddingWrapper(model_path, device, tcrb_data, batch_size=batch_size, method="mean", layers=[-1])
        tcrb_embeddings_df = pd.DataFrame(embed_wrap.embeddings)
        # Save embeddings to CSV
        tcrb_embeddings_df.to_csv(output_path, index=False)
        print(f"Embeddings saved successfully to {output_path}")
    except Exception as e:
        sys.exit(f"Error during embedding creation: {str(e)}")

if __name__ == "__main__":
    main()
