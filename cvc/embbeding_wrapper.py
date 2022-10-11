import os
import anndata as ad
import scanpy as sc
from cvc import model_utils
from cvc import plot_utils
import importlib
importlib.reload(plot_utils)


# create class embedding wrapper
class EmbeddingWrapper:
    def __init__(self, model, device, sequences_df, batch_size=1024, pbar=True, **embedding_kwargs):
        self.model = model
        self.device = device
        self.sequences_df = sequences_df
        sequences = sequences_df.Sequences.tolist()
        self.embeddings = model_utils.get_transformer_embeddings(
            self.model,
            sequences,
            batch_size=batch_size,
            device=self.device,
            pbar=pbar,
            **embedding_kwargs
        )

    # create adata object of the embedding
    def create_anndata(self):
        anndata = ad.AnnData(self.embeddings, obs=self.sequences_df)
        sc.pp.pca(anndata, n_comps=50)
        sc.pp.neighbors(anndata)
        sc.tl.umap(anndata)
        return anndata

    # plot embedding
    def plot_embedding(self, color_embed, color_map, title=None, legend_size=7, plot_pdf_path=None, anndata=None):
        anndata = self.create_anndata() if anndata is None else anndata
        plot_utils.plot_anndata_rep(
            anndata,
            color=color_embed,
            direct_label=False,
            cmap=color_map,
            title=title,
            legend_size=legend_size,
            fname=plot_pdf_path,
        ).show()
