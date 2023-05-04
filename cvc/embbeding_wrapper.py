import os
import anndata as ad
import scanpy as sc
from cvc import model_utils
from cvc import plot_utils
import importlib
importlib.reload(plot_utils)

SAVEFIG_DPI = 1200

class EmbeddingWrapper:
    def __init__(self, model, device, sequences_df, batch_size=1024, pbar=True, max_len=64, embeddings=None, **embedding_kwargs):
        self.model = model
        self.device = device
        self.sequences_df = sequences_df
        sequences = sequences_df.Sequences.tolist()
        # if embeddings are not provided, create them
        if embeddings is None:
            self.embeddings = model_utils.get_transformer_embeddings(
                self.model,
                sequences,
                batch_size=batch_size,
                device=self.device,
                pbar=pbar,
                max_len=max_len,
                **embedding_kwargs
            )

    # create adata object of the embedding
    def create_anndata(self, n_comps=50):
        anndata = ad.AnnData(self.embeddings, obs=self.sequences_df)
        sc.pp.pca(anndata, n_comps=n_comps)
        sc.pp.neighbors(anndata)
        sc.tl.umap(anndata)
        return anndata

    # plot embedding
    def plot_embedding(self, color_embed, color_map, title=None, legend_size=7, plot_pdf_path=None, anndata=None, n_comps=50, fname=None):
        anndata = self.create_anndata(n_comps) if anndata is None else anndata
        fig = plot_utils.plot_anndata_rep(
            anndata,
            color=color_embed,
            direct_label=False,
            cmap=color_map,
            title=title,
            legend_size=legend_size,
            fname=plot_pdf_path,
        )
        if fname is not None:
            fig.savefig(fname, bbox_inches="tight", dpi=SAVEFIG_DPI)
        fig.show()
