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
            self.embeddings: np.ndarray = model_utils.get_transformer_embeddings(
                self.model,
                sequences,
                batch_size=batch_size,
                device=self.device,
                pbar=pbar,
                max_len=max_len,
                **embedding_kwargs
            )
        else:
            self.embeddings = embeddings

    # get embeddings
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # re-load model here

    # create adata object of the embedding
    def create_anndata(self, n_comps=50):
        anndata = ad.AnnData(self.embeddings, obs=self.sequences_df)
        sc.pp.pca(anndata, n_comps=n_comps)
        sc.pp.neighbors(anndata)
        sc.tl.umap(anndata)
        return anndata

    # plot embedding
    def plot_embedding(self, color_embed, color_map, title=None, legend_size=7, n_comps=50, plot_pdf_path=None, anndata=None, fname=None):
        anndata = self.create_anndata(n_comps) if anndata is None else anndata
        fig = plot_utils.plot_anndata_rep(
            anndata,
            color=color_embed,
            direct_label=False,
            cmap=color_map,
            title=title,
            legend_size=legend_size,
            fname=fname,
        )

        # use title connected with underscore as file name
        fname = fname if fname is not None else title.replace(" ", "_") + ".svg"
        print("Saving figure to {}".format(fname))
        if fname is not None:
            fig.savefig(fname, bbox_inches="tight", dpi=SAVEFIG_DPI)
        fig.show()
        plt.show()


class GenericModelEmbeddings:
    def __init__(self, model_name=None, tokenizer_class=None, model_class=None, sequences_df=None, embeddings=None, device="cuda:1"):
        self.tokenizer = tokenizer_class.from_pretrained(model_name) if tokenizer_class and model_name else None
        self.model = model_class.from_pretrained(model_name).to(device) if model_class and model_name else None
        self.device = device
        self.sequences_df = sequences_df
        self.embeddings= embeddings

    def get_embeddings(self, sequences, batch_size=1024):
        all_embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size)):
            batched_sequences = sequences[i:i + batch_size]
            inputs = self.tokenizer(batched_sequences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                all_embeddings.append(last_hidden_state[:, 0, :].cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def plot_embeddings(self, embeddings, title, label, fig_name):
        if label == "Private_Public_label":
            color_map = ListedColormap(['gold', 'darkblue'])
        else:
            color_map = ListedColormap(sns.color_palette("Spectral", 13))
        embed_wrap = EmbeddingWrapper(None, self.device, self.sequences_df, pbar=True, embeddings=embeddings)
        embed_wrap.sequences_df.sort_values(by=label, inplace=True, ascending=False)
        embed_wrap.plot_embedding(
            color_embed=label,
            color_map=color_map,
            title=title,
            legend_size=5,
            n_comps=50,
            fname=fig_name,
        )

