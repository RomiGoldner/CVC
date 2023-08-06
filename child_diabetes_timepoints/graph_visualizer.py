"""Module with embedding visualization tools."""
import functools
import inspect
import itertools
import math
import sys
import warnings
from collections import Counter
from multiprocessing import cpu_count
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ensmallen import Graph  # pylint: disable=no-name-in-module
from ensmallen.datasets.graph_retrieval import normalize_node_name
from environments_utils import is_colab, is_notebook
from humanize import apnumber, intword
from matplotlib import collections as mc
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from userinput.utils import must_be_in_set

from embiggen.embedders import embed_graph
from embiggen.embedding_transformers import GraphTransformer, NodeTransformer
from embiggen.utils.abstract_edge_feature import AbstractEdgeFeature
from embiggen.utils.abstract_models import format_list
from embiggen.utils.abstract_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)
from embiggen.utils.abstract_models.embedding_result import EmbeddingResult
from embiggen.utils.abstract_models.abstract_classifier_model import (
    AbstractClassifierModel,
)
from embiggen.utils.pipeline import iterate_graphs


class Annotation3D(Annotation):
    """Annotate the point xyz with text s"""

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


try:
    from ddd_subplots import display_video_at_path, rotate
    from ddd_subplots import subplots as subplots_3d
except ImportError:
    warnings.warn(
        "We were not able to detect the CV2 package and libGL.so, therefore "
        "you will not be able to execute 3D animations with the visualization "
        "pipeline."
    )


class GraphVisualizer:
    """Tools to visualize the graph embeddings."""

    DEFAULT_SCATTER_KWARGS = dict(s=5, alpha=0.7)
    DEFAULT_EDGES_SCATTER_KWARGS = dict(alpha=0.5)
    DEFAULT_SUBPLOT_KWARGS = dict(figsize=(7, 7), dpi=200)

    def __init__(
        self,
        graph: Union[Graph, str],
        support: Optional[Graph] = None,
        subgraph_of_interest: Optional[Graph] = None,
        repository: Optional[str] = None,
        version: Optional[str] = None,
        decomposition_method: str = "TSNE",
        n_components: int = 2,
        rotate: bool = False,
        video_format: str = "webm",
        duration: int = 10,
        fps: int = 24,
        node_embedding_method_name: str = "auto",
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
        minimum_node_degree: int = 0,
        maximum_node_degree: Optional[int] = None,
        only_from_same_component: Union[bool, str] = "auto",
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        source_node_types_names: Optional[Union[str, List[str]]] = None,
        destination_node_types_names: Optional[Union[str, List[str]]] = None,
        source_edge_types_names: Optional[Union[str, List[str]]] = None,
        destination_edge_types_names: Optional[Union[str, List[str]]] = None,
        source_nodes_prefixes: Optional[Union[str, List[str]]] = None,
        destination_nodes_prefixes: Optional[Union[str, List[str]]] = None,
        edge_type_names: Optional[List[Optional[str]]] = None,
        show_graph_name: Union[str, bool] = "auto",
        number_of_columns_in_legend: int = 2,
        show_embedding_method: bool = True,
        show_edge_embedding_methods: bool = True,
        show_separability_considerations_explanation: bool = True,
        show_heatmaps_description: bool = True,
        show_non_existing_edges_sampling_description: bool = True,
        automatically_display_on_notebooks: bool = True,
        number_of_subsampled_nodes: int = 20_000,
        number_of_subsampled_edges: int = 10_000,
        number_of_subsampled_negative_edges: int = 10_000,
        number_of_holdouts_for_cluster_comments: int = 5,
        random_state: int = 42,
        decomposition_kwargs: Optional[Dict] = None,
        verbose: bool = False,
    ):
        """Create new GraphVisualizer object.

        Parameters
        --------------------------
        graph: Union[Graph, str]
            The graph to visualize.
            If a string was provided, we try to retrieve the given
            graph name using the Ensmallen automatic graph retrieval.
        support: Optional[Graph] = None
            The support graph to use to compute metrics such as degrees,
            Adamic-Adar and so on. This is useful when the graph to be
            visualized is a small component of a larger graph, for instance
            when visualizing the test graph of an holdout where most edges
            are left in the training graph.
            Without providing a support graph, it would not be possible
            to visualize the correct metrics of the provided graph as we
            would have to assume that this graph is a `true` graph by itself,
            and not a subgraph of a larger one.
            The support graph is also employed in the sampling of the negative
            edges, so to avoid sampling edges that exist in the original graph,
            and is also used in the computation of the embedding when requested.
            This provided graph must share the same vocabulary of the support graph.
            When visualizing an holdout, the train graph should be used as support
            for the validation or test graphs.
        subgraph_of_interest: Optional[Graph] = None
            The graph to use to sample the negative edges.
            This graph's node degree distribution is expected to be the one best capturing
            the are of interest of a task (for instance a subgraph edge prediction)
            and is therefore best suited to sample the negative edges.
            If we were to use the support graph for cases of this type we would
            sample negative edges that are far away from the positive edges we are
            taking into consideration, making the visualization biased and make the
            task look artificially easy.
            We expected for the provided graph to be contained in this subgraph of interest.
        repository: Optional[str] = None
            Repository of the provided graph.
            This only applies when the provided graph is a
            graph name that can be retrieved using ensmallen.
            Providing this parameter with an actual graph
            object will cause an exception to be raised.
        version: Optional[str] = None
            version of the provided graph.
            This only applies when the provided graph is a
            graph name that can be retrieved using ensmallen.
            Providing this parameter with an actual graph
            object will cause an exception to be raised.
        decomposition_method: str = "TSNE",
            The decomposition method to use.
            The supported methods are UMAP, TSNE and PCA.
        n_components: int = 2,
            Number of components to reduce the image to.
            Currently we support 2D, 3D and 4D visualizations.
        rotate: bool = False,
            Whether to create a rotating animation.
        video_format: str = "webm"
            What video format to use for the animations.
        duration: int = 15,
            Duration of the animation in seconds.
        fps: int = 24,
            Number of frames per second in animations.
        node_embedding_method_name: str = "auto",
            Name of the node embedding method used.
            If "auto" is used, then we try to infer the type of
            node embedding algorithm used, which in some cases is
            recognizable automatically.
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
            Edge embedding method.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        only_from_same_component: Union[bool, str] = "auto"
            Whether to sample negative edges only from the same connected component.
            This should generally be set to `True`, but in some corner cases when
            the graph to visualize has extremely dense (or very small, like tuples)
            components it will raise an exception as it is not possible to sample
            negative edges from such densely connected components.
            By default, it is set to `auto`, which will set it to `True` if the
            graph is smaller than 50M nodes, as computing and masking the connected
            components can get expensive on very large graphs, such as WikiData.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges only with source and
            destination nodes that have different node types.
            Using this parameter will raise an exception when the provided
            graph wither does not have node types or has exclusively constant
            node types.
        minimum_node_degree: Optional[int] = 0
            The minimum node degree of either the source or
            destination node to be sampled.
        maximum_node_degree: Optional[int] = None
            The maximum node degree of either the source or
            destination node to be sampled.
            By default, the number of nodes.
        source_node_types_names: Optional[str, List[str]] = None
            Node type names of the nodes to be samples as sources.
            If a node has any of the provided node types,
            it can be sampled as a source node.
        destination_node_types_names: Optional[str, List[str]] = None
            Node type names of the nodes to be samples as destinations.
            If a node has any of the provided node types,
            it can be sampled as a destination node.
        source_edge_types_names: Optional[str, List[str]] = None
            Edge type names of the nodes to be samples as sources.
            If a node has any of the provided edge types,
            it can be sampled as a source node.
        destination_edge_types_names: Optional[str, List[str]] = None
            Edge type names of the nodes to be samples as destinations.
            If a node has any of the provided edge types,
            it can be sampled as a destination node.
        source_nodes_prefixes: Optional[str, List[str]] = None
            Prefixes of the nodes names to be samples as sources.
            If a node starts with any of the provided prefixes,
            it can be sampled as a source node.
        destination_nodes_prefixes: Optional[str, List[str]] = None
            Prefixes of the nodes names to be samples as destinations.
            If a node starts with any of the provided prefixes,
            it can be sampled as a destinations node.
        edge_type_names: Optional[List[Optional[str]]] = None
            Edge type names of the edges to show in the positive graph.
        show_graph_name: Union[str, bool] = "auto"
            Whether to show the graph name in the plots.
            By default, it is shown if the graph does not have a trivial
            name such as `Graph`.
        number_of_columns_in_legend: int = 2
            The number of columns to be used with the legend.
        show_embedding_method: bool = True
            Whether to show the node embedding method.
            By default, we show it if we can detect it.
        show_edge_embedding_methods: bool = True
            Whether to show the edge embedding method.
            By default, we show it if we can detect it.
        show_separability_considerations_explanation: bool = True
            Whether to explain how the separability considerations are obtained
            in the captions of the images.
        show_heatmaps_description: bool = True
            Whether to describe the heatmaps
            in the captions of the images.
        show_non_existing_edges_sampling_description: bool = True
            Whether to describe the modalities used to
            sample the negative edges.
        automatically_display_on_notebooks: bool = True
            Whether to automatically show the plots and the captions
            using the display command when in jupyter notebooks.
        number_of_subsampled_nodes: int = 20_000
            Number of points to subsample.
            Some graphs have a number of nodes and edges in the millions.
            Using non-CUDA versions of TSNE, the dimensionality reduction
            procedure can take a considerable amount of time.
            For this porpose, we include the possibility to subsample the
            points to the given number.
        number_of_subsampled_edges: int = 20_000
            Number of edges to subsample.
            The same considerations described for the subsampled nodes number
            also apply for the edges number.
            Not subsampling the edges in most graphs is a poor life choice.
        number_of_subsampled_negative_edges: int = 20_000
            Number of edges to subsample.
            The same considerations described for the subsampled nodes number
            also apply for the edges number.
            Not subsampling the edges in most graphs is a poor life choice.
        number_of_holdouts_for_cluster_comments: int = 5
            Number of holdouts to execute for getting the comments
            about clusters separability.
        random_state: int = 42
            The random state to reproduce the visualizations.
        decomposition_kwargs: Optional[Dict] = None
            Kwargs to forward to the selected decomposition method.
        verbose: bool = False
            Whether to show loading bars and logs.

        Raises
        ---------------------------
        ValueError,
            If the target decomposition size is not supported.
        ModuleNotFoundError,
            If TSNE decomposition has been required and no module supporting
            it is installed.
        """
        graph = next(
            iterate_graphs(graphs=graph, repositories=repository, versions=version)
        )

        self._graph = graph
        if support is None:
            support = self._graph

        if subgraph_of_interest is None:
            subgraph_of_interest = self._graph

        self._support = support
        self._subgraph_of_interest = subgraph_of_interest
        self._number_of_columns_in_legend = number_of_columns_in_legend

        if isinstance(source_node_types_names, str):
            source_node_types_names = [source_node_types_names]
        if isinstance(destination_node_types_names, str):
            destination_node_types_names = [destination_node_types_names]
        if isinstance(source_edge_types_names, str):
            source_edge_types_names = [source_edge_types_names]
        if isinstance(destination_edge_types_names, str):
            destination_edge_types_names = [destination_edge_types_names]
        if isinstance(source_nodes_prefixes, str):
            source_nodes_prefixes = [source_nodes_prefixes]
        if isinstance(destination_nodes_prefixes, str):
            destination_nodes_prefixes = [destination_nodes_prefixes]
        if isinstance(edge_type_names, str):
            edge_type_names = [edge_type_names]

        edge_prediction_graph_kwargs = dict(
            minimum_node_degree=minimum_node_degree,
            maximum_node_degree=maximum_node_degree,
            source_node_types_names=source_node_types_names,
            destination_node_types_names=destination_node_types_names,
            source_edge_types_names=source_edge_types_names,
            destination_edge_types_names=destination_edge_types_names,
            source_nodes_prefixes=source_nodes_prefixes,
            destination_nodes_prefixes=destination_nodes_prefixes,
        )

        if any(
            number_of_subsamples is None or number_of_subsamples > 100_000
            for number_of_subsamples in (
                number_of_subsampled_nodes,
                number_of_subsampled_edges,
                number_of_subsampled_negative_edges,
            )
        ):
            warnings.warn(
                "One of the number of subsamples requested is either None "
                "(so no subsampling is executed) or it is higher than "
                "100k values. Note that all the available decomposition "
                "algorithms supported do not scale too well on "
                "datasets of this size and their visualization may "
                "just produce Gaussian spheres, even though the data "
                "is informative."
            )

        self._positive_graph = graph.sample_positive_graph(
            number_of_samples=min(
                number_of_subsampled_edges, graph.get_number_of_edges()
            ),
            random_state=random_state,
            edge_type_names=edge_type_names,
            support=self._support,
            **edge_prediction_graph_kwargs,
        )

        if only_from_same_component == "auto":
            only_from_same_component = graph.get_number_of_nodes() < 50_000_000

        # We sample the negative edges using the subgraph of interest as base graph
        # to follow its scale free distribution, which may be different from the
        # main graph scale free distribution when particular filters are applied to it.
        # For instance, the scale free distribution of one particular edge type
        # may be very different from the whole graph scale free distribution.
        # Furthermore, we avoid sampling false negatives by passing to the
        # method also the support graph.
        try:
            self._negative_graph = self._subgraph_of_interest.sample_negative_graph(
                number_of_negative_samples=min(
                    number_of_subsampled_negative_edges,
                    self._positive_graph.get_number_of_edges(),
                ),
                random_state=random_state,
                use_scale_free_distribution=True,
                graph_to_avoid=self._support,
                support=self._support,
                only_from_same_component=only_from_same_component,
                sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                **edge_prediction_graph_kwargs,
            )
        except ValueError as exception:
            warnings.warn(str(exception))
            self._negative_graph = None

        self._number_of_subsampled_nodes = number_of_subsampled_nodes
        self._subsampled_node_ids = None

        self._rotate = rotate
        self._graph_name = self._graph.get_name()

        if show_graph_name == "auto":
            show_graph_name = self._graph_name.lower() != "graph"

        self._show_graph_name = show_graph_name
        self._show_embedding_method = show_embedding_method
        self._show_edge_embedding_methods = show_edge_embedding_methods
        self._edge_embedding_methods = edge_embedding_methods
        self._verbose = verbose

        self._show_separability_considerations_explanation = (
            show_separability_considerations_explanation
        )
        self._show_heatmaps_description = show_heatmaps_description
        self._show_non_existing_edges_sampling_description = (
            show_non_existing_edges_sampling_description
        )
        self._automatically_display_on_notebooks = automatically_display_on_notebooks
        self._currently_plotting_edge_embedding = False

        self._number_of_holdouts_for_cluster_comments = (
            number_of_holdouts_for_cluster_comments
        )

        self._embedding_method_name = node_embedding_method_name

        self._node_decomposition = None
        self._positive_edge_decomposition = None
        self._negative_edge_decomposition = None

        self._has_autodetermined_embedding_name = False

        self._random_state = random_state
        self._video_format = video_format
        self._duration = duration
        self._fps = fps

        if decomposition_kwargs is None:
            decomposition_kwargs = {}

        self._n_components = n_components

        self._decomposition_method = must_be_in_set(
            decomposition_method, ("PCA", "TSNE", "UMAP"), "decomposition method"
        )
        self._decomposition_kwargs = decomposition_kwargs

    def iterate_subsampled_node_ids(self) -> Iterator[int]:
        """Return iterator over the node IDs of the subsampled graph."""
        if self._subsampled_node_ids is None:
            return range(self._graph.get_number_of_nodes())
        return iter(self._subsampled_node_ids)

    def get_number_of_subsampled_nodes(self) -> int:
        """Return the number of subsampled nodes."""
        return sum([1 for _ in self.iterate_subsampled_node_ids()])

    def _handle_notebook_display(
        self, *args: List, caption: Optional[str] = None
    ) -> Optional[Union[Tuple[Figure, Axes], Tuple[Figure, Axes, str]]]:
        """Handles whether to display provided data in a Jupyter Notebook or return them.

        Parameters
        ---------------------
        figure: Figure
            The figure to display.
        axes: Axes
            Axes of the figure.
        *args: List
            Capturing arbitrary additional parameters.
        caption: Optional[str] = None
            Optional caption for this figure.
        """
        # This is a visualization run for rotation.
        if len(args) < 2:
            figure = None
            axes = None
        else:
            figure, axes = args[:2]
        if (is_notebook() or is_colab()) and self._automatically_display_on_notebooks:
            from IPython.display import HTML, display

            if figure is not None:
                display(figure)
            if caption is not None:
                display(
                    HTML(
                        '<p style="text-align: justify; word-break: break-all;">{}</p>'.format(
                            caption
                        )
                    )
                )
            plt.close()
        elif caption is None or self._rotate:
            return (figure, axes, *args[2:])
        else:
            return (figure, axes, *args[2:], caption)

    def get_separability_comments_description(
        self, letters: Optional[List[str]] = None
    ) -> str:
        """Returns separability comments description for the provided letters."""
        if not self._show_separability_considerations_explanation:
            return ""

        number_of_letters = 0 if letters is None else len(letters)
        plural = "s" if number_of_letters else ""

        return (
            " The separability consideration{plural} {letters} derive from "
            "evaluating a Decision Tree trained on {holdouts_number} Monte Carlo holdouts, "
            "with a 70/30 split between training and test sets."
        ).format(
            plural=plural,
            letters="for figure{plural} {letters} ".format(
                plural=plural, letters=format_list(letters, bold_words=True)
            )
            if number_of_letters > 0
            else "",
            holdouts_number=apnumber(self._number_of_holdouts_for_cluster_comments),
        )

    def get_heatmaps_comments(self, letters: Optional[List[str]] = None) -> str:
        """Returns description of the heatmaps for the provided letters."""
        if (
            not self._show_heatmaps_description
            or letters is not None
            and len(letters) == 0
        ):
            return ""

        number_of_letters = 0 if letters is None else len(letters)
        plural = "s" if number_of_letters else ""

        return (
            " In the heatmap{plural}, {letters}"
            "low and high values appear in red and blue hues, respectively. "
            "Intermediate values appear in either a yellow or cyan hue. "
            "The values are on a logarithmic scale"
        ).format(
            plural=plural,
            letters="{}, ".format(format_list(letters, bold_words=True))
            if number_of_letters > 0
            else "",
        )

    def get_non_existing_edges_sampling_description(self) -> str:
        """Returns description on how the non-existing edges are sampled."""
        if not self._show_non_existing_edges_sampling_description:
            return ""

        caption = (" We have sampled {} existing and {} non-existing edges.").format(
            intword(self._positive_edge_decomposition.shape[0]),
            intword(self._negative_edge_decomposition.shape[0]),
        )

        if self._graph.has_disconnected_nodes():
            caption += (
                " We have sampled the non-existent edges' source "
                "and destination nodes by avoiding any disconnected "
                "nodes present in the graph to avoid biases."
            )

        return caption

    def get_decomposition_method(self) -> Callable:
        # Adding a warning for when decomposing methods that
        # embed nodes using a cosine similarity / distance approach
        # in order to avoid false negatives, that is bad TSNE decompositions
        # while the embedding is actually good.
        if (
            self._n_components < 3
            and self._decomposition_method in ("UMAP", "TSNE")
            and self._embedding_method_name
            in ("Node2Vec GloVe", "DeepWalk GloVe", "First-order LINE")
        ):
            metric = self._decomposition_kwargs.get("metric")
            if metric is not None and metric != "cosine":
                warnings.warn(
                    "Please do be advised that when using a node embedding method "
                    "such as Glove, which embeds nodes using a dot product, it is "
                    "highly suggested to use a `cosine` metric. Using a different "
                    f"metric, such as the one you have provided ({metric}) may lead "
                    "to worse decompositions using UMAP or t-SNE."
                )
            else:
                # Otherwise we switch to using a cosine metric.
                self._decomposition_kwargs["metric"] = "cosine"

        if self._decomposition_method == "UMAP":
            # The UMAP package graph is not automatically installed
            # with the Embiggen package because it has multiple possible
            # installation options that are left to the user.
            # It can be, generally speaking, installed using:
            #
            # ```bash
            # pip install umap-learn
            # ````
            from umap import UMAP

            return UMAP(
                **{
                    **dict(
                        n_components=self._n_components,
                        random_state=self._random_state,
                        transform_seed=self._random_state,
                        n_jobs=cpu_count(),
                        tqdm_kwds=dict(
                            desc="Computing UMAP", leave=False, dynamic_ncols=True
                        ),
                        verbose=self._verbose,
                    ),
                    **self._decomposition_kwargs,
                }
            ).fit_transform
        elif self._decomposition_method == "TSNE":
            from sklearn.manifold import TSNE  # pylint: disable=import-outside-toplevel

            return TSNE(
                **{
                    **dict(
                        n_components=self._n_components,
                        n_jobs=cpu_count(),
                        random_state=self._random_state,
                        verbose=self._verbose,
                        learning_rate=200,
                        n_iter=400,
                        init="random",
                        method="exact" if self._n_components == 4 else "barnes_hut",
                    ),
                    **self._decomposition_kwargs,
                }
            ).fit_transform
        elif self._decomposition_method == "PCA":
            return PCA(
                **{
                    **dict(
                        n_components=self._n_components,
                        random_state=self._random_state,
                    ),
                    **self._decomposition_kwargs,
                }
            ).fit_transform

    def _shuffle(
        self,
        *args: List[Union[np.ndarray, pd.DataFrame, None]],
    ) -> List[np.ndarray]:
        """Return given arrays shuffled synchronously.

        The reason to shuffle the points is mainly that this avoids for
        'fake' clusters to appear simply by stacking the points by class
        artifically according to how the points are sorted.

        Parameters
        ------------------------
        *args: List[Union[np.ndarray, pd.DataFrame, None]]
            The lists to shuffle.

        Returns
        ------------------------
        Shuffled data using given random state.
        """
        index = np.arange(args[0].shape[0])
        random_state = np.random.RandomState(  # pylint: disable=no-member
            seed=self._random_state
        )
        random_state.shuffle(index)
        return [
            arg[index]
            if isinstance(arg, np.ndarray)
            else arg.iloc[index]
            if isinstance(arg, pd.DataFrame)
            else None
            for arg in args
        ]

    def decompose(self, X: np.ndarray) -> np.ndarray:
        """Return requested decomposition of given array.

        Parameters
        -----------------------
        X: np.ndarray,
            The data to embed.

        Raises
        -----------------------
        ValueError,
            If the given vector has less components than the required
            decomposition target.

        Returns
        -----------------------
        The obtained decomposition.
        """
        if X.shape[1] == self._n_components:
            return X
        if X.shape[1] < self._n_components:
            raise ValueError(
                "The vector to decompose has less components than "
                "the decomposition target."
            )
        # Some embedding method have complex values.
        # Such values are, of course, not supported by UMAP, TSNE or PCA.
        # For such cases, we need to convert the complex value into a real
        # value. If the user desires to use some different approach, it can
        # be applied to the embedding before providing it to this visualization tool.
        if "complex" in str(X.dtype):
            X = np.hstack([np.real(X), np.imag(X)])
        if (
            self._decomposition_method == "TSNE"
            and X.shape[1] > 50
            and X.shape[0] > 50
        ):
            X = PCA(n_components=50, random_state=self._random_state).fit_transform(X)
        return self.get_decomposition_method()(X)

    def _normalize_label(self, labels: List[str]) -> List[str]:
        last_element = labels[-1]
        if last_element.lower().startswith("other"):
            labels = [
                label
                for label in sanitize_ml_labels(
                    [normalize_node_name(label) for label in labels[:-1]]
                )
            ]
            labels.append(last_element)
        return [
            label
            for label in sanitize_ml_labels(
                [normalize_node_name(label) for label in labels]
            )
        ]

    def _set_legend(
        self,
        axes: Axes,
        labels: List[str],
        handles: List[HandlerBase],
        loc: str = "best",
    ):
        """Set the legend with the given values and handles transparency.

        Parameters
        ----------------------------
        axes: Axes,
            The axes on which to put the legend.
        labels: List[str],
            Labels to put in the legend.
        handles: List,
            Handles to display in the legend (the curresponding matplotlib
            objects).
        loc: str = 'best'
            Position for the legend.
        """
        number_of_columns = (
            1
            if len(labels) <= 2 and any(len(label) > 20 for label in labels)
            else self._number_of_columns_in_legend
        )

        labels = [
            f"{label[:20]}..." if len(label) > 20 and number_of_columns == 2 else label
            for label in self._normalize_label(labels)
        ]
        legend = axes.legend(
            handles=handles,
            labels=labels,
            loc=loc,
            ncol=number_of_columns,
            prop={"size": 8},
            **(
                dict(handler_map={tuple: HandlerTuple(ndivide=None)})
                if len(handles) > 0 and isinstance(handles[0], tuple)
                else {}
            ),
        )

        # Setting maximum alpha to the visualization
        # to avoid transparency in the dots.
        for legend_handle in legend.legendHandles:
            legend_handle.set_alpha(1)
            try:
                legend_handle._legmarker.set_alpha(1)
            except AttributeError:
                pass

    def automatically_detect_embedding_method(
        self, node_embedding: np.ndarray
    ) -> Optional[str]:
        """Detect node embedding method using heuristics, where possible."""
        # Rules to detect TFIDF/BERT embedding
        if node_embedding.dtype == "float16" and node_embedding.shape[1] == 768:
            return "TFIDF-weighted BERT"
        return self._embedding_method_name

    def fit_nodes(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
    ):
        """Executes fitting for plotting node embeddings.

        Parameters
        -------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        """
        node_features = AbstractClassifierModel.normalize_node_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_features=node_features,
            allow_automatic_feature=True,
        )
        node_type_features = AbstractClassifierModel.normalize_node_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
        )

        # If necessary, we proceed with the subsampling
        if (
            self._number_of_subsampled_nodes is not None
            and self._graph.get_number_of_nodes() > self._number_of_subsampled_nodes
        ):
            if self._graph.has_node_types():
                train_size = (
                    self._number_of_subsampled_nodes / self._graph.get_number_of_nodes()
                )
                node_type_counts = self._graph.get_node_type_names_counts_hashmap()
                _, least_common_count = min(
                    node_type_counts.items(), key=lambda x: x[1]
                )
                (
                    self._subsampled_node_ids,
                    _,
                ) = self._graph.get_node_label_holdout_indices(
                    train_size=train_size,
                    use_stratification=not (
                        self._graph.has_multilabel_node_types()
                        or self._graph.has_singleton_node_types()
                        or least_common_count * train_size < 1
                    ),
                    random_state=self._random_state,
                )
            else:
                self._subsampled_node_ids = np.random.randint(
                    self._graph.get_number_of_nodes(),
                    size=self._number_of_subsampled_nodes,
                )
            subsampled_node_ids = self._subsampled_node_ids
        else:
            subsampled_node_ids = np.arange(self._graph.get_number_of_nodes())

        assert node_features is not None or node_type_features is not None

        node_transformer = NodeTransformer(aligned_mapping=True)
        node_transformer.fit(
            node_feature=node_features,
            node_type_feature=node_type_features,
        )
        node_embedding = node_transformer.transform(
            subsampled_node_ids,
            node_types=(self._support if self._support.has_node_types() else None),
        )

        assert node_embedding is not None
        assert node_embedding.shape[0] == self.get_number_of_subsampled_nodes(), (
            f"The number of rows of the node embedding ({node_embedding.shape[0]}) "
            f"does not match the number of subsampled nodes ({self.get_number_of_subsampled_nodes()}). "
            f"The graph has {self._graph.get_number_of_nodes()} nodes, and the subsampled "
            f"nodes are {self._subsampled_node_ids.size}."
        )

        self._node_decomposition = self.decompose(node_embedding)
        assert self._node_decomposition.shape[0] == self.get_number_of_subsampled_nodes()

    def _get_positive_edges_embedding(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                pd.DataFrame,
                np.ndarray,
                List[Union[pd.DataFrame, np.ndarray]],
            ]
        ] = None,
    ) -> np.ndarray:
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        embedding: Union[pd.DataFrame, np.ndarray]
            Embedding obtained from SkipGram, CBOW or GloVe or others.
        """
        return self._get_edge_embedding(
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            graph=self._positive_graph,
        )

    def fit_edges(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                pd.DataFrame,
                np.ndarray,
                List[Union[pd.DataFrame, np.ndarray]],
            ]
        ] = None,
    ):
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge type features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        """
        node_features = AbstractClassifierModel.normalize_node_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_features=node_features,
            allow_automatic_feature=True,
        )
        node_type_features = AbstractClassifierModel.normalize_node_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
        )
        edge_type_features = AbstractClassifierModel.normalize_edge_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            edge_type_features=edge_type_features,
            allow_automatic_feature=True,
        )
        edge_features = AbstractClassifierModel.normalize_edge_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            edge_features=edge_features,
            allow_automatic_feature=True,
        )

        self._currently_plotting_edge_embedding = len(node_features) + len(node_type_features) == 0
        self._positive_edge_decomposition = self.decompose(
            self._get_positive_edges_embedding(
                node_features=node_features,
                node_type_features=node_type_features,
                edge_type_features=edge_type_features,
                edge_features=edge_features,
            )
        )

    def _get_edge_embedding(
        self,
        node_features: List[Union[pd.DataFrame, np.ndarray]],
        node_type_features: List[Union[pd.DataFrame, np.ndarray]],
        edge_type_features: List[Union[pd.DataFrame, np.ndarray]],
        edge_features: List[Type[AbstractEdgeFeature]],
        graph: Graph,
    ) -> np.ndarray:
        """Executes aggregation of negative edge embeddings.

        Parameters
        -------------------------
        node_features: List[Union[pd.DataFrame, np.ndarray]],
            The node features to use.
        node_type_features: List[Union[pd.DataFrame, np.ndarray]],
            The node type features to use.
        edge_type_features: List[Union[pd.DataFrame, np.ndarray]],
            The edge type features to use.
        edge_features: List[Type[AbstractEdgeFeature]]
            The edge features to use.
        graph: Graph
            Graph to use to compute the edge embedding.
        """
        graph_transformer = GraphTransformer(
            methods=self._edge_embedding_methods,
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )

        edge_features = [
            feature
            for edge_feature in edge_features
            for feature in edge_feature.get_edge_feature_from_graph(
                graph=graph,
                support=self._support,
            ).values()
        ]
        graph_transformer.fit(
            node_feature=node_features,
            node_type_feature=node_type_features,
            edge_type_features=edge_type_features,
        )
        return graph_transformer.transform(
            graph,
            node_types=(graph if graph.has_node_types() and graph_transformer.has_node_type_features() else None),
            edge_types=(graph if graph.has_edge_types() and graph_transformer.has_edge_type_features() else None),
            edge_features=edge_features
        )

    def _get_negative_edge_embedding(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                pd.DataFrame,
                np.ndarray,
                List[Union[pd.DataFrame, np.ndarray]],
            ]
        ],
    ) -> np.ndarray:
        """Executes aggregation of negative edge embeddings.

        Parameters
        -------------------------
        embedding: Union[pd.DataFrame, np.ndarray]
            Embedding obtained from SkipGram, CBOW or GloVe or others.
        """
        return self._get_edge_embedding(
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            graph=self._negative_graph,
        )

    def fit_negative_and_positive_edges(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                pd.DataFrame,
                np.ndarray,
                List[Union[pd.DataFrame, np.ndarray]],
            ]
        ] = None,
    ):
        """Executes fitting for plotting negative edge embeddings.

        Parameters
        -------------------------
        embedding: Optional[Union[pd.DataFrame, np.ndarray, str]] = None
            Embedding of the graph nodes.
            If a string is provided, we will run the node embedding
            from one of the available methods.
        **embedding_kwargs: Dict
            Kwargs to be forwarded to the node embedding algorithm.
        """
        node_features = AbstractClassifierModel.normalize_node_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_features=node_features,
            allow_automatic_feature=True,
        )
        node_type_features = AbstractClassifierModel.normalize_node_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
        )
        edge_type_features = AbstractClassifierModel.normalize_edge_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            edge_type_features=edge_type_features,
            allow_automatic_feature=True,
        )
        edge_features = AbstractClassifierModel.normalize_edge_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            edge_features=edge_features,
            allow_automatic_feature=True,
        )
        self._currently_plotting_edge_embedding = len(node_features) + len(node_type_features) == 0
        positive_edge_embedding = self._get_positive_edges_embedding(
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )
        negative_edge_embedding = self._get_negative_edge_embedding(
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )
        raw_edge_embedding = np.vstack(
            [positive_edge_embedding, negative_edge_embedding]
        )

        edge_embedding = self.decompose(raw_edge_embedding)
        self._positive_edge_decomposition = edge_embedding[
            : positive_edge_embedding.shape[0]
        ]
        self._negative_edge_decomposition = edge_embedding[
            positive_edge_embedding.shape[0] :
        ]

    def _get_figure_and_axes(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        **kwargs: Dict,
    ) -> Tuple[Figure, Axes]:
        """Return tuple with figure and axes built using provided kwargs and defaults.

        Parameters
        ---------------------------
        figure: Optional[Figure] = None
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        **kwargs: Dict
            Dictionary of parameters to pass to the instantiation of the new figure and axes if one was not initially provided.

        Raises
        ---------------------------
        ValueError
            If the figure object is None but the axes is some or viceversa.

        Returns
        ---------------------------
        Tuple with the figure and axes.
        """
        if figure is not None and axes is not None:
            return (figure, axes)
        if figure is None and axes is not None or figure is not None and axes is None:
            raise ValueError(
                (
                    "Either both figure and axes objects must be None "
                    "and thefore new ones must be created or neither of "
                    "them can be None."
                )
            )
        if self._n_components == 2:
            figure, axes = plt.subplots(
                **{**GraphVisualizer.DEFAULT_SUBPLOT_KWARGS, **kwargs}, squeeze=False
            )
            if axes.size == 1:
                axes = axes.flatten()[0]
        else:
            figure, axes = subplots_3d(
                **{**GraphVisualizer.DEFAULT_SUBPLOT_KWARGS, **kwargs}, squeeze=False
            )
            if axes.size == 1:
                axes = axes.flatten()[0]
        figure.patch.set_facecolor("white")
        return figure, axes

    def _get_complete_title(self, title: str, show_edge_embedding: bool = False) -> str:
        """Return the complete title for the figure.

        Parameters
        -------------------
        title: str
            Initial title to incorporate.
        show_edge_embedding: bool = False
            Whether to add to the title the edge embedding.
        """
        if self._show_graph_name:
            title = f"{title} - {self._graph_name}"

        if (
            show_edge_embedding
            and self._show_embedding_method
            and self._embedding_method_name is not None
            and self._embedding_method_name != "auto"
        ):
            title = f"{title} - {self._embedding_method_name}"

        if (
            show_edge_embedding
            and not self._currently_plotting_edge_embedding
            and self._edge_embedding_methods is not None
        ):
            title = f"{title} - {self._edge_embedding_methods}"

        return sanitize_ml_labels(title)

    def _plot_scatter(
        self,
        points: np.ndarray,
        title: str,
        colors: Optional[List[int]] = None,
        edgecolors: Optional[List[int]] = None,
        labels: List[str] = None,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        apply_tight_layout: bool = True,
        return_collections: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes, Tuple[Collection]]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        points: np.ndarray,
            Points to plot.
        title: str,
            Title to use for the plot.
        colors: Optional[List[int]] = None,
            List of the colors to use for the scatter plot.
        edgecolors: Optional[List[int]] = None,
            List of the edge colors to use for the scatter plot.
        labels: List[str] = None,
            Labels for the different colors.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        loc: str = 'best'
            Position for the legend.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices, we only plot the
            training points.
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_collections: bool = False,
            Whether to return the scatter plot collections.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If given train and test indices overlap.

        Returns
        ------------------------------
        Figure and Axis of the plot, followed by tuple of collections.
        """
        assert points.shape[0] > 0

        if train_indices is not None and test_indices is not None:
            if np.isin(train_indices, test_indices).any():
                raise ValueError("The train and test indices overlap.")

        figure, axes = self._get_figure_and_axes(figure=figure, axes=axes, **kwargs)

        if self._n_components == 2:
            axes.axis("equal")
        else:
            axes.axis("auto")

        scatter_kwargs = {
            **GraphVisualizer.DEFAULT_SCATTER_KWARGS,
            **(dict(linewidths=0) if edgecolors is None else dict(linewidths=0.5)),
            **({} if scatter_kwargs is None else scatter_kwargs),
        }

        train_test_mask = np.zeros((points.shape[0],))

        if train_indices is not None:
            train_test_mask[train_indices] = 1

        if test_indices is not None:
            train_test_mask[test_indices] = 2

        legend_elements = []
        collections = []

        color_map = {
            "blue": "#4e79a7",
            "orange": "#f28e2b",
            "red": "#e15759",
            "cyan": "#76b7b2",
            "green": "#59a14e",
            "yellow": "#edc949",
            "purple": "#b07aa2",
            "pink": "#ff9da7",
            "brown": "#9c755f",
            "grey": "#bab0ac",
            "violet": "#6a3d9a",
            "lime": "#b2df8a",
            "teal": "#1f78b4",
            "lavender": "#b15928",
            "beige": "#8c510a",
            "maroon": "#bf812d",
            "mint": "#66c2a5",
            "olive": "#fc8d62",
        }

        color_hexas = np.array(list(color_map.values()))

        if colors is not None:
            if "cmap" in scatter_kwargs:
                cmap = scatter_kwargs.pop("cmap")
            else:
                color_names_to_be_used = list(color_map.keys())[: int(colors.max() + 1)]
                cmap = ListedColormap(color_hexas[: int(colors.max() + 1)])
        else:
            cmap = None

        if train_indices is None and test_indices is None:
            assert isinstance(points, np.ndarray), "Points must be a numpy array."
            if self._n_components == 2:
                assert (
                    points.shape[1] == 2
                ), f"Points must be 2-dimensional, but they are {points.shape[1]}-dimensional."
            elif self._n_components == 3:
                assert (
                    points.shape[1] == 3
                ), f"Points must be 3-dimensional, but they are {points.shape[1]}-dimensional."
            elif self._n_components == 4:
                points = points[:, :3]
            else:
                raise RuntimeError(
                    f"The number of components must be either 2 or 3, but it is {self._n_components}."
                )

            scatter = axes.scatter(
                *points.T,
                **dict(
                    **dict(
                        c=colors,
                        edgecolors=None if edgecolors is None else cmap(edgecolors),
                        marker=train_marker,
                        cmap=cmap,
                    ),
                    **scatter_kwargs,
                ),
            )
            collections.append(scatter)
            legend_elements.extend(scatter.legend_elements()[0])

        if train_indices is not None:
            train_mask = train_test_mask == 1
            train_scatter = axes.scatter(
                *points[train_mask].T,
                c=colors[train_mask],
                edgecolors=None if edgecolors is None else cmap(edgecolors[train_mask]),
                marker=train_marker,
                cmap=cmap,
                **scatter_kwargs,
            )
            collections.append(train_scatter)
            legend_elements.append(train_scatter.legend_elements()[0])

        if test_indices is not None:
            test_mask = train_test_mask == 2
            test_scatter = axes.scatter(
                *points[test_mask].T,
                c=colors[test_mask],
                edgecolors=None if edgecolors is None else cmap(edgecolors[test_mask]),
                marker=test_marker,
                cmap=cmap,
                **scatter_kwargs,
            )
            collections.append(test_scatter)
            legend_elements.append(test_scatter.legend_elements()[0])

        rectangle_to_fill_legend = matplotlib.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
        )

        if all(e is not None for e in (colors, train_indices, test_indices, labels)):
            unique_train_colors = np.unique(colors[train_mask])
            unique_test_colors = np.unique(colors[test_mask])
            new_legend_elements = []
            train_element_index = 0
            test_element_index = 0
            for color in np.unique(colors):
                new_tuple = []
                if color in unique_train_colors:
                    new_tuple.append(legend_elements[0][train_element_index])
                    train_element_index += 1
                else:
                    new_tuple.append(rectangle_to_fill_legend)
                if color in unique_test_colors:
                    new_tuple.append(legend_elements[1][test_element_index])
                    test_element_index += 1
                else:
                    new_tuple.append(rectangle_to_fill_legend)

                new_legend_elements.append(tuple(new_tuple))
            legend_elements = new_legend_elements

        if show_legend and labels is not None:
            self._set_legend(axes, labels, legend_elements, loc=loc)

        if self._n_components == 2:
            axes.set_axis_off()

        if show_title:
            axes.set_title(title)

        if apply_tight_layout:
            figure.tight_layout()

        if return_collections and not self._rotate:
            return_values = figure, axes, collections
        else:
            return_values = figure, axes

        if return_caption:
            # If the colors were not provided, then this is
            # an heatmap and we need to return its caption
            # if it was requested.
            if colors is None or labels is None:
                return (*return_values, self.get_heatmaps_comments())

            caption = format_list(
                [
                    "{quotations}{label}{quotations} in {color_name}".format(
                        label=label,
                        color_name=color_name,
                        quotations="'" if "other" not in label.lower() else "",
                    )
                    for color_name, label in zip(
                        color_names_to_be_used, self._normalize_label(labels)
                    )
                ]
            )

            return_values = (*return_values, caption)
        return return_values

    def _wrapped_plot_scatter(self, **kwargs):
        if self._rotate:
            try:
                kwargs["loc"] = "lower right"
                path = "{}.{}".format(
                    kwargs["title"].lower().replace(" ", ""), self._video_format
                )
                kwargs["return_caption"] = False
                rotate(
                    self._plot_scatter,
                    path=path,
                    duration=self._duration,
                    fps=self._fps,
                    verbose=self._verbose,
                    **kwargs,
                )
            except (Exception, KeyboardInterrupt) as exception:
                raise exception
            to_display = display_video_at_path(path)
            if to_display is None:
                return ()
            return to_display
        return self._plot_scatter(**kwargs)

    def _plot_types(
        self,
        points: np.ndarray,
        title: str,
        types: List[int],
        type_labels: List[str],
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        predictions: Optional[List[int]] = None,
        k: int = 7,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        **kwargs,
    ) -> Optional[Tuple[Figure, Axes]]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        title: str,
            Title to use for the plot.
        points: np.ndarray,
            Points to plot.
        types: List[int],
            Types of the provided points.
        type_labels: List[str],
            List of the labels for the provided types.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return the caption for the provided types.
        loc: str = 'best'
            Position for the legend.
        predictions: Optional[List[int]] = None,
            List of the labels predicted.
            If None, no prediction is visualized.
        k: int = 7,
            Number of node types to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other",
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).
        ValueError,
            If the number of given type labels does not match the number
            of given type counts.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        # if k > 9:
        #     raise ValueError("Values of k greater than 9 are not supported!")

        if not isinstance(type_labels, np.ndarray):
            raise ValueError(
                "The parameter type_labels was expected to be a numpy array, "
                f"but an object of type `{type(type_labels)}` was provided."
            )

        if not isinstance(types, np.ndarray):
            raise ValueError(
                "The parameter types was expected to be a numpy array, "
                f"but an object of type `{type(types)}` was provided."
            )

        counts = np.bincount(types)
        number_of_non_zero_types = (counts != 0).astype(int).sum()
        number_of_types = len(counts)

        # We want to avoid to use a space in the legend for
        # the "Other" values when they would only contain
        # a single class.
        if number_of_types == k + 1:
            k = k + 1

        top_counts = [
            index
            for index, _ in sorted(
                enumerate(zip(counts, type_labels)), key=lambda x: x[1], reverse=True
            )[:k]
        ]

        type_labels = list(type_labels[top_counts])

        for i, element_type in enumerate(types):
            if element_type not in top_counts:
                types[i] = k
            else:
                types[i] = top_counts.index(element_type)

        if predictions is not None:
            predictions = predictions.copy()
            for i, element_type in enumerate(predictions):
                if element_type not in top_counts:
                    predictions[i] = k
                else:
                    predictions[i] = top_counts.index(element_type)

        if k < number_of_types:
            type_labels.append(other_label.format(number_of_types - k))

        result = self._wrapped_plot_scatter(
            **{
                **dict(
                    return_caption=return_caption,
                    points=points,
                    title=title,
                    colors=types,
                    edgecolors=predictions,
                    labels=type_labels,
                    show_title=show_title,
                    show_legend=show_legend,
                    loc=loc,
                    figure=figure,
                    axes=axes,
                    scatter_kwargs=scatter_kwargs,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_marker=train_marker,
                    test_marker=test_marker,
                ),
                **kwargs,
            }
        )

        if not return_caption or self._rotate:
            return result

        if number_of_non_zero_types == 1:
            return result

        fig, axes, color_caption = result

        test_accuracies = []

        if min(Counter(types).values()) == 1:
            SplitterClass = ShuffleSplit
        else:
            SplitterClass = StratifiedShuffleSplit

        for train_indices, test_indices in SplitterClass(
            n_splits=self._number_of_holdouts_for_cluster_comments,
            test_size=0.3,
            random_state=self._random_state,
        ).split(points, types):
            model = DecisionTreeClassifier(max_depth=5)

            train_x, test_x = points[train_indices], points[test_indices]
            train_y, test_y = types[train_indices], types[test_indices]

            model.fit(train_x, train_y)

            test_accuracies.append(
                balanced_accuracy_score(test_y, model.predict(test_x))
            )

        mean_accuracy = np.mean(test_accuracies)
        std_accuracy = np.std(test_accuracies)

        if mean_accuracy > 0.55:
            if mean_accuracy > 0.90:
                descriptor = "easily recognizable clusters"
            elif mean_accuracy > 0.80:
                descriptor = "recognizable clusters"
            elif mean_accuracy > 0.65:
                descriptor = "some clusters"
            else:
                descriptor = "some possible clusters"
            type_caption = f"The {title.lower()} form {descriptor}"
        else:
            type_caption = (
                f"The {title.lower()} do not appear " "to form recognizable clusters"
            )

        caption = f"{color_caption}. {type_caption} (Balanced accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%})"

        # If requested we automatically add the description of these considerations.
        caption += self.get_separability_comments_description()

        return fig, axes, caption

    def plot_edge_segments(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ) -> Tuple[Figure, Axes]:
        """Plot edge segments between the nodes of the graph.

        Parameters
        ------------------------
        figure: Optional[Figure] = None
            The figure object to plot over.
            If None, a new figure is created automatically and returned.
        axes: Optional[Axes] = None
            The axes object to plot over.
            If None, a new axes is created automatically and returned.
        scatter_kwargs: Optional[Dict] = None
            Dictionary of parameters to pass to the scattering of the edges.
        **kwargs: Dict
            Dictionary of parameters to pass to the instantiation of the new figure and axes if one was not initially provided.

        Returns
        ------------------------
        Tuple with either the provided or created figure and axes.
        """
        if self._node_decomposition is None:
            raise ValueError("Node fitting must be executed before plot.")

        figure, axes = self._get_figure_and_axes(figure=figure, axes=axes, **kwargs)

        if self._subsampled_node_ids is not None:
            edge_node_ids = self._graph.get_edge_ids_from_node_ids(
                node_ids=self._subsampled_node_ids,
                add_selfloops_where_missing=False,
                complete=False,
            )
            edge_node_ids = np.array(
                [
                    [
                        np.where(self._subsampled_node_ids == src)[0][0],
                        np.where(self._subsampled_node_ids == dst)[0][0],
                    ]
                    for src, dst in edge_node_ids
                    if src in self._subsampled_node_ids
                    and dst in self._subsampled_node_ids
                ]
            )
        else:
            edge_node_ids = self._graph.get_edge_node_ids(directed=False)

        if edge_node_ids.size == 0:
            return figure, axes

        if self._n_components == 3:
            lines_collection = Line3DCollection(
                self._node_decomposition[edge_node_ids],
                linewidths=1,
                zorder=0,
                **{
                    **GraphVisualizer.DEFAULT_EDGES_SCATTER_KWARGS,
                    **({} if scatter_kwargs is None else scatter_kwargs),
                },
            )
        else:
            lines_collection = mc.LineCollection(
                self._node_decomposition[edge_node_ids],
                linewidths=1,
                zorder=0,
                **{
                    **GraphVisualizer.DEFAULT_EDGES_SCATTER_KWARGS,
                    **({} if scatter_kwargs is None else scatter_kwargs),
                },
            )
        axes.add_collection(lines_collection)

        return figure, axes

    def plot_nodes(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ) -> Tuple[Figure, Axes]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        loc: str = 'best'
            Position for the legend.
        annotate_nodes: Union[str, bool] = "auto",
            Whether to show the node name when scattering them.
            The default behaviour, "auto", means that it will
            enable this feature automatically when the graph has
            less than 100 nodes.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._node_decomposition is None:
            raise ValueError("Node fitting must be executed before plot.")

        if show_edges == "auto":
            show_edges = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure, axes, scatter_kwargs=edge_scatter_kwargs, **kwargs
            )

        returned_values = self._wrapped_plot_scatter(
            points=self._node_decomposition,
            title=self._get_complete_title("Nodes embedding"),
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
            **kwargs,
        )

        if annotate_nodes and returned_values:
            figure, axes = returned_values[:2]
            self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_decomposition,
            )

        return self._handle_notebook_display(*returned_values)

    def annotate_nodes(
        self, figure: Figure, axes: Axes, points: np.ndarray
    ) -> Tuple[Figure, Axes]:
        """Annotate nodes of provided graph.

        Parameters
        ------------------------------
        figure: Figure,
            Figure to use to plot.
        axes: Axes,
            Axes to use to plot.
        points: np.ndarray,
            Points to plot.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        for node_name, point in zip(
            (
                self._graph.get_node_name_from_node_id(node_id)
                for node_id in self.iterate_subsampled_node_ids()
            ),
            points,
        ):
            if point.size == 3:
                axes.add_artist(
                    Annotation3D(node_name, point, fontsize=8, ha="center", va="center")
                )
            else:
                axes.annotate(node_name, point, fontsize=8, ha="center", va="center")
        return (figure, axes)

    def plot_edges(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ) -> Tuple[Figure, Axes]:
        """Plot edge embedding of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._positive_edge_decomposition is None:
            raise ValueError(
                "Edge fitting must be executed before plot. "
                "Please do call the `visualizer.fit_edges()` "
                "method before plotting the nodes."
            )

        return self._handle_notebook_display(
            *self._wrapped_plot_scatter(
                points=self._positive_edge_decomposition,
                title=self._get_complete_title(
                    "Edges embedding", show_edge_embedding=True
                ),
                figure=figure,
                axes=axes,
                scatter_kwargs=scatter_kwargs,
                train_indices=train_indices,
                test_indices=test_indices,
                train_marker=train_marker,
                test_marker=test_marker,
                show_title=show_title,
                show_legend=show_legend,
                loc=loc,
                **kwargs,
            )
        )

    def plot_positive_and_negative_edges(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ) -> Tuple[Figure, Axes]:
        """Plot edge embedding of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if (
            self._positive_edge_decomposition is None
            or self._negative_edge_decomposition is None
        ):
            raise ValueError(
                "Positive and negative edge fitting must be executed before plot. "
                "Please do call the `visualizer.fit_negative_and_positive_edges()` "
                "method before plotting the nodes."
            )

        points = np.vstack(
            [
                self._negative_edge_decomposition,
                self._positive_edge_decomposition,
            ]
        )

        types = np.concatenate(
            [
                np.zeros(self._negative_edge_decomposition.shape[0], dtype="int64"),
                np.ones(self._positive_edge_decomposition.shape[0], dtype="int64"),
            ]
        )

        points, types = self._shuffle(points, types)

        returned_values = self._plot_types(
            points=points,
            title=self._get_complete_title("Edge prediction", show_edge_embedding=True),
            types=types,
            type_labels=np.array(["Non-existent", "Existent"]),
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            **kwargs,
        )

        if not return_caption or self._rotate:
            if self._rotate:
                return returned_values
            return self._handle_notebook_display(*returned_values)

        fig, axes, types_caption = returned_values

        caption = (
            f"<i>Existent and non-existent edges</i>: {types_caption}."
            + self.get_non_existing_edges_sampling_description()
        )

        return self._handle_notebook_display(fig, axes, caption)

    def _plot_positive_and_negative_edges_metric(
        self,
        metric_name: str,
        edge_metric_callback: Optional[Callable[[Graph], np.ndarray]] = None,
        edge_metrics: Optional[np.ndarray] = None,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot provided edge metric heatmap for positive and negative edges.

        Parameters
        ------------------------------
        metric_name: str
            Name of the metric that will be computed.
        edge_metric_callback: Optional[Callable[[int, int], float]] = None
            Callback to compute the metric given two nodes.
        edge_metrics: Optional[np.ndarray] = None
            Precomputed edge metrics.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot the scatter plot.
            If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._positive_edge_decomposition is None:
            raise ValueError(
                "Positive and negative edge fitting must be executed before plot. "
                "Please do call the `visualizer.fit_negative_and_positive_edges()` "
                "method before plotting the nodes."
            )

        if edge_metrics is None and edge_metric_callback is None:
            raise ValueError(
                "Neither the edge metrics nor the edge metric callback was "
                "provided and therefore we cannot plot the edge metrics."
            )

        if edge_metrics is None:
            edge_metrics = np.concatenate(
                (
                    edge_metric_callback(subgraph=self._negative_graph),
                    edge_metric_callback(subgraph=self._positive_graph),
                )
            )

            # Filter the edge metrics relative to edges that are not
            # to be displayed.
            if not self._graph.is_directed():
                edge_metrics = edge_metrics[
                    np.concatenate(
                        [
                            self._negative_graph.get_directed_source_node_ids()
                            <= self._negative_graph.get_directed_destination_node_ids(),
                            self._positive_graph.get_directed_source_node_ids()
                            <= self._positive_graph.get_directed_destination_node_ids(),
                        ]
                    )
                ]

        points = np.vstack(
            [
                self._negative_edge_decomposition,
                self._positive_edge_decomposition,
            ]
        )

        points, shuffled_edge_metrics = self._shuffle(points, edge_metrics)

        returned_values = self._wrapped_plot_scatter(
            points=points,
            title=self._get_complete_title(metric_name, show_edge_embedding=True),
            colors=shuffled_edge_metrics + sys.float_info.epsilon,
            figure=figure,
            axes=axes,
            scatter_kwargs={
                **({} if scatter_kwargs is None else scatter_kwargs),
                "cmap": plt.cm.get_cmap("RdYlBu"),
                "norm": LogNorm(),
            },
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            return_collections=True,
            **kwargs,
        )

        if not self._rotate:
            if return_caption:
                figure, axes, scatter, color_caption = returned_values
            else:
                figure, axes, scatter = returned_values

            color_bar = figure.colorbar(scatter[0], ax=axes)
            color_bar.set_alpha(1)
            color_bar.draw_all()
            returned_values = figure, axes

        if not return_caption:
            return self._handle_notebook_display(figure, axes, scatter)

        edge_metrics = edge_metrics.reshape((-1, 1))

        types = np.concatenate(
            [
                np.zeros(self._negative_edge_decomposition.shape[0], dtype=bool),
                np.ones(self._positive_edge_decomposition.shape[0], dtype=bool),
            ]
        )

        test_accuracies = []

        if min(Counter(types).values()) == 1:
            SplitterClass = ShuffleSplit
        else:
            SplitterClass = StratifiedShuffleSplit

        for train_indices, test_indices in SplitterClass(
            n_splits=self._number_of_holdouts_for_cluster_comments,
            test_size=0.3,
            random_state=self._random_state,
        ).split(edge_metrics, types):
            model = DecisionTreeClassifier(max_depth=5)

            train_x, test_x = edge_metrics[train_indices], edge_metrics[test_indices]
            train_y, test_y = types[train_indices], types[test_indices]

            model.fit(train_x, train_y)

            test_accuracies.append(
                balanced_accuracy_score(test_y, model.predict(test_x))
            )

        mean_accuracy = np.mean(test_accuracies)
        std_accuracy = np.std(test_accuracies)

        if mean_accuracy > 0.55:
            if mean_accuracy > 0.90:
                descriptor = "is an outstanding edge prediction feature"
            elif mean_accuracy > 0.65:
                descriptor = "is a good edge prediction feature"
            else:
                descriptor = "may be considered an edge prediction feature"
            metric_caption = f"This metric {descriptor}"
        else:
            metric_caption = "The metric is not useful as an " "edge prediction feature"

        caption = f"<i>{metric_name} heatmap</i>. {metric_caption} (Balanced accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%}).{color_caption}"

        # If requested we automatically add the description of these considerations.
        caption += self.get_separability_comments_description()
        caption += self.get_non_existing_edges_sampling_description()

        return self._handle_notebook_display(*returned_values, caption=caption)

    def _plot_positive_and_negative_edges_metric_histogram(
        self,
        metric_name: str,
        edge_metric_callback: Optional[Callable[[int, int], float]] = None,
        edge_metrics: Optional[np.ndarray] = None,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph node degree distribution.

        Parameters
        ------------------------------
        metric_name: str
            Name of the metric that will be computed.
        edge_metric_callback: Optional[Callable[[int, int], float]] = None
            Callback to compute the metric given two nodes.
        edge_metrics: Optional[np.ndarray] = None
            Precomputed edge metrics.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        if axes is None:
            figure, axes = plt.subplots(figsize=(5, 5))
            figure.patch.set_facecolor("white")

        if self._negative_graph.is_directed() and edge_metrics is None:
            number_of_negative_edges = (
                self._negative_graph.get_number_of_directed_edges()
            )
        else:
            number_of_negative_edges = (
                self._negative_graph.get_number_of_undirected_edges()
            )

        if edge_metrics is None:
            edge_metrics = np.concatenate(
                (
                    edge_metric_callback(subgraph=self._negative_graph),
                    edge_metric_callback(subgraph=self._positive_graph),
                )
            )

        assert len(edge_metrics.shape) == 1

        number_of_bins = max(50, np.log(edge_metrics.size + 1).astype("int64") + 1)

        axes.hist(
            edge_metrics[:number_of_negative_edges],
            bins=number_of_bins,
            log=True,
            label="Non-existent",
        )
        axes.hist(
            edge_metrics[number_of_negative_edges:],
            bins=number_of_bins,
            log=True,
            alpha=0.7,
            label="Existent",
        )
        axes.set_xlim(edge_metrics.min(), edge_metrics.max())
        axes.set_ylabel("Counts (log scale)")
        axes.set_xlabel(metric_name)
        axes.legend(
            loc="best",
            prop={"size": 8},
        )
        axes.set_title(
            f"{metric_name} distribution of graph {self._graph_name}"
            if self._show_graph_name
            else f"{metric_name} distribution"
        )

        if apply_tight_layout:
            figure.tight_layout()

        if not return_caption:
            return self._handle_notebook_display(figure, axes)

        caption = (
            f"<i>{metric_name} distribution.</i> {metric_name} values are on the "
            "horizontal axis and edge counts are on the vertical axis on a logarithmic scale."
        )

        return self._handle_notebook_display(figure, axes, caption=caption)

    def plot_positive_and_negative_adamic_adar_histogram(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges Adamic Adar metric distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_positive_and_negative_edges_metric_histogram(
            metric_name="Adamic-Adar",
            edge_metric_callback=self._support.get_adamic_adar_scores,
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def plot_positive_and_negative_edges_adamic_adar(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot Adamic Adar metric heatmap for sampled existent and non-existent edges.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_positive_and_negative_edges_metric(
            metric_name="Adamic-Adar",
            edge_metric_callback=self._support.get_adamic_adar_scores,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            **kwargs,
        )

    def plot_positive_and_negative_preferential_attachment_histogram(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges Adamic Adar metric distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_positive_and_negative_edges_metric_histogram(
            metric_name="Preferential Attachment",
            edge_metric_callback=self._support.get_preferential_attachment_scores,
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def plot_positive_and_negative_edges_preferential_attachment(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot Preferential Attachment metric heatmap for sampled existent and non-existent edges.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_positive_and_negative_edges_metric(
            metric_name="Preferential Attachment",
            edge_metric_callback=self._support.get_preferential_attachment_scores,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            **kwargs,
        )

    def plot_positive_and_negative_jaccard_coefficient_histogram(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges Jaccard Coefficient metric distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_positive_and_negative_edges_metric_histogram(
            metric_name="Jaccard Coefficient",
            edge_metric_callback=self._support.get_jaccard_coefficient_scores,
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def plot_positive_and_negative_edges_jaccard_coefficient(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot Jaccard Coefficient metric heatmap for sampled existent and non-existent edges.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_positive_and_negative_edges_metric(
            metric_name="Jaccard Coefficient",
            edge_metric_callback=self._support.get_jaccard_coefficient_scores,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            **kwargs,
        )

    def plot_positive_and_negative_resource_allocation_index_histogram(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges Resource Allocation Index metric distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_positive_and_negative_edges_metric_histogram(
            metric_name="Resource Allocation Index",
            edge_metric_callback=self._support.get_resource_allocation_index_scores,
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def plot_positive_and_negative_edges_resource_allocation_index(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot Resource Allocation Index metric heatmap for sampled existent and non-existent edges.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_positive_and_negative_edges_metric(
            metric_name="Resource Allocation Index",
            edge_metric_callback=self._support.get_resource_allocation_index_scores,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            return_caption=return_caption,
            show_legend=show_legend,
            loc=loc,
            **kwargs,
        )

    def _get_flatten_unknown_node_ontologies(self) -> Tuple[List[str], np.ndarray]:
        """Returns unique ontologies and node ontologies adjusted for the current instance."""
        ontology_names = [
            self._graph.get_ontology_from_node_id(node_id)
            for node_id in self.iterate_subsampled_node_ids()
        ]

        # The following is needed to normalize the multiple types
        ontologies_counts = Counter(ontology_names)
        ontologies_by_frequencies = {
            ontology: i
            for i, (ontology, _) in enumerate(
                sorted(ontologies_counts.items(), key=lambda x: x[1], reverse=True)
            )
        }
        unknown_ontology_id = len(ontologies_counts)

        return (
            list(ontologies_by_frequencies.keys()),
            np.fromiter(
                (
                    unknown_ontology_id
                    if ontology is None
                    else ontologies_by_frequencies[ontology]
                    for ontology in ontology_names
                ),
                dtype=np.uint32,
            ),
        )

    def _get_flatten_multi_label_and_unknown_node_types(self) -> np.ndarray:
        """Returns flattened node type IDs adjusted for the current instance."""
        # The following is needed to normalize the multiple types
        node_types_counts = self._graph.get_node_type_id_counts_hashmap()
        top_10_node_types = {
            node_type: 50 - i
            for i, node_type in enumerate(
                sorted(node_types_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            )
        }
        node_types_counts = {
            node_type: top_10_node_types.get(node_type, 0)
            for node_type in node_types_counts
        }
        node_types_number = self._graph.get_number_of_node_types()
        unknown_node_types_id = node_types_number

        # When we have multiple node types for a given node, we set it to
        # the most common node type of the set.
        return np.fromiter(
            (
                unknown_node_types_id
                if node_type_ids is None
                else sorted(
                    node_type_ids,
                    key=lambda node_type: node_types_counts[node_type],
                    reverse=True,
                )[0]
                for node_type_ids in (
                    self._graph.get_node_type_ids_from_node_id(node_id)
                    for node_id in self.iterate_subsampled_node_ids()
                )
            ),
            dtype=np.uint32,
        )

    def _get_flatten_unknown_edge_types(self) -> np.ndarray:
        """Returns flattened edge type IDs adjusted for the current instance."""
        # The following is needed to normalize the unknown types
        unknown_edge_types_id = self._graph.get_number_of_edge_types()
        # When we have multiple node types for a given node, we set it to
        # the most common node type of the set.
        return np.fromiter(
            (
                unknown_edge_types_id if edge_type_id is None else edge_type_id
                for edge_type_id in (
                    self._positive_graph.get_directed_edge_type_ids()
                    if self._positive_graph.is_directed()
                    else self._positive_graph.get_upper_triangular_edge_type_ids()
                )
            ),
            dtype=np.uint32,
        )

    def plot_node_types(
        self,
        node_type_predictions: Optional[List[int]] = None,
        k: int = 13, #TODO: change to 13 from 7
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other {} node types",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        # change size of legend
        legend_size: int = 10,
        return_caption: bool = True,
        loc: str = "best",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        annotate_nodes: Union[str, bool] = "auto",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        node_type_predictions: Optional[List[int]] = None,
            Predictions of the node types.
        k: int = 7,
            Number of node types to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} node types"
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if not self._graph.has_node_types():
            raise ValueError("The graph does not have node types!")

        if self._node_decomposition is None:
            raise ValueError(
                "Node fitting must be executed before plot. "
                "Please do call the `visualizer.fit_nodes()` "
                "method before plotting the nodes."
            )

        if show_edges == "auto":
            show_edges = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure, axes, scatter_kwargs=edge_scatter_kwargs, **kwargs
            )

        if self._subsampled_node_ids is not None:
            if node_type_predictions is not None:
                node_type_predictions = node_type_predictions[self._subsampled_node_ids]

            if train_indices is not None:
                train_indices = np.fromiter(
                    (
                        np.where(self._subsampled_node_ids == node_id)[0][0]
                        for node_id in train_indices
                        if node_id in self._subsampled_node_ids
                    ),
                    dtype=train_indices.dtype,
                )
            if test_indices is not None:
                test_indices = np.fromiter(
                    (
                        np.where(self._subsampled_node_ids == node_id)[0][0]
                        for node_id in test_indices
                        if node_id in self._subsampled_node_ids
                    ),
                    dtype=test_indices.dtype,
                )

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_number_of_nodes() < 50 and not self._rotate

        node_types = self._get_flatten_multi_label_and_unknown_node_types()

        node_type_names_iter = (
            self._graph.get_node_type_name_from_node_type_id(node_id)
            for node_id in range(self._graph.get_number_of_node_types())
        )

        if self._graph.has_unknown_node_types():
            node_type_names_iter = itertools.chain(
                node_type_names_iter, iter(("Unknown",))
            )

        node_type_names = np.array(
            list(node_type_names_iter),
            dtype=str,
        )

        returned_values = self._plot_types(
            self._node_decomposition,
            self._get_complete_title("Node types"),
            types=node_types,
            type_labels=node_type_names,
            predictions=node_type_predictions,
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            **kwargs,
        )

        if annotate_nodes:
            figure, axes = returned_values[:2]
            self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_decomposition,
            )

        if not return_caption or self._rotate:
            if self._rotate:
                return returned_values
            return self._handle_notebook_display(*returned_values)

        # TODO! Add caption node abount gaussian ball!
        fig, axes, types_caption = returned_values

        caption = f"<i>Node types</i>: {types_caption}."

        return self._handle_notebook_display(fig, axes, caption=caption)

    def plot_node_ontologies(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other {} ontologies",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        annotate_nodes: Union[str, bool] = "auto",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} node ontologies"
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o"
            The marker to use to draw the training points.
        test_marker: str = "X"
            The marker to use to draw the test points.
        show_title: bool = True
            Whether to show the figure title.
        show_legend: bool = True
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError
            If node fitting was not yet executed.
        ValueError
            If the graph does not have node ontologies.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        self._graph.must_have_node_ontologies()

        if self._node_decomposition is None:
            raise ValueError(
                "Node fitting must be executed before plot. "
                "Please do call the `visualizer.fit_nodes()` "
                "method before plotting the nodes."
            )

        if show_edges == "auto":
            show_edges = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure, axes, scatter_kwargs=edge_scatter_kwargs, **kwargs
            )

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_number_of_nodes() < 50 and not self._rotate

        unique_ontologies, ontology_ids = self._get_flatten_unknown_node_ontologies()

        if self._graph.has_unknown_node_ontologies():
            unique_ontologies.append("Unknown")

        unique_ontologies = np.array(
            unique_ontologies,
            dtype=str,
        )

        returned_values = self._plot_types(
            self._node_decomposition,
            self._get_complete_title("Node ontologies"),
            types=ontology_ids,
            type_labels=unique_ontologies,
            k=13, #TODO change to 13 from 7
            figure=figure,
            axes=axes,
            return_caption=return_caption,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
            **kwargs,
        )

        if annotate_nodes:
            fig, axes = returned_values[:2]
            self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_decomposition,
            )

        if not return_caption:
            return self._handle_notebook_display(*returned_values)

        # TODO! Add caption node abount gaussian ball!
        fig, axes, types_caption = returned_values

        caption = f"<i>Detected node ontologies</i>: {types_caption}."

        return self._handle_notebook_display(fig, axes, caption=caption)

    def plot_connected_components(
        self,
        k: int = 7,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other {} components",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        k: int = 7,
            Number of components to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} components",
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        annotate_nodes: Union[str, bool] = "auto"
            Whether to show the node names when plotting them.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        show_edges: Union[str, bool] = "auto"
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._node_decomposition is None:
            raise ValueError("Node fitting must be executed before plot.")

        if show_edges == "auto":
            show_edges = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure, axes, scatter_kwargs=edge_scatter_kwargs, **kwargs
            )

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_number_of_nodes() < 50 and not self._rotate

        components, components_number, _, _ = self._support.get_connected_components()
        sizes = np.bincount(components, minlength=components_number).tolist()
        sizes_backup = list(sizes)
        largest_component_size = max(sizes)

        # We do not show a single "connected component"
        # when such a component has size 3 or lower, as it would
        # really be pushing what a "Main component" is meant to be.
        if largest_component_size <= 3:
            largest_component_size = None

        labels = ["Size {}".format(size) for size in [size for size in sizes]]

        # Creating a new "component" with all the nodes of the
        # categories of `Triples`, `Tuples` and `Singletons`.
        current_component_number = components_number
        for expected_component_size, component_name in (
            (1, "Singletons"),
            (2, "Tuples"),
            (3, "Triples"),
            (None, "Minor components"),
        ):
            new_component_size = 0
            for i in range(len(components)):
                # If this is one of the newly created components
                # we skip it.
                if components[i] >= components_number:
                    continue
                nodes_component_size = sizes_backup[components[i]]
                is_in_odd_component = (
                    expected_component_size is not None
                    and nodes_component_size == expected_component_size
                )
                is_in_minor_component = (
                    expected_component_size is None
                    and largest_component_size is not None
                    and nodes_component_size < largest_component_size
                )
                if is_in_odd_component or is_in_minor_component:
                    sizes[components[i]] -= 1
                    components[i] = current_component_number
                    new_component_size += 1

            if new_component_size > 0:
                labels.append("{}".format(component_name))
                sizes.append(new_component_size)
                current_component_number += 1

        if self._subsampled_node_ids is not None:
            components = components[self._subsampled_node_ids]

        components_remapping = {
            old_component_id: new_component_id
            for new_component_id, (old_component_id, _) in enumerate(
                sorted(
                    [
                        (old_component_id, size)
                        for old_component_id, size in enumerate(sizes)
                        if size > 0
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        }

        labels = [
            labels[old_component_id] for old_component_id in components_remapping.keys()
        ]

        if largest_component_size is not None:
            labels[0] = "Main component"

        # Remap all other components
        for i in range(len(components)):
            components[i] = components_remapping[components[i]]

        returned_values = self._plot_types(
            self._node_decomposition,
            self._get_complete_title("Components"),
            types=components,
            type_labels=np.array(labels, dtype=str),
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            **kwargs,
        )

        if annotate_nodes:
            figure, axes = returned_values[:2]
            self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_decomposition,
            )

        if not return_caption or self._rotate:
            if self._rotate:
                return returned_values
            return self._handle_notebook_display(*returned_values)

        # TODO! Add caption node abount gaussian ball!
        fig, axes, types_caption = returned_values

        caption = f"<i>Connected components</i>: {types_caption}."

        return self._handle_notebook_display(fig, axes, caption=caption)

    def _plot_node_metric(
        self,
        metric: np.ndarray,
        metric_name: str,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ):
        """Plot node degrees heatmap.

        Parameters
        ------------------------------
        metric: np.ndarray
            Metric to plot.
        metric_name: str,
            Name of the metric to plot.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._node_decomposition is None:
            raise ValueError(
                "Node fitting must be executed before plot."
                "Please do call the `visualizer.fit_nodes()` "
                "method before plotting the nodes."
            )

        assert isinstance(metric_name, str)
        assert len(metric_name) > 0
        assert isinstance(metric, np.ndarray)
        assert metric.shape[0] == self._graph.get_number_of_nodes() or metric.shape[0] == self._node_decomposition.shape[0], (
            f"Metric shape {metric.shape} does not match "
            f"node decomposition shape {self._node_decomposition.shape}. "
            f"The graph has {self._graph.get_number_of_nodes()} nodes."
        )

        if metric.shape[0] == self._support.get_number_of_nodes():
            metric = np.fromiter(
                (metric[node_id] for node_id in self.iterate_subsampled_node_ids()),
                dtype=metric.dtype,
            )

        assert metric.shape[0] == self.get_number_of_subsampled_nodes()

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if show_edges == "auto":
            show_edges = self._graph.get_number_of_nodes() < 50 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure, axes, scatter_kwargs=edge_scatter_kwargs, **kwargs
            )

        if self._rotate:
            return_caption = False

        mask = metric > 0

        if not np.any(mask):
            zeroed = True
            mask = np.full_like(mask, True, dtype=bool)
        else:
            zeroed = False

        returned_values = self._wrapped_plot_scatter(
            points=self._node_decomposition[mask],
            title=self._get_complete_title(metric_name),
            colors=metric[mask],
            figure=figure,
            axes=axes,
            scatter_kwargs={
                **({} if scatter_kwargs is None else scatter_kwargs),
                "cmap": plt.cm.get_cmap("RdYlBu"),
                **(
                    {"norm": LogNorm()}
                    if use_log_scale and not zeroed
                    else {}
                ),
            },
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            return_collections=True,
            **kwargs,
        )

        if not self._rotate:
            if return_caption:
                figure, axes, scatter, color_caption = returned_values
            else:
                figure, axes, scatter = returned_values
            color_bar = figure.colorbar(scatter[0], ax=axes)
            color_bar.set_alpha(1)
            color_bar.draw_all()

        if annotate_nodes:
            figure, axes = self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_decomposition,
            )

        if not return_caption or self._rotate:
            if self._rotate:
                return returned_values
            return self._handle_notebook_display(figure, axes, scatter)

        # TODO! Add caption node abount gaussian ball!
        caption = f"<i>{metric_name} heatmap</i>.{color_caption}"

        return self._handle_notebook_display(figure, axes, caption=caption)

    def plot_node_degrees(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ):
        """Plot node degrees heatmap.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_node_metric(
            metric=np.fromiter(
                (
                    self._support.get_node_degree_from_node_id(node_id)
                    for node_id in self.iterate_subsampled_node_ids()
                ),
                dtype=np.uint32,
            ),
            metric_name="Node degrees",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            use_log_scale=use_log_scale,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            annotate_nodes=annotate_nodes,
            show_edges=show_edges,
            edge_scatter_kwargs=edge_scatter_kwargs,
            **kwargs,
        )

    def plot_node_triangles(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ):
        """Plot Triangless heatmap.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_node_metric(
            metric=self._support.get_number_of_triangles_per_node(),
            metric_name="Triangless",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            use_log_scale=use_log_scale,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            annotate_nodes=annotate_nodes,
            show_edges=show_edges,
            edge_scatter_kwargs=edge_scatter_kwargs,
            **kwargs,
        )

    def plot_node_squares(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ):
        """Plot node squares heatmap.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_node_metric(
            metric=self._support.get_number_of_squares_per_node(),
            metric_name="Squares",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            use_log_scale=use_log_scale,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            annotate_nodes=annotate_nodes,
            show_edges=show_edges,
            edge_scatter_kwargs=edge_scatter_kwargs,
            **kwargs,
        )

    def plot_approximated_closeness_centrality(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ):
        """Plot approximated closeness centrality heatmap.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_node_metric(
            metric=self._support.get_approximated_closeness_centrality(),
            metric_name="Approx closeness centrality",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            use_log_scale=use_log_scale,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            annotate_nodes=annotate_nodes,
            show_edges=show_edges,
            edge_scatter_kwargs=edge_scatter_kwargs,
            **kwargs,
        )

    def plot_approximated_harmonic_centrality(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: Union[str, bool] = "auto",
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict,
    ):
        """Plot approximated harmonic centrality heatmap.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        show_edges: Union[str, bool] = "auto",
            Whether to show edges between the different nodes
            shown in the scatter plot.
            It is enabled by default with `auto` when the graph
            has less than 50 nodes.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_node_metric(
            metric=self._support.get_approximated_harmonic_centrality(),
            metric_name="Approx harmonic centrality",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            use_log_scale=use_log_scale,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            annotate_nodes=annotate_nodes,
            show_edges=show_edges,
            edge_scatter_kwargs=edge_scatter_kwargs,
            **kwargs,
        )

    def plot_edge_types(
        self,
        edge_type_predictions: Optional[List[int]] = None,
        k: int = 7,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other {} edge types",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        edge_type_predictions: Optional[List[int]] = None,
            Predictions of the edge types.
        k: int = 7,
            Number of edge types to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} edge types",
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption for the image.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If the graph does not have edge types.
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if not self._graph.has_edge_types():
            raise ValueError("The graph does not have edge types!")

        if self._positive_edge_decomposition is None:
            raise ValueError(
                "Edge fitting was not yet executed! "
                "Please do call the `visualizer.fit_edges()` "
                "method before plotting the nodes."
            )

        if self._rotate:
            return_caption = False

        edge_types = self._get_flatten_unknown_edge_types()

        edge_type_names_iter = (
            self._graph.get_edge_type_name_from_edge_type_id(edge_id)
            for edge_id in range(self._graph.get_number_of_edge_types())
        )

        if self._graph.has_unknown_edge_types():
            edge_type_names_iter = itertools.chain(
                edge_type_names_iter, iter(("Unknown",))
            )

        edge_type_names = np.array(list(edge_type_names_iter), dtype=str)

        returned_values = self._plot_types(
            self._positive_edge_decomposition,
            self._get_complete_title("Edge types", show_edge_embedding=True),
            types=edge_types,
            type_labels=edge_type_names,
            predictions=edge_type_predictions,
            k=k,
            figure=figure,
            axes=axes,
            return_caption=return_caption,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
            **kwargs,
        )

        if self._rotate:
            return returned_values

        if not return_caption:
            return self._handle_notebook_display(*returned_values)

        # TODO! Add caption node abount gaussian ball!
        fig, axes, types_caption = returned_values

        caption = f"<i>Edge types</i>: {types_caption}"

        return self._handle_notebook_display(fig, axes, caption=caption)

    def plot_edge_weights(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption for this plot.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if not self._positive_graph.has_edge_weights():
            raise ValueError("The graph does not have edge weights!")

        if self._positive_edge_decomposition is None:
            raise ValueError(
                "Edge fitting must be executed before plot. "
                "Please do call the `visualizer.fit_edges()` "
                "method before plotting the nodes."
            )

        if self._positive_graph.is_directed():
            weights = self._positive_graph.get_directed_edge_weights()
        else:
            weights = self._positive_graph.get_undirected_edge_weights()

        returned_values = self._wrapped_plot_scatter(
            points=self._positive_edge_decomposition,
            title=self._get_complete_title("Edge weights", show_edge_embedding=True),
            colors=weights,
            figure=figure,
            axes=axes,
            scatter_kwargs={
                **({} if scatter_kwargs is None else scatter_kwargs),
                "cmap": plt.cm.get_cmap("RdYlBu"),
                "norm": LogNorm(),
            },
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            return_caption=return_caption,
            loc=loc,
            return_collections=True,
            **kwargs,
        )

        if not self._rotate:
            if return_caption:
                figure, axes, scatter, color_caption = returned_values
            else:
                figure, axes, scatter = returned_values
            color_bar = figure.colorbar(scatter[0], ax=axes)
            color_bar.set_alpha(1)
            color_bar.draw_all()
            returned_values = figure, axes

        if not return_caption:
            return self._handle_notebook_display(*returned_values)

        caption = f"<i>Edge weights heatmap</i>{color_caption}."

        return self._handle_notebook_display(*returned_values, caption=caption)

    def _plot_positive_and_negative_edges_distance_histogram(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        distance_name: str,
        distance_callback: str,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges distance distribution.

        Parameters
        ------------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node type features to use.
        distance_name: str
            The title for the heatmap.
        distance_callback: str
            The callback to use to compute the distances.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        graph_transformer = GraphTransformer(
            methods=distance_callback,
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )
        graph_transformer.fit(
            node_feature=node_features,
            node_type_feature=node_type_features,
        )

        return self._plot_positive_and_negative_edges_metric_histogram(
            metric_name=distance_name,
            edge_metrics=np.vstack(
                [
                    graph_transformer.transform(
                        self._negative_graph,
                        node_types=(self._negative_graph if graph_transformer.has_node_type_features() else None),
                    ),
                    graph_transformer.transform(
                        self._positive_graph,
                        node_types=(self._positive_graph if graph_transformer.has_node_type_features() else None),
                    ),
                ]
            ).flatten(),
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def _plot_positive_and_negative_edges_distance(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        distance_name: str,
        distance_callback: str,
        **kwargs: Dict,
    ):
        """Plot distances of node features for positive and negative edges.

        Parameters
        ------------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node type features to use.
        distance_name: str
            The title for the heatmap.
        distance_callback: str
            The callback to use to compute the distances.
        **kwargs: Dict
            Additional kwargs to forward.

        Raises
        ------------------------------
        ValueError
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        graph_transformer = GraphTransformer(
            methods=distance_callback,
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )

        graph_transformer.fit(
            node_feature=node_features,
            node_type_feature=node_type_features,
        )

        return self._plot_positive_and_negative_edges_metric(
            metric_name=distance_name,
            edge_metrics=np.vstack(
                [
                    graph_transformer.transform(
                        self._negative_graph,
                        node_types=(self._negative_graph if graph_transformer.has_node_type_features() else None),
                    ),
                    graph_transformer.transform(
                        self._positive_graph,
                        node_types=(self._positive_graph if graph_transformer.has_node_type_features() else None),
                    ),
                ]
            ),
            **kwargs,
        )

    def plot_positive_and_negative_edges_euclidean_distance_histogram(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges Euclidean distance distribution.

        Parameters
        ------------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node type features to use.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_positive_and_negative_edges_distance_histogram(
            node_features=node_features,
            node_type_features=node_type_features,
            distance_name="Euclidean distance",
            distance_callback="L2Distance",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def plot_positive_and_negative_edges_euclidean_distance(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot L2 Distance heatmap for sampled existent and non-existent edges.

        Parameters
        ------------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node type features to use.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        return self._plot_positive_and_negative_edges_distance(
            node_features=node_features,
            node_type_features=node_type_features,
            distance_name="Euclidean distance",
            distance_callback="L2Distance",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            return_caption=return_caption,
            show_legend=show_legend,
            loc=loc,
            **kwargs,
        )

    def plot_positive_and_negative_edges_cosine_similarity_histogram(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the positive and negative edges Cosine similarity distribution.

        Parameters
        ------------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node type features to use.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_positive_and_negative_edges_distance_histogram(
            node_features=node_features,
            node_type_features=node_type_features,
            distance_name="Cosine similarity",
            distance_callback="CosineSimilarity",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            return_caption=return_caption,
        )

    def plot_positive_and_negative_edges_cosine_similarity(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ],
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        return_caption: bool = True,
        loc: str = "best",
        **kwargs: Dict,
    ):
        """Plot Cosine similarity heatmap for sampled existent and non-existent edges.

        Parameters
        ------------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]]
            The node type features to use.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        return_caption: bool = True,
            Whether to return a caption.
        loc: str = 'best'
            Position for the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        returned_values = self._plot_positive_and_negative_edges_distance(
            node_features=node_features,
            node_type_features=node_type_features,
            distance_name="Cosine similarity",
            distance_callback="CosineSimilarity",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            return_caption=return_caption,
            show_legend=show_legend,
            loc=loc,
            **kwargs,
        )

        if not return_caption:
            return returned_values

        figure, axes, caption = returned_values

        caption += (
            " Do note that the cosine similarity has been shifted from the "
            "range of [-1, 1] to the range [0, 2] "
            "to be visualized in a logarithmic heatmap."
        )

        return figure, axes, caption

    def plot_dot(self, engine: str = "neato"):
        """Return dot plot of the current graph.

        Parameters
        ------------------------------
        engine: str = "neato"
            The engine to use to visualize the graph.

        Raises
        ------------------------------
        ModuleNotFoundError
            If graphviz is not installed.
        """
        try:
            import graphviz
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError(
                "In order to run the graph Dot visualization, "
                "the graphviz library must be installed. This "
                "library is not an explicit dependency of "
                "Embiggen because it may be hard to install "
                "on some systems and cause the Embiggen library "
                "to fail the installation.\n"
                "In order to install graphviz, try running "
                "`pip install graphviz`."
            ) from exception
        return graphviz.Source(self._graph.to_dot(), engine=engine)

    def _plot_node_metric_distribution(
        self,
        metric: np.ndarray,
        metric_name: str,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        show_title: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph node metric distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        show_title: bool = True
            Wether to show the figure title.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        assert isinstance(metric, np.ndarray), (
            "The metric must be a numpy array! " f"Got {type(metric)} instead."
        )
        assert metric.ndim == 1, (
            "The metric must be a 1D array! " f"Got {metric.ndim} dimensions instead."
        )
        assert isinstance(metric_name, str), (
            "The metric name must be a string! " f"Got {type(metric_name)} instead."
        )
        assert len(metric_name) > 0

        if axes is None:
            figure, axes = plt.subplots(figsize=(5, 5))
            figure.patch.set_facecolor("white")
        number_of_buckets = min(100, metric.size // 10)
        axes.hist(metric, bins=number_of_buckets, log=True)
        axes.set_ylabel("Count (log scale)")
        axes.set_xlabel(metric_name)
        if self._show_graph_name:
            title = f"{metric_name} distribution of graph {self._graph_name}"
        else:
            title = f"{metric_name} distribution"
        if show_title:
            axes.set_title(title)
        if apply_tight_layout:
            figure.tight_layout()

        if not return_caption:
            return self._handle_notebook_display(figure, axes)

        caption = (
            f"<i>{metric_name} distribution.</i> {metric_name} is on the "
            "horizontal axis and node counts are on the vertical axis on a logarithmic scale."
        )

        return self._handle_notebook_display(figure, axes, caption=caption)

    def plot_node_degree_distribution(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        show_title: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph node degree distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        show_title: bool = True
            Wether to show the figure title.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_node_metric_distribution(
            metric=self._support.get_non_zero_subgraph_node_degrees(self._graph),
            metric_name="Node degree",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            show_title=show_title,
            return_caption=return_caption,
        )

    def plot_triangle_distribution(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        show_title: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph triangle distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        show_title: bool = True
            Wether to show the figure title.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_node_metric_distribution(
            metric=self._support.get_number_of_triangles_per_node(),
            metric_name="Triangles",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            show_title=show_title,
            return_caption=return_caption,
        )

    def plot_square_distribution(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        show_title: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph square distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        show_title: bool = True
            Wether to show the figure title.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_node_metric_distribution(
            metric=self._support.get_number_of_squares_per_node(),
            metric_name="Squares",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            show_title=show_title,
            return_caption=return_caption,
        )

    def plot_approximated_harmonic_centrality_distribution(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        show_title: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph Approximated Harmonic Centrality distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        show_title: bool = True
            Wether to show the figure title.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_node_metric_distribution(
            metric=self._support.get_approximated_harmonic_centrality(),
            metric_name="Approx Harmonic Centrality",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            show_title=show_title,
            return_caption=return_caption,
        )

    def plot_approximated_closeness_centrality_distribution(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        show_title: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph Approximated Closeness Centrality distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        show_title: bool = True
            Wether to show the figure title.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        return self._plot_node_metric_distribution(
            metric=self._support.get_approximated_closeness_centrality(),
            metric_name="Approx Closeness Centrality",
            figure=figure,
            axes=axes,
            apply_tight_layout=apply_tight_layout,
            show_title=show_title,
            return_caption=return_caption,
        )

    def plot_edge_weight_distribution(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Figure] = None,
        apply_tight_layout: bool = True,
        return_caption: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph node degree distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_caption: bool = True,
            Whether to return a caption for the plot.
        """
        if axes is None:
            figure, axes = plt.subplots(figsize=(5, 5))
        number_of_buckets = min(100, self._graph.get_number_of_directed_edges() // 10)
        axes.hist(
            self._graph.get_directed_edge_weights(), bins=number_of_buckets, log=True
        )
        axes.set_ylabel("Number of edges (log scale)")
        axes.set_xlabel("Edge Weights")
        if self._show_graph_name:
            title = f"Weights distribution of graph {self._graph_name}"
        else:
            title = "Weights distribution"
        axes.set_title(title)
        if apply_tight_layout:
            figure.tight_layout()

        if not return_caption:
            return self._handle_notebook_display(figure, axes)

        caption = (
            "<i>Edge weights distribution.</i> Edge weights on the "
            "horizontal axis and edge counts on the vertical axis on a logarithmic scale."
        )

        return self._handle_notebook_display(figure, axes, caption=caption)

    def _fit_and_plot_all(
        self,
        points: List[np.ndarray],
        nrows: int,
        ncols: int,
        plotting_callbacks: List[Callable],
        show_letters: bool,
    ) -> Tuple[Figure, Axes]:
        """Fits and plots all available features of the graph.

        Parameters
        -------------------------
            Kwargs to be forwarded to the node embedding algorithm.
        """
        decompositions_backup = [
            self._node_decomposition,
            self._positive_edge_decomposition,
            self._negative_edge_decomposition,
        ]

        if self._currently_plotting_edge_embedding:
            (
                self._positive_edge_decomposition,
                self._negative_edge_decomposition,
            ) = points
        else:
            (
                self._node_decomposition,
                self._positive_edge_decomposition,
                self._negative_edge_decomposition,
            ) = points

        figure, axes = self._get_figure_and_axes(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), dpi=96
        )
        figure.patch.set_facecolor("white")
        number_of_total_plots = len(plotting_callbacks)

        flat_axes = np.array(axes).flatten()

        # Backing up ang off some of the visualizations
        # so we avoid duplicating their content.
        show_name_backup = self._show_graph_name
        show_node_embedding_backup = self._show_embedding_method
        show_edge_embedding_backup = self._show_edge_embedding_methods
        automatically_display_backup = self._automatically_display_on_notebooks
        show_separability_backup = self._show_separability_considerations_explanation
        show_heatmaps_backup = self._show_heatmaps_description
        non_existing_edges_sampling = self._show_non_existing_edges_sampling_description
        self._show_graph_name = False
        self._show_embedding_method = False
        self._show_edge_embedding_methods = False
        self._automatically_display_on_notebooks = False
        self._show_separability_considerations_explanation = False
        self._show_heatmaps_description = False
        self._show_non_existing_edges_sampling_description = False

        complete_caption = (
            f"<b>{self._decomposition_method} decomposition and properties distribution"
            f" of the {self._graph_name} graph using the {sanitize_ml_labels(self._embedding_method_name)} node embedding:</b>"
        )

        heatmaps_letters = []
        evaluation_letters = []

        for ax, plot_callback, letter in zip(
            flat_axes, itertools.chain(plotting_callbacks), "abcdefghjkilmnopqrstuvwxyz"
        ):
            figure, axes, caption = plot_callback(
                figure=figure,
                axes=ax,
                **(
                    dict(loc="lower center")
                    if "loc" in inspect.signature(plot_callback).parameters
                    else dict()
                ),
                apply_tight_layout=False,
            )

            if "heatmap" in caption.lower():
                heatmaps_letters.append(letter)
            if "accuracy" in caption.lower():
                evaluation_letters.append(letter)
            complete_caption += f" <b>({letter})</b> {caption}"

            if show_letters:
                if self._n_components >= 3:
                    additional_kwargs = dict(z=0, y=0, x=0)
                else:
                    additional_kwargs = dict(y=1.1, x=0)

                ax.text(
                    s=letter,
                    size=18,
                    color="black",
                    weight="bold",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    **additional_kwargs,
                )

        complete_caption += "<br>"

        self._show_edge_embedding_methods = show_edge_embedding_backup
        self._show_embedding_method = show_node_embedding_backup
        self._automatically_display_on_notebooks = automatically_display_backup
        self._show_separability_considerations_explanation = show_separability_backup
        self._show_heatmaps_description = show_heatmaps_backup
        self._show_non_existing_edges_sampling_description = non_existing_edges_sampling

        # If requested we automatically add the description of the heatmaps.
        complete_caption += self.get_heatmaps_comments(heatmaps_letters)
        # If requested we automatically add the description of these considerations.
        complete_caption += self.get_separability_comments_description(
            evaluation_letters
        )
        complete_caption += self.get_non_existing_edges_sampling_description()

        for axis in flat_axes[number_of_total_plots:]:
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.axis("off")

        if show_name_backup:
            figure.suptitle(
                self._get_complete_title(self._graph_name, show_edge_embedding=True),
                fontsize=20,
            )
            if self._n_components != 3:
                figure.tight_layout(rect=[0, 0.03, 1, 0.96])
        elif self._n_components != 3:
            figure.tight_layout()

        self._show_graph_name = show_name_backup
        (
            self._node_decomposition,
            self._positive_edge_decomposition,
            self._negative_edge_decomposition,
        ) = decompositions_backup

        return self._handle_notebook_display(
            figure, flat_axes, caption=complete_caption
        )

    def fit_and_plot_all(
        self,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                pd.DataFrame,
                np.ndarray,
                List[Union[pd.DataFrame, np.ndarray]],
            ]
        ] = None,
        number_of_columns: int = 4,
        show_letters: bool = True,
        include_distribution_plots: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Fits and plots all available features of the graph.

        Parameters
        -------------------------
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge type features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        number_of_columns: int = 4
            Number of columns to use for the layout.
        show_letters: bool = True
            Whether to show letters on the top left of the subplots.
        include_distribution_plots: bool = True
            Whether to include the distribution plots for the degrees
            and the edge weights, if they are present.
        """
        node_features = AbstractClassifierModel.normalize_node_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_features=node_features,
            allow_automatic_feature=True,
        )
        node_type_features = AbstractClassifierModel.normalize_node_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
        )
        edge_type_features = AbstractClassifierModel.normalize_edge_type_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            edge_type_features=edge_type_features,
            allow_automatic_feature=True,
        )
        edge_features = AbstractClassifierModel.normalize_edge_features(
            graph=self._support,
            support=self._support,
            random_state=self._random_state,
            edge_features=edge_features,
            allow_automatic_feature=True,
        )
        if len(node_features) + len(node_type_features) > 0:
            self.fit_nodes(
                node_features=node_features,
                node_type_features=node_type_features,
            )
        self.fit_negative_and_positive_edges(
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )

        node_scatter_plot_methods_to_call = []
        distribution_plot_methods_to_call = []

        if len(node_features) + len(node_type_features) > 0:
            node_scatter_plot_methods_to_call.extend(
                [
                    self.plot_node_degrees,
                    *(
                        (self.plot_node_triangles,)
                        if self._support.get_number_of_nodes() < 10_000_000
                        else ()
                    ),
                    *(
                        (self.plot_node_squares,)
                        if self._support.get_number_of_edges() < 1_000_000
                        else ()
                    ),
                    *(
                        (self.plot_approximated_closeness_centrality,)
                        if self._support.get_number_of_nodes() < 10_000_000
                        else ()
                    ),
                    *(
                        (self.plot_approximated_harmonic_centrality,)
                        if self._support.get_number_of_nodes() < 10_000_000
                        else ()
                    ),
                ]
            )

            distribution_plot_methods_to_call.extend(
                [
                    self.plot_node_degree_distribution,
                    *(
                        (self.plot_triangle_distribution,)
                        if self._support.get_number_of_nodes() < 10_000_000
                        else ()
                    ),
                    *(
                        (self.plot_square_distribution,)
                        if self._support.get_number_of_edges() < 1_000_000
                        else ()
                    ),
                    *(
                        (self.plot_approximated_closeness_centrality_distribution,)
                        if self._support.get_number_of_nodes() < 10_000_000
                        else ()
                    ),
                    *(
                        (self.plot_approximated_harmonic_centrality_distribution,)
                        if self._support.get_number_of_nodes() < 10_000_000
                        else ()
                    ),
                ]
            )

        def plot_distance_wrapper(plot_distance):
            @functools.wraps(plot_distance)
            def wrapped_plot_distance(**kwargs):
                return plot_distance(
                    node_features=node_features,
                    node_type_features=node_type_features,
                    **kwargs
                )

            return wrapped_plot_distance

        edge_scatter_plot_methods_to_call = [
            self.plot_positive_and_negative_edges,
            self.plot_positive_and_negative_edges_adamic_adar,
            self.plot_positive_and_negative_edges_jaccard_coefficient,
            self.plot_positive_and_negative_edges_preferential_attachment,
            self.plot_positive_and_negative_edges_resource_allocation_index,
        ]

        if len(node_features) + len(node_type_features) > 0:
            edge_scatter_plot_methods_to_call.extend([
                plot_distance_wrapper(
                    self.plot_positive_and_negative_edges_euclidean_distance
                ),
                plot_distance_wrapper(
                    self.plot_positive_and_negative_edges_cosine_similarity
                ),
            ])

        distribution_plot_methods_to_call.extend([
            self.plot_positive_and_negative_adamic_adar_histogram,
            self.plot_positive_and_negative_jaccard_coefficient_histogram,
            self.plot_positive_and_negative_preferential_attachment_histogram,
            self.plot_positive_and_negative_resource_allocation_index_histogram,
        ])

        if len(node_features) + len(node_type_features) > 0:
            distribution_plot_methods_to_call.extend([
                plot_distance_wrapper(
                    self.plot_positive_and_negative_edges_euclidean_distance_histogram
                ),
                plot_distance_wrapper(
                    self.plot_positive_and_negative_edges_cosine_similarity_histogram
                ),
            ])

        if (
            not self._currently_plotting_edge_embedding
            and self._graph.has_node_types()
            and not self._graph.has_homogeneous_node_types()
        ):
            node_scatter_plot_methods_to_call.append(self.plot_node_types)

        if (
            not self._currently_plotting_edge_embedding
            and self._graph.has_node_ontologies()
            and not self._graph.has_homogeneous_node_ontologies()
        ):
            node_scatter_plot_methods_to_call.append(self.plot_node_ontologies)

        if (
            not self._currently_plotting_edge_embedding
            and not self._support.is_connected()
            and not self._support.is_directed()
        ):
            node_scatter_plot_methods_to_call.append(self.plot_connected_components)

        if (
            self._positive_graph.has_edge_types()
            and not self._positive_graph.has_homogeneous_edge_types()
        ):
            edge_scatter_plot_methods_to_call.append(self.plot_edge_types)

        if (
            self._positive_graph.has_edge_weights()
            and not self._positive_graph.has_constant_edge_weights()
        ):
            edge_scatter_plot_methods_to_call.append(self.plot_edge_weights)
            distribution_plot_methods_to_call.append(self.plot_edge_weight_distribution)

        if not include_distribution_plots or self._rotate or self._n_components > 2:
            distribution_plot_methods_to_call = []

        plotting_callbacks = [
            callback
            for callbacks in (
                node_scatter_plot_methods_to_call,
                edge_scatter_plot_methods_to_call,
                distribution_plot_methods_to_call,
            )
            for callback in callbacks
        ]

        number_of_total_plots = len(plotting_callbacks)

        nrows = max(int(math.ceil(number_of_total_plots / number_of_columns)), 1)
        ncols = min(number_of_columns, number_of_total_plots)

        points = []

        if not self._currently_plotting_edge_embedding:
            points.append(self._node_decomposition)

        points.extend(
            [self._positive_edge_decomposition, self._negative_edge_decomposition]
        )

        if self._rotate:
            path = "fit_and_plot_all.webm"
            display_backup = self._automatically_display_on_notebooks
            self._automatically_display_on_notebooks = False
            rotate_backup = self._rotate
            self._rotate = False
            rotate(
                self._fit_and_plot_all,
                path=path,
                points=points,
                duration=self._duration,
                fps=self._fps,
                verbose=self._verbose,
                nrows=nrows,
                ncols=ncols,
                plotting_callbacks=plotting_callbacks,
                show_letters=show_letters,
            )
            to_display = display_video_at_path(path, width="100%", height=None)
            figure, axes, complete_caption = self._fit_and_plot_all(
                points=points,
                nrows=nrows,
                ncols=ncols,
                plotting_callbacks=plotting_callbacks,
                show_letters=show_letters,
            )
            self._rotate = rotate_backup
            self._automatically_display_on_notebooks = display_backup
            return self._handle_notebook_display(
                to_display, None, caption=complete_caption
            )
        return self._fit_and_plot_all(
            points=points,
            nrows=nrows,
            ncols=ncols,
            plotting_callbacks=plotting_callbacks,
            show_letters=show_letters,
        )

