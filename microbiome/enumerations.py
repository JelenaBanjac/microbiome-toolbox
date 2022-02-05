from enum import Enum

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap import UMAP


class ReferenceGroup(Enum):
    USER_DEFINED = 1
    NOVELTY_DETECTION = 2


class TimeUnit(Enum):
    HOUR = 1/24
    DAY = 1
    MONTH = 30
    YEAR = 365


class FeatureColumnsType(Enum):
    BACTERIA = 1
    METADATA = 2
    BACTERIA_AND_METADATA = 3


class AnomalyType(Enum):
    PREDICTION_INTERVAL = 1
    LOW_PASS_FILTER = 2
    ISOLATION_FOREST = 3


class FeatureExtraction(Enum):
    NONE = 0
    NEAR_ZERO_VARIANCE = 1
    CORRELATION = 2
    TOP_K_IMPORTANT = 3


class Normalization(Enum):
    NORMALIZED = True
    NON_NORMALIZED = False


class EmbeddingModelType(Enum):
    PCA = PCA
    TSNE = TSNE
    UMAP = UMAP
    ISOMAP = Isomap
