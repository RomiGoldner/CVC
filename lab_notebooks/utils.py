from cvc import utils as ut
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier

DEVICE = ut.get_device(0)

# Get path of the project root
def get_project_root():
    from pathlib import Path
    return str(Path(__file__).parent.parent)

# GCP paths
HOME_DIR_GCP = get_project_root()
# DATA_DIR = HOME_DIR_GCP + "/CDR3_data/"
DATA_DIR = '/home/dsi/rgoldner/CDR3_data/'
# model trained on 5 million sequences - 2.5 million private and 2.5 million public
TRANSFORMER = HOME_DIR_GCP + "/output_5mil_even_priv_pub"
# model trained on 2.2 million single cells (represented with concatenated CDR3 sequences)
SC_TRANSFORMER = HOME_DIR_GCP + "/output_dir_singlecell_v2"

# classification algorithms
# xgBoost
def xgb_classify(train_embeddings, embed_train_labels_num, validation_embeddings, embed_val_labels_num, seed=None):
    # Initialize classifier
    xgb_classifier = xgb.XGBClassifier(n_jobs=-1, random_state=seed)
    # Fit
    xgb_classifier.fit(train_embeddings, embed_train_labels_num)
    # Predict
    preds = xgb_classifier.predict(validation_embeddings)
    # Score
    acc_score = accuracy_score(embed_val_labels_num, preds)
    # convert to float
    acc_score = float(acc_score)
    return acc_score*100, preds, xgb_classifier

# LDA
def lda_classify(train_embeddings, embed_train_labels_num, validation_embeddings, embed_val_labels_num, seed=None):
    # Initialize classifier
    lda_classifier = LDA(n_components=1)
    # Fit
    lda_classifier.fit(train_embeddings, embed_train_labels_num)
    # Predict
    preds = lda_classifier.predict(validation_embeddings)
    # Score
    acc_score = accuracy_score(embed_val_labels_num, preds)
    return acc_score*100, preds, lda_classifier


# Random Forest
def rf_classify(train_embeddings, embed_train_labels_num, validation_embeddings, embed_val_labels_num, seed=None,
                n_estimators=100, max_depth=6):
    # Initialize classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    # Fit
    rf_classifier.fit(train_embeddings, embed_train_labels_num)
    # Predict
    preds = rf_classifier.predict(validation_embeddings)
    # Score
    acc_score = accuracy_score(embed_val_labels_num, preds)
    return acc_score*100, preds, rf_classifier