from cvc import utils as ut
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

DEVICE = ut.get_device(0)
# GCP paths
HOME_DIR_GCP = "/home/romi/projects/preTCR"
DATA_DIR = HOME_DIR_GCP + "/CDR3_data/"
# model trained on 5 million sequences - 2.5 million private and 2.5 million public
TRANSFORMER = HOME_DIR_GCP + "/output_5mil_even_priv_pub"
# TRANSFORMER = HOME_DIR_GCP + "/output_dir_91m_v22_91m"
# TRANSFORMER = HOME_DIR_GCP + "/output_dir_v23_17m_8gpu_batch256"

# classification algorithms
# xgBoost
def xgb_classify(train_embeddings, embed_train_labels_num, validation_embeddings, embed_val_labels_num):
    # Initialize classifier
    xgb_classifier = xgb.XGBClassifier(n_jobs=-1)
    # Fit
    xgb_classifier.fit(train_embeddings, embed_train_labels_num)
    # Predict
    preds = xgb_classifier.predict(validation_embeddings)
    # Score
    acc_score = accuracy_score(embed_val_labels_num, preds)
    return acc_score*100, preds, xgb_classifier

# LDA
def lda_classify(train_embeddings, embed_train_labels_num, validation_embeddings, embed_val_labels_num):
    # Initialize classifier
    lda_classifier = LDA(n_components=1)
    # Fit
    lda_classifier.fit(train_embeddings, embed_train_labels_num)
    # Predict
    preds = lda_classifier.predict(validation_embeddings)
    # Score
    acc_score = accuracy_score(embed_val_labels_num, preds)
    return acc_score*100, preds, lda_classifier