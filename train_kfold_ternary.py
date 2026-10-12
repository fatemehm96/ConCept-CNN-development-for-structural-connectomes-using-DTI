"""
K-Fold Cross-Validation Training Pipeline -- TERNARY (FTD / AD / CN)
------------------------------------------------------------------------
Same corrected methodology as train_kfold_regularized.py (binary version):
  - Split test fold FIRST, then train/val, then scale (fit on train only),
    then SMOTE (train/val only, never test)
  - GlobalAveragePooling2D instead of Flatten (reduces dense1 from ~277M
    to ~65K parameters)
  - Dropout 0.5 + L2 regularization on dense layers
  - 5-fold stratified cross-validation

Differences for the ternary case:
  - Output layer: 3 units + softmax (instead of 1 unit + sigmoid)
  - Loss: categorical_crossentropy (labels one-hot encoded via LabelBinarizer,
    which produces a 3-column one-hot matrix automatically for 3+ classes)
  - Metrics: macro-averaged precision/recall/F1 (treats all 3 classes equally
    regardless of class size), one-vs-rest AUC (macro-averaged), and
    per-class sensitivity/specificity derived from the 3x3 confusion matrix

Labels expected in the raw .npy file (set when you built it):
  CN = 0, FTD = 1, AD = 2  (matches TERNARY_LABELS in build_npy_from_csv.py)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from keras.layers import (Dense, Dropout, BatchNormalization, Activation, Input,
                           Conv2D, GlobalAveragePooling2D, AveragePooling2D, concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "ternary_FA.npy"
N_ROWS = 83
N_COLS = 84
N_CLASSES = 3
CLASS_NAMES = ["CN", "FTD", "AD"]   # index order must match labels 0, 1, 2

N_FOLDS = 5
RANDOM_STATE = 42
VAL_SIZE_WITHIN_FOLD = 0.15

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.2
L2_REG = 1e-3

OUTPUT_PREFIX = "ternary_kfold"

# ============================================================
# STEP 1: Load raw data
# ============================================================

raw = np.load(DATA_FILE)
print("Raw data shape:", raw.shape)

y_all = raw[:, 0]
X_all = raw[:, 1:]
print("X shape:", X_all.shape, "| y shape:", y_all.shape)
print("Class distribution:", pd.Series(y_all).value_counts().to_dict())

# ============================================================
# STEP 2: Regularized model architecture (ternary output)
# ============================================================

def build_fc_regularized_ternary(input_shape=(83, 84, 1), n_classes=3,
                                  dropout_rate=0.2, l2_reg=1e-3):
    data = Input(shape=input_shape)

    coarse_channel = Conv2D(128, kernel_size=(1, 3), padding='valid', activation='relu', name='coarse_conv1')(data)
    coarse_channel = AveragePooling2D((1, 3), strides=(1, 2), name='coarse_AVG_pool1')(coarse_channel)
    coarse_channel = Conv2D(256, kernel_size=(1, 3), padding='valid', activation='relu', name='coarse_conv2')(coarse_channel)
    coarse_channel = AveragePooling2D((1, 3), strides=(1, 2), name='coarse_AVG_pool2')(coarse_channel)

    medium_channel = Conv2D(128, kernel_size=(1, 5), padding='valid', activation='relu', name='medium_conv1')(data)
    medium_channel = AveragePooling2D((1, 3), strides=(1, 2), name='medium_AVG_pool1')(medium_channel)
    medium_channel = Conv2D(256, kernel_size=(1, 5), padding='valid', activation='relu', name='medium_conv2')(medium_channel)
    medium_channel = AveragePooling2D((1, 3), strides=(1, 2), name='medium_AVG_pool2')(medium_channel)

    fine_channel = Conv2D(128, kernel_size=(1, 7), padding='valid', activation='relu', name='fine_conv1')(data)
    fine_channel = AveragePooling2D((1, 3), strides=(1, 2), name='fine_AVG_pool1')(fine_channel)
    fine_channel = Conv2D(256, kernel_size=(1, 7), padding='valid', activation='relu', name='fine_conv2')(fine_channel)
    fine_channel = AveragePooling2D((1, 3), strides=(1, 2), name='fine_AVG_pool2')(fine_channel)

    global_channel = Conv2D(256, kernel_size=(1, 84), padding='valid', activation='relu', name='global_conv1')(data)

    img_feat = concatenate([coarse_channel, medium_channel, fine_channel, global_channel], axis=2)
    img_feat = GlobalAveragePooling2D()(img_feat)

    img_feat = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg), name='dense1')(img_feat)
    img_feat = Dropout(dropout_rate)(img_feat)
    img_feat = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg), name='dense2')(img_feat)
    img_feat = Dropout(dropout_rate)(img_feat)
    img_feat = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg), name='dense3')(img_feat)
    img_feat = Dropout(dropout_rate)(img_feat)
    img_feat = BatchNormalization()(img_feat)

    # Ternary output: n_classes units + softmax (instead of 1 unit + sigmoid)
    out = Dense(n_classes, use_bias=False)(img_feat)
    out = BatchNormalization()(out)
    out = Activation('softmax', name='prediction_layer')(out)

    model = Model(inputs=data, outputs=out, name="Multi-filter-CNN-Regularized-Ternary")
    return model


sample_model = build_fc_regularized_ternary(input_shape=(N_ROWS, N_COLS, 1), n_classes=N_CLASSES,
                                             dropout_rate=DROPOUT_RATE, l2_reg=L2_REG)
sample_model.summary()
del sample_model
tf.keras.backend.clear_session()   # del alone doesn't free GPU memory -- this does

# ============================================================
# STEP 3: K-Fold Cross-Validation
# ============================================================

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
all_true = []
all_pred = []
all_prob = []   # will store the full (n_samples, n_classes) probability array

for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
    print(f"\n{'='*60}\nFOLD {fold_idx}/{N_FOLDS}\n{'='*60}")

    X_trainval, X_test = X_all[trainval_idx], X_all[test_idx]
    y_trainval, y_test = y_all[trainval_idx], y_all[test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_WITHIN_FOLD,
        random_state=RANDOM_STATE, stratify=y_trainval
    )

    scaler = MinMaxScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train))
    X_val_scaled = np.nan_to_num(scaler.transform(X_val))
    X_test_scaled = np.nan_to_num(scaler.transform(X_test))

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_aug, y_train_aug = sm.fit_resample(X_train_scaled, y_train)
    X_val_aug, y_val_aug = sm.fit_resample(X_val_scaled, y_val)

    X_train_final = X_train_aug.reshape(-1, N_ROWS, N_COLS, 1)
    X_val_final = X_val_aug.reshape(-1, N_ROWS, N_COLS, 1)
    X_test_final = X_test_scaled.reshape(-1, N_ROWS, N_COLS, 1)

    # LabelBinarizer one-hot encodes automatically for 3+ classes.
    lb = LabelBinarizer()
    y_train_final = lb.fit_transform(y_train_aug)
    y_val_final = lb.transform(y_val_aug)
    y_test_final = lb.transform(y_test)

    print(f"Train: {X_train_final.shape} | Val: {X_val_final.shape} | Test: {X_test_final.shape}")

    model = build_fc_regularized_ternary(input_shape=(N_ROWS, N_COLS, 1), n_classes=N_CLASSES,
                                          dropout_rate=DROPOUT_RATE, l2_reg=L2_REG)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )

    # Save this fold's per-epoch accuracy/loss so we can plot training curves later.
    history_df = pd.DataFrame(history.history)
    history_df["fold"] = fold_idx
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df.to_csv(f"{OUTPUT_PREFIX}_fold{fold_idx}_history.csv", index=False)
    

    y_pred_prob = model.predict(X_test_final)          # shape (n_test, 3)
    y_pred = np.argmax(y_pred_prob, axis=1)             # class index 0/1/2
    y_true = np.argmax(y_test_final, axis=1)

    fold_acc = accuracy_score(y_true, y_pred)
    fold_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    fold_sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    fold_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        fold_auc = roc_auc_score(y_test_final, y_pred_prob, average='macro', multi_class='ovr')
    except ValueError:
        fold_auc = np.nan

    # Per-class specificity from the 3x3 confusion matrix:
    # for each class c, specificity = TN / (TN + FP), treating c as "positive"
    # and all other classes as "negative".
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    specificities = []
    for c in range(N_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        specificities.append(spec)
    fold_specificity_macro = np.nanmean(specificities)

    print(f"Fold {fold_idx} -- Accuracy: {fold_acc:.4f} | Sensitivity(macro): {fold_sensitivity:.4f} | "
          f"Specificity(macro): {fold_specificity_macro:.4f} | Precision(macro): {fold_precision:.4f} | "
          f"F1(macro): {fold_f1:.4f} | AUC(macro,ovr): {fold_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order CN/FTD/AD):")
    print(cm)

    fold_results.append({
        "fold": fold_idx,
        "accuracy": fold_acc,
        "sensitivity_macro": fold_sensitivity,
        "specificity_macro": fold_specificity_macro,
        "precision_macro": fold_precision,
        "f1_macro": fold_f1,
        "auc_macro_ovr": fold_auc,
        "n_test": len(y_true)
    })

    all_true.extend(y_true.tolist())
    all_pred.extend(y_pred.tolist())
    all_prob.append(y_pred_prob)

    tf.keras.backend.clear_session()

# ============================================================
# STEP 4: Aggregate results across all folds
# ============================================================

results_df = pd.DataFrame(fold_results)
print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY ({N_FOLDS}-fold, TERNARY)\n{'='*60}")
print(results_df.to_string(index=False))
print(f"\nMean accuracy:            {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
print(f"Mean sensitivity (macro): {results_df['sensitivity_macro'].mean():.4f} +/- {results_df['sensitivity_macro'].std():.4f}")
print(f"Mean specificity (macro):{results_df['specificity_macro'].mean():.4f} +/- {results_df['specificity_macro'].std():.4f}")
print(f"Mean precision (macro):  {results_df['precision_macro'].mean():.4f} +/- {results_df['precision_macro'].std():.4f}")
print(f"Mean F1 (macro):         {results_df['f1_macro'].mean():.4f} +/- {results_df['f1_macro'].std():.4f}")
print(f"Mean AUC (macro, ovr):   {results_df['auc_macro_ovr'].mean():.4f} +/- {results_df['auc_macro_ovr'].std():.4f}")

overall_cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
print(f"\nPooled confusion matrix (every subject, one prediction each), order {CLASS_NAMES}:")
print(overall_cm)

results_df.to_csv(f"{OUTPUT_PREFIX}_fold_results.csv", index=False)

pooled_df = pd.DataFrame({"true_label": all_true, "predicted_label": all_pred})
pooled_df.to_csv(f"{OUTPUT_PREFIX}_pooled_predictions.csv", index=False)

print(f"\nSaved per-fold results to {OUTPUT_PREFIX}_fold_results.csv")
print(f"Saved pooled predictions to {OUTPUT_PREFIX}_pooled_predictions.csv")
