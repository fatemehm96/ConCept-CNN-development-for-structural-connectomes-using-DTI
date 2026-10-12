"""
K-Fold Cross-Validation Training Pipeline (Regularized ConCeptCNN)
----------------------------------------------------------------------
Changes from the original single-split pipeline (train_full_pipeline.py):

1. GlobalAveragePooling2D replaces Flatten before dense1.
   This cuts dense1 from ~277M parameters down to ~65K, since the
   83 x 51 x 256 feature map is averaged down to 256 values instead
   of being fully flattened. This is a genuine architecture change
   from the original ConCeptCNN in myModel.py -- report it as a
   revised/regularized variant, not as the original model, if you
   use these results.

2. Dropout increased from 0.2 to 0.5, and L2 weight regularization
   added to dense1 (the layer with the most capacity).

3. K-fold cross-validation (default 5-fold, stratified) replaces the
   single train/test split. Each subject is used in the held-out
   test fold exactly once across all folds, giving a far more stable
   accuracy estimate than one 36-subject split -- important given
   how small this dataset is.

IMPORTANT ON METHODOLOGY:
   Within each fold: the fold's own test portion is set aside FIRST,
   then the remaining data is split into train/val, THEN SMOTE is
   applied only to train (and optionally val). The test fold is
   never touched by SMOTE or by the scaler's fit() -- this preserves
   the leak-free order established in the single-split version.

   Do NOT run this many times and report whichever fold split or
   configuration gives the best mean accuracy -- pick one configuration,
   run it once, and report the result honestly, whatever it is.
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

DATA_FILE = "FTD_CN_FA.npy"     # change per class-pair
N_ROWS = 83
N_COLS = 84

N_FOLDS = 5             # 5-fold or 10-fold are both reasonable given 240 subjects
RANDOM_STATE = 42

VAL_SIZE_WITHIN_FOLD = 0.15   # validation carved out of each fold's training portion

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4    # much lower than before -- GAP made the model far smaller,
                         # so the old 1e-2 (tuned for a 277M-param dense1) was too
                         # aggressive and caused training to collapse
DROPOUT_RATE = 0.5
L2_REG = 1e-3

OUTPUT_PREFIX = "FTD_CN_kfold"

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
# STEP 2: Regularized model architecture
# ============================================================

def build_fc_regularized(input_shape=(83, 84, 1), dropout_rate=0.5, l2_reg=1e-3):
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

    # GlobalAveragePooling2D replaces Flatten -- collapses the (83, 51, 256)
    # feature map to a 256-length vector instead of flattening to 1,083,648.
    img_feat = GlobalAveragePooling2D()(img_feat)

    img_feat = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg), name='dense1')(img_feat)
    img_feat = Dropout(dropout_rate)(img_feat)
    img_feat = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg), name='dense2')(img_feat)
    img_feat = Dropout(dropout_rate)(img_feat)
    img_feat = Dense(256, use_bias=False, kernel_regularizer=l2(l2_reg), name='dense3')(img_feat)
    img_feat = Dropout(dropout_rate)(img_feat)
    img_feat = BatchNormalization()(img_feat)
    out = Dense(1, use_bias=False)(img_feat)
    out = BatchNormalization()(out)
    out = Activation('sigmoid', name='prediction_layer')(out)

    model = Model(inputs=data, outputs=out, name="Multi-filter-CNN-Regularized")
    return model


# Print the architecture once, before the fold loop, just to confirm param count
print("Model architecture will be printed when fold 1 builds its model.")

# ============================================================
# STEP 3: K-Fold Cross-Validation
# ============================================================

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
all_true = []
all_pred = []
all_prob = []

for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
    print(f"\n{'='*60}\nFOLD {fold_idx}/{N_FOLDS}\n{'='*60}")

    X_trainval, X_test = X_all[trainval_idx], X_all[test_idx]
    y_trainval, y_test = y_all[trainval_idx], y_all[test_idx]

    # Split the fold's train+val pool further into train and val.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_WITHIN_FOLD,
        random_state=RANDOM_STATE, stratify=y_trainval
    )

    # Scale -- fit on train only.
    scaler = MinMaxScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train))
    X_val_scaled = np.nan_to_num(scaler.transform(X_val))
    X_test_scaled = np.nan_to_num(scaler.transform(X_test))

    # SMOTE -- train and val only. Test fold is NEVER augmented.
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_aug, y_train_aug = sm.fit_resample(X_train_scaled, y_train)
    X_val_aug, y_val_aug = sm.fit_resample(X_val_scaled, y_val)

    # Reshape for model input.
    X_train_final = X_train_aug.reshape(-1, N_ROWS, N_COLS, 1)
    X_val_final = X_val_aug.reshape(-1, N_ROWS, N_COLS, 1)
    X_test_final = X_test_scaled.reshape(-1, N_ROWS, N_COLS, 1)

    lb = LabelBinarizer()
    y_train_final = lb.fit_transform(y_train_aug).ravel()
    y_val_final = lb.transform(y_val_aug).ravel()
    y_test_final = lb.transform(y_test).ravel()

    print(f"Train: {X_train_final.shape} | Val: {X_val_final.shape} | Test: {X_test_final.shape}")

    # Fresh model each fold -- no weight leakage between folds.
    model = build_fc_regularized(input_shape=(N_ROWS, N_COLS, 1),
                                  dropout_rate=DROPOUT_RATE, l2_reg=L2_REG)
    if fold_idx == 1:
        model.summary()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
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

    # Evaluate on this fold's held-out, unaugmented test portion.
    y_pred_prob = model.predict(X_test_final).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    fold_acc = accuracy_score(y_test_final, y_pred)
    fold_f1 = f1_score(y_test_final, y_pred)
    fold_precision = precision_score(y_test_final, y_pred, zero_division=0)
    fold_sensitivity = recall_score(y_test_final, y_pred, zero_division=0)  # = recall = TPR

    # Specificity = TNR = TN / (TN + FP) -- not built into sklearn directly,
    # so derive it from the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred, labels=[0, 1]).ravel()
    fold_specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    try:
        fold_auc = roc_auc_score(y_test_final, y_pred_prob)
    except ValueError:
        fold_auc = np.nan

    print(f"Fold {fold_idx} -- Accuracy: {fold_acc:.4f} | Sensitivity: {fold_sensitivity:.4f} | "
          f"Specificity: {fold_specificity:.4f} | Precision: {fold_precision:.4f} | "
          f"F1: {fold_f1:.4f} | AUC: {fold_auc:.4f}")

    fold_results.append({
        "fold": fold_idx,
        "accuracy": fold_acc,
        "sensitivity": fold_sensitivity,
        "specificity": fold_specificity,
        "precision": fold_precision,
        "f1": fold_f1,
        "auc": fold_auc,
        "n_test": len(y_test_final)
    })

    all_true.extend(y_test_final.tolist())
    all_pred.extend(y_pred.tolist())
    all_prob.extend(y_pred_prob.tolist())

    tf.keras.backend.clear_session()  # free memory between folds

# ============================================================
# STEP 4: Aggregate results across all folds
# ============================================================

results_df = pd.DataFrame(fold_results)
print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY ({N_FOLDS}-fold)\n{'='*60}")
print(results_df.to_string(index=False))
print(f"\nMean accuracy:    {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
print(f"Mean sensitivity: {results_df['sensitivity'].mean():.4f} +/- {results_df['sensitivity'].std():.4f}")
print(f"Mean specificity: {results_df['specificity'].mean():.4f} +/- {results_df['specificity'].std():.4f}")
print(f"Mean precision:   {results_df['precision'].mean():.4f} +/- {results_df['precision'].std():.4f}")
print(f"Mean F1:          {results_df['f1'].mean():.4f} +/- {results_df['f1'].std():.4f}")
print(f"Mean AUC:         {results_df['auc'].mean():.4f} +/- {results_df['auc'].std():.4f}")

# Overall confusion matrix pooling every subject's held-out prediction
# (each subject appears in exactly one test fold, so this covers the whole dataset)
overall_cm = confusion_matrix(all_true, all_pred)
print("\nPooled confusion matrix (every subject, one prediction each):")
print(overall_cm)

results_df.to_csv(f"{OUTPUT_PREFIX}_fold_results.csv", index=False)

pooled_df = pd.DataFrame({"true_label": all_true, "predicted_label": all_pred, "predicted_probability": all_prob})
pooled_df.to_csv(f"{OUTPUT_PREFIX}_pooled_predictions.csv", index=False)

print(f"\nSaved per-fold results to {OUTPUT_PREFIX}_fold_results.csv")
print(f"Saved pooled predictions to {OUTPUT_PREFIX}_pooled_predictions.csv")
