import numpy as np
import scipy as sp
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from mne.io import read_raw_ant
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import combined_analysis_bachelor.code.functions_for_pipeline as funcs


# 3. movement detection itself (choosing method?, threshold param, activity length param -> output = on/offsets
 # --> for EMG and ACC seperated or together always?


## removing off and onsets that are too close together (splitting one arm raise)
def old_take_out_short_off_onset(onsets, offsets, min_time_period, sampling_freq):
    onsets_clean = onsets.copy()
    offsets_clean = offsets.copy()
    min_sample_period = min_time_period * sampling_freq

    if hasattr(onsets, 'tolist'):
        onsets_clean = onsets.tolist()
        offsets_clean = offsets.tolist()

    # Iterate in reverse to safely delete items
    for i in range(len(offsets) - 2, -1, -1):  # Start from the end
        time_between = onsets[i + 1] - offsets[i]
        if time_between <= min_sample_period:
            del onsets_clean[i + 1]
            del offsets_clean[i]

    return onsets_clean, offsets_clean

def take_out_short_off_onset(onsets, offsets, min_time_period, sampling_freq):
    """
    Entfernt kurze Pausen zwischen Aktivitätsblöcken.
    Wenn der Abstand zwischen offset[i] und onset[i+1] <= min_time_period ist,
    werden offset[i] und onset[i+1] entfernt (d. h. die beiden Blöcke werden zusammengelegt).
    """
    # In Listen umwandeln (Kopien!)
    if hasattr(onsets, 'tolist'):
        onsets_clean = onsets.tolist()
    else:
        onsets_clean = list(onsets)
    if hasattr(offsets, 'tolist'):
        offsets_clean = offsets.tolist()
    else:
        offsets_clean = list(offsets)

    # Nichts zu tun?
    if len(onsets_clean) == 0 or len(offsets_clean) == 0:
        return onsets_clean, offsets_clean

    # Schutz: Falls ein Offset vor dem ersten Onset liegt → verwerfen
    # (kann passieren, wenn das Signal initial über Threshold startet)
    while len(offsets_clean) > 0 and len(onsets_clean) > 0 and offsets_clean[0] < onsets_clean[0]:
        del offsets_clean[0]
        if len(offsets_clean) == 0:
            return onsets_clean, offsets_clean

    min_sample_period = float(min_time_period) * float(sampling_freq)

    # Wir benötigen Paare (offset[i], onset[i+1]) → obere Grenze begrenzen
    # i darf höchstens len(onsets_clean)-2 sein; und natürlich < len(offsets_clean)
    n_pairs = min(len(offsets_clean), max(0, len(onsets_clean) - 1))

    # Rückwärts iterieren, damit Del sicher ist
    for i in range(n_pairs - 1, -1, -1):
        # Sicherstellen, dass die Indizes noch konsistent sind
        if i >= len(offsets_clean) or (i + 1) >= len(onsets_clean):
            continue
        time_between = onsets_clean[i + 1] - offsets_clean[i]
        if time_between <= min_sample_period:
            # Blöcke zusammenlegen: onset[i+1] und offset[i] entfernen
            del onsets_clean[i + 1]
            del offsets_clean[i]

    return onsets_clean, offsets_clean



# get array of final on- and onsets pairs #
def new_on_offsets(new_onsets, new_offsets, end_index=None):
    """Paare aus On- und Offsets bilden, fehlende ergänzen oder abschneiden."""
    onsets_and_offsets = []
    n_on = len(new_onsets)
    n_off = len(new_offsets)

    if n_on > n_off and end_index is not None:
        new_offsets = list(new_offsets) + [end_index] * (n_on - n_off)
    elif n_off > n_on:
        new_offsets = new_offsets[:n_on]

    for onset, offset in zip(new_onsets, new_offsets):
        onsets_and_offsets.append([onset, offset])
    return onsets_and_offsets



## create binary array ##
def fill_activity_mask(on_and_offsets, sampling_freq, time_column):
    """takes in final on- and offsets and creates binary array where the periods between on- and offset are set to True. Time points outside of these periods are False
    args
    on_and_offsets: list that holds pairs of on- and onsets
    sampling_freq: sampling frequency
    time_column: time column (Sync_Time) of the recording

    returns binary (boolean) array"""

    zeros = np.zeros(len(time_column))
    mask = np.zeros_like(zeros, dtype=bool)  # all to False
    for onset, offset in on_and_offsets:
        start = int(round(onset))
        end = int(round(offset))
        mask[max(0, start):min(len(mask), end + 1)] = True # sicher, dass hier end+1 hin muss?
    return mask


def create_behavioral_array(emg_activities_binary, acc_activities_binary):
    """compares the EMG and ACC binary arrays and outputs
    a number for each detected activity state"""
    behavioral_list = []
    for i, active in enumerate(emg_activities_binary):
        if active == False and acc_activities_binary[i] == False:
            behavioral_list.append(0) # = rest
        elif active == True and acc_activities_binary[i] == True:
            behavioral_list.append(1) # = move
        elif active == True and acc_activities_binary[i] == False:
            behavioral_list.append(2) # = suppression
        elif active == False and acc_activities_binary[i] == True:
            behavioral_list.append(1) # = gets also labelled as move for now
    behavioral_array = np.array(behavioral_list)

    return behavioral_array


def create_behavioral_array_arm(delt_activities_binary, brachioradialis_activities_binary, acc_activities_binary):
    behavioral = []
    for i, active in enumerate(acc_activities_binary):
        if active == False and delt_activities_binary[i] == False and brachioradialis_activities_binary[i] == False:
            behavioral.append(0) # = rest
        elif active == True and delt_activities_binary[i] == True and brachioradialis_activities_binary[i] == True:
            behavioral.append(1) # options for move
        elif active == True and delt_activities_binary[i] == True and brachioradialis_activities_binary[i] == False:
            behavioral.append(1)
        elif active == True and delt_activities_binary[i] == False and brachioradialis_activities_binary[i] == True:
            behavioral.append(1)
        elif active == False and delt_activities_binary[i] == True and brachioradialis_activities_binary[i] == True:
            behavioral.append(2) # options for suppression
        elif active == False and delt_activities_binary[i] == True and brachioradialis_activities_binary[i] == False:
            behavioral.append(2)
        elif active == False and delt_activities_binary[i] == False and brachioradialis_activities_binary[i] == True:
            behavioral.append(2)

    behavioral_array = np.array(behavioral)
    return behavioral_array













# ============================== machine learning part ===================================#

# --- splitting classification data --- #
def split_recording_by_manual_segments(
        filepath: str,
        output_dir: str,
        segments_dict: dict,  # e.g., {"rest1": [start, end], ...}
        channels: list,
        EMG_cols: list,
        location_dict: dict = None,
        plot_channel: str = None
):
    # Load CNT file
    raw = read_raw_ant(filepath, preload=True)
    times = raw.times
    channels = raw.ch_names
    basename = os.path.splitext(os.path.basename(filepath))[0]
    source_filename = os.path.basename(filepath)

    # === Metadata collection ===
    metadata = []

    for i, (label, bounds) in enumerate(segments_dict.items()):
        if len(bounds) != 2:
            print(f"❌ Segment '{label}' does not have [start, end] – skipping.")
            continue

        if label.lower() == "fail":
            print(f"⚠️  Segment {i + 1} ('fail') skipped.")
            continue

        start, end = bounds
        if end <= start:
            print(f"❌ Invalid bounds for segment '{label}' (start >= end) – skipping.")
            continue

        # Crop and get data
        segment = raw.copy().crop(tmin=start, tmax=end)
        data, seg_times = segment[channels, :]
        seg_ch_names = segment.ch_names

        # channel renaming #
        col_names = [location_dict.get(ch, "drop") for ch in seg_ch_names]

        df = pd.DataFrame(data.T, columns=col_names)
        df = df[[col for col in df.columns if not col.startswith("drop")]]
        df["time_sec"] = seg_times

        for col in df.columns:
            if col in EMG_cols:
                df[col] *= 1e6

        # File output
        safe_label = label.replace(" ", "_").lower()
        filename = f"{basename}_{safe_label}.h5"
        outpath = os.path.join(output_dir, filename)
        df.to_hdf(outpath, key="data", mode="w")

        # Save metadata for this segment
        metadata.append({
            "label": label,
            "filename": filename,
            "start_sec": float(start),
            "end_sec": float(end),
            "duration_sec": float(end - start),
            "source_file": source_filename
        })

    # === Save metadata to JSON ===
    metadata_path = output_dir / "segments_metadata.json"
    print(metadata_path)
    # metadata_path.mkdir(exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Files seperated. H5 and Metadata JSON saved!")


def create_train_df(filepaths, window_size, overlap_value, sf, output_dir):
    df_list = []

    for filepath in filepaths:
        # =========== processing of files ===========#
        df = pd.read_hdf(filepath, key="data")
        df = funcs.crop_edges(df)

        # --- renaming emg cols ---#
        emg_muscles = ["brachioradialis", "deltoideus"]
        emg_cols_raw = [col for col in df.columns if any(m in col for m in emg_muscles)]
        rename_map = {col: re.sub(r"(_L|_R)$", "", col) for col in emg_cols_raw}
        df = df.rename(columns=rename_map)

        # --- getting col names ---#
        EMG_cols = [col for col in df.columns if any(m in col for m in emg_muscles)]
        ACC_cols = ["SVM"]

        # --- calculating envelope & tkeo and adding them to df and also smoothing acc data ---#
        df = funcs.add_tkeo_add_envelope(df, EMG_cols, 20, 3, 1000)
        df = funcs.apply_savgol_rms_acc(df, ACC_cols, savgol_window_length=21, savgol_polyorder=3, rms_window_size=100)
        df = df.drop(ACC_cols, axis=1)

        df = df.dropna(axis=0, how="any", ignore_index=True)  # dropping NaNs
        df["time_sec"] = df["time_sec"] - df["time_sec"].iloc[0]  # resetting time column to start at 0s
        EMG_cols = [col for col in df.columns if any(m in col for m in emg_muscles)]  # getting out final emg cols
        ACC_cols = ["SVM_smooth_rms"]  # final acc cols

        # ========================== extracting features and creating dfs ==============================#
        feature_rows = []

        step_size = window_size * overlap_value
        for window_start in range(0, len(df) - window_size + 1, step_size):  # with 50% overlap
            end = window_start + window_size
            window = df.iloc[window_start:end]

            feats = funcs.extract_features_from_window(window, EMG_cols, ACC_cols)

            start_ms = int((window_start / sf) * 1000)
            end_ms = int((end / sf) * 1000)
            feats["window_range_ms"] = f"{start_ms}-{end_ms}ms"

            # subject-ID
            filename = os.path.basename(filepath)
            split = filename.split("_")
            sub_ID = f"{split[0]}-{split[2]}"
            feats["sub_ID"] = sub_ID

            # label
            feats["label"] = re.search(r"_(rest|move|suppr)", filename).group(1)

            feature_rows.append(feats)

        # final DataFrame for one recording
        df_feats = pd.DataFrame(feature_rows)
        col_order = ["sub_ID", "label", "window_range_ms"] + [col for col in df_feats.columns if
                                                              col not in ["sub_ID", "label", "window_range_ms"]]
        df_recs_train = df_feats[col_order]
        df_list.append(df_recs_train)

    # getting out df from all recs #

    df_train = pd.concat(df_list)
    df_train.to_hdf(output_dir, key="data", mode="w")


def permutation_importance(model, X_test, y_test, metric=accuracy_score, n_repeats=5):
    """
    shuffels each feature column 5 times and outputs the mean difference to the baseline accuracy
    (=accuracy with all features)
    :param model: which model you want to test it on
    :param X_test:
    :param y_test:
    :param metric: sets with which metric (e.g accuracy) you want to test the change of performance
    :param n_repeats: how many times each column gets shuffeled
    :return: returns the mean difference from the metric with the shuffeled col vs. the metric score with all cols
    """
    X_test = pd.DataFrame(X_test)
    baseline = metric(y_test, model.predict(X_test))
    importances = pd.Series(0.0, index=X_test.columns)

    for col in X_test.columns:
        drop_scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            drop = baseline - metric(y_test, model.predict(X_permuted))
            drop_scores.append(drop)
        importances[col] = np.mean(drop_scores)

    return importances




from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

def prepare_loso_data(df_train):
    """
    processes Leave-One-Subject-Out Cross-Validation Splits with Z-Scoring.

    Args:
        df_train (pd.DataFrame): holds all training data:
            'sub_ID', 'label', 'window_range_ms' and Feature-columns.

    Returns:
        List[dict]: every entry holds X_train, X_test, y_train, y_test, scaler, test_sub_ID
        ( = every entry represents a different combination of training subs and one test sub)
    """
    # extracting non-feat and feat columns
    non_feature_cols = ["sub_ID", "label", "window_range_ms"]
    feature_cols = [col for col in df_train.columns if col not in non_feature_cols]

    # get arrays
    X = df_train[feature_cols].values
    y = df_train["label"].values
    groups = df_train["sub_ID"].values

    # LOSO
    logo = LeaveOneGroupOut()
    loso_data = []

    # loop through LOSO-Splits
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Z-Scoring
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # Test-Proband merken
        test_sub = np.unique(groups[test_idx])[0] # e.g "sub_E"

        # Speichern
        loso_data.append({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
            "test_sub_ID": test_sub,
            "groups": groups
        })

    return loso_data



# --- plotting PCA for a certain test-sub with train data from the other subs --- #
def plot_subject_feature_space(train_df, test_fold, feature_cols, title=None):
    """
    Visualising trainingdata + testdata of a LOSO-Fold in PCA-2D-Space.
    """

    # --- extracting training- & testdata ---
    X_train = test_fold["X_train"]
    y_train = test_fold["y_train"]
    X_test = test_fold["X_test"]
    y_test = test_fold["y_test"]
    test_sub = test_fold["test_sub_ID"]

    # Combine for plot
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    origin = np.array(["train"] * len(X_train) + ["test"] * len(X_test)) # is data point coming from train or test sub

    # --- PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    # --- DataFrame for plot ---
    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "label": y_all,
        "origin": origin
    })

    # --- Plot ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="label", style="origin", s=80)
    plt.title(title or f"Feature-Space (Test Subject: {test_sub})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pca



def plot_knn_predictions_clean(X_train, X_test, y_train, y_test, test_sub, k=5):
    # PCA-Reduction
    pca = PCA(n_components=2)
    X_all = np.vstack([X_train, X_test])
    X_pca = pca.fit_transform(X_all)
    X_train_pca = X_pca[:len(X_train)]
    X_test_pca = X_pca[len(X_train):]

    # colors and markers
    color_map = {"move": "blue", "rest": "orange", "suppr": "green"}
    marker_map = {"correct": "x", "incorrect": "P"}

    # KNN
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))

    # training points
    for label in np.unique(y_train):
        mask = np.array(y_train) == label
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                    color=color_map[label], alpha=0.3, label=f"{label} (train)", s=30)

    # test points
    for i in range(len(y_test)):
        true_label = y_test[i]
        pred_label = y_pred[i]
        correct = true_label == pred_label
        marker = marker_map["correct"] if correct else marker_map["incorrect"]
        plt.scatter(X_test_pca[i, 0], X_test_pca[i, 1],
                    color=color_map[true_label], marker=marker,
                    edgecolor='black', s=80,
                    label=f"{true_label} (test - {'✔' if correct else '✘'})")

    plt.title(f"KNN (k={k}) – {test_sub}\nTest Accuracy: {acc:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    # only one legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.tight_layout()
    plt.show()

    return acc



def plot_loso_pca(loso_data, feature_names=None, n_top_features=10, title=None):
    """
    Führt PCA auf Trainingsdaten jedes LOSO-Folds durch, transformiert Testdaten,
    kombiniert alle Daten für einen Plot und zeigt Top-Features nach PCA-Bedeutung.

    Args:
        loso_data (list of dict): Jeder Fold ist ein Dict mit Schlüsseln:
            - 'X_train', 'y_train', 'X_test', 'y_test'
        feature_names (list of str): Optional. Namen der Features (für Top-Feature-Analyse)
        n_top_features (int): Wie viele wichtige Features ausgegeben werden sollen
        title (str): Optionaler Titel für den Plot
    """
    X_alls = []
    y_alls = []
    origins = []
    pca_components = []

    for fold in loso_data:
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]

        # fit PCA only on train data
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # save components
        pca_components.append(pca.components_)

        # combine for plotting
        X_all_fold = np.vstack([X_train_pca, X_test_pca])
        y_all_fold = np.concatenate([y_train, y_test])
        origin_fold = np.array(["train"] * len(X_train) + ["test"] * len(X_test))

        X_alls.append(X_all_fold)
        y_alls.append(y_all_fold)
        origins.append(origin_fold)

    # combine folds
    X_all_combined = np.vstack(X_alls)
    y_all_combined = np.concatenate(y_alls)
    origin_combined = np.concatenate(origins)

    # df for plot
    df_plot = pd.DataFrame({
        "PC1": X_all_combined[:, 0],
        "PC2": X_all_combined[:, 1],
        "label": y_all_combined,
        "origin": origin_combined
    })

    # plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="label", style="origin", s=80)
    plt.title(title or "PCA-Visualisierung über alle LOSO-Folds (Train/Test getrennt)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- analyze most important feats ---
    if feature_names is not None:
        pca_components_arr = np.array(pca_components)  # shape: (n_folds, 2, n_features)
        avg_components = np.mean(np.abs(pca_components_arr), axis=0)  # shape: (2, n_features)

        pc1 = avg_components[0]
        pc2 = avg_components[1]

        # DataFrame mit Feature-Namen und Werten
        df_features = pd.DataFrame({
            "feature": feature_names,
            "PC1": pc1,
            "PC2": pc2
        })

        # Top positive & negative Features für PC1
        print(f"\nTop {n_top_features} positive Features for PC1:")
        print(df_features.sort_values("PC1", ascending=False).head(n_top_features).to_string(index=False))

        print(f"\nTop {n_top_features} negative Features for PC1:")
        print(df_features.sort_values("PC1", ascending=True).head(n_top_features).to_string(index=False))

        print(f"\nTop {n_top_features} positive Features for PC2:")
        print(df_features.sort_values("PC2", ascending=False).head(n_top_features).to_string(index=False))

        print(f"\nTop {n_top_features} negative Features for PC2:")
        print(df_features.sort_values("PC2", ascending=True).head(n_top_features).to_string(index=False))




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

def fit_model_acc_sens_spec(loso_data, model_name, ):
    """
    trains model and outputs mean accuracy, mean sensitivity and mean specifity for the LOSO
    folds
    :param loso_data: holds dicts of each fold (train data etc)
    :param model_name: sets which model you want to use (possibilities: logistic_regression, KNN, random_forest)
    :return: prints out mean acc, mean sens, mean spec
    """
    # string classes
    class_names = ["rest", "move", "suppr"]
    le = LabelEncoder()
    le.fit(class_names)

    if model_name.lower() == "logistic_regression":
        model = LogisticRegression(
        solver='lbfgs',
        C=0.1,
        penalty='l2',
        max_iter=1000
        )

    if model_name.lower() == "knn":
        model = KNeighborsClassifier()

    if model_name.lower() == "random_forest":
        model = RandomForestClassifier(max_features=6)

    accuracies = []
    sensitivities = {cls: [] for cls in class_names}
    specificities = {cls: [] for cls in class_names}

    for fold in loso_data:
        y_train_enc = le.transform(fold["y_train"])
        y_test_enc = le.transform(fold["y_test"])

        model.fit(fold["X_train"], y_train_enc)
        preds = model.predict(fold["X_test"])

        # Accuracy
        acc = accuracy_score(y_test_enc, preds)
        accuracies.append(acc)

        # sensitivity & specificity for each class/label
        for idx, cls in enumerate(class_names):
            y_test_bin = (y_test_enc == idx).astype(int)
            preds_bin = (preds == idx).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test_bin, preds_bin).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            sensitivities[cls].append(sensitivity)
            specificities[cls].append(specificity)

    # results
    print(f"{model_name} Accuracy Ø: {np.mean(accuracies):.3f}")
    print("Accuracy for each fold:", np.round(accuracies, 3))

    for cls in class_names:
        mean_sens = np.mean(sensitivities[cls])
        mean_spec = np.mean(specificities[cls])
        print(f"{cls}: sensitivity Ø = {mean_sens:.3f}, specificity Ø = {mean_spec:.3f}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_cm(model_name, folds_data, sub_id):
    """
    plots confusion matrix for one test sub/fold from the LOSO-Validation
    :param model_name: which model you want to perform the test with
    :param folds_data: holds the dicts for all folds
    :param sub_id: the subject that will be tested, the remaining subs make up the training data
    :return: plots cm plot
    """
    if model_name.lower() == "logistic_regression":
        model = LogisticRegression(
            solver='lbfgs',
            C=0.1,
            penalty='l2',
            max_iter=1000
        )
    elif model_name.lower() == "knn":
        model = KNeighborsClassifier()

    elif model_name.lower() == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError(
            f"Unknown model: {model_name}. /n Possible models are: Logistic_Regression, KNN, Random_Forest")

    # Find the fold where 'sub_ID' matches the input `sub_id`
    fold = None
    for f in folds_data:
        if f.get("test_sub_ID") == sub_id:
            fold = f
            break

    if fold is None:
        raise ValueError(f"No fold found for sub_ID: {sub_id}")

    X_test = fold["X_test"]
    y_true = fold["y_test"]
    test_sub = fold["test_sub_ID"]

    # train model and test
    model.fit(fold["X_train"], fold["y_train"])
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)

    # 2. Plotten
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", values_format=".0f")
    plt.title(f"Confusion Matrix {model_name} (Test Subject: {test_sub})")
    plt.grid(False)
    plt.show()




from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_loso_roc(loso_data, model_name, class_names, mode='one-vs-all'):
    """
    mode: 'one-vs-all' → 3 curves (one for each class)
          'macro'       → an averaged ROC for all classes
    """

    if model_name.lower() == "logistic_regression":
        model = LogisticRegression(
        solver='lbfgs',
        C=0.1,
        penalty='l2',
        max_iter=1000
        )

    if model_name.lower() == "knn":
        model = KNeighborsClassifier()

    if model_name.lower() == "random_forest":
        model = RandomForestClassifier()

    n_classes = len(class_names)
    le = LabelEncoder()
    le.fit(class_names)

    # collect all true labels and scores for all folds
    all_y_true_bin = []
    all_y_score = []

    for fold in loso_data:
        y_train_enc = le.transform(fold["y_train"])
        y_test_enc = le.transform(fold["y_test"])

        model.fit(fold["X_train"], y_train_enc)
        probs = model.predict_proba(fold["X_test"])  # shape: (n_samples, n_classes)

        # one-hot encode y_test
        y_test_bin = label_binarize(y_test_enc, classes=range(n_classes))

        all_y_true_bin.append(y_test_bin)
        all_y_score.append(probs)

    # add all together
    y_true_all = np.vstack(all_y_true_bin)
    y_score_all = np.vstack(all_y_score)

    if mode == 'one-vs-all':
        plt.figure()
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_all[:, i], y_score_all[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cls} vs. the rest (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC (One-vs-All for all LOSO-Folds)")
        plt.legend()
        plt.grid()
        plt.show()

    if mode == 'macro':
        # Interpolation auf gemeinsamen FPR-Raster
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        aucs = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_all[:, i], y_score_all[:, i])
            interp_tpr = np.interp(all_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Start bei (0,0)
            mean_tpr += interp_tpr
            aucs.append(auc(fpr, tpr))
        mean_tpr /= n_classes
        mean_tpr[-1] = 1.0  # Ende bei (1,1)
        mean_auc = np.mean(aucs)
        # Plot
        plt.figure()
        plt.plot(all_fpr, mean_tpr, color='blue', label=f"Macro-Averaged ROC (AUC = {mean_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Macro-Averaged ROC über alle Klassen (LOSO)")
        plt.legend()
        plt.grid()
        plt.show()




# schonmal funktion für die kontinuierlichen daten!
def detect_onsets(predictions, target_class='movement'):
    onsets = []
    for i in range(1, len(predictions)):
        if predictions[i-1] != target_class and predictions[i] == target_class:
            onsets.append(i)
    return onsets






##### ================================== GUI MOVEMENT DETECTION FUNCTIONS =========================================#####
from matplotlib.widgets import Slider

# --- optional: SciPy for "label"; automatic fallback to numpy if scipy isnt available ---
try:
    from scipy.ndimage import label as scipy_label
    def _label_runs(mask):
        # compatile with scipy.ndimage.label
        return scipy_label(mask.astype(np.uint8))
except Exception:
    def _label_runs(mask):
        # simple fallback-implementation without scipy
        mask = mask.astype(bool)
        # Grenzen der Runs finden
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, mask.size]
        # build "label image"
        lbl_img = np.zeros_like(mask, dtype=int)
        for i, (s, e) in enumerate(zip(starts, ends), start=1):
            lbl_img[s:e] = i
        n_labels = len(starts)
        return lbl_img, n_labels


# ---------- helpers ----------
_COLUMN_ALIASES = {
    # allows for different col_names in the dfs
    "SVM_L": ["SVM_L", "ACC_SVM_L", "svm_l", "acc_svm_l"],
    "SVM_R": ["SVM_R", "ACC_SVM_R", "svm_r", "acc_svm_r"],
    "brachioradialis_L": ["brachioradialis_L", "Brachioradialis_L"],
    "brachioradialis_R": ["brachioradialis_R", "Brachioradialis_R"],
    "deltoideus_L": ["deltoideus_L", "Deltoideus_L"],
    "deltoideus_R": ["deltoideus_R", "Deltoideus_R"],
    "tibialisAnterior_L": ["tibialisAnterior_L", "TibialisAnterior_L"],
    "tibialisAnterior_R": ["tibialisAnterior_R", "TibialisAnterior_R"],
    "Sync_Time (s)": ["Sync_Time (s)", "time", "Time", "t"]
}


def _parse_meta_from_basename(base_noext):
    # sub
    m = re.search(r"(?:^|_)(sub-[A-Za-z0-9]+)", base_noext, flags=re.I)
    sub = m.group(1).split("-")[1] if m else "unk"

    # setup
    if re.search(r"setup[_-]?a", base_noext, re.I):
        setup = "setupA"
    elif re.search(r"setup[_-]?b", base_noext, re.I):
        setup = "setupB"
    else:
        setup = "setupA"  # Fallback

    # task (Deckt deine Namen ab)
    m_task = re.search(r"(Move1|Move2|MoveMockDys|RestMockDys|Move|Rest)", base_noext, re.I)
    task = m_task.group(1) if m_task else "Move"
    return sub, setup, task


def _find_column(df, canonical_name):
    """looks for correct col in df with the alias list"""
    for cand in _COLUMN_ALIASES.get(canonical_name, [canonical_name]):
        if cand in df.columns:
            return cand
    return None

def _get_baseline_segment_by_key(df, baseline_dict, baseline_key, sf):
    """
    baseline_key: e.g. "arm_L_setupA_Move1", "arm_R_setupA_Rest", "leg_L_setupB_Move" ...
    Fallback: first 5 seconds.
    """
    # find time column
    t_col = _find_column(df, "Sync_Time (s)")
    if t_col is None:
        times = np.arange(len(df)) / float(sf)
    else:
        times = df[t_col].to_numpy()

    if baseline_dict and baseline_key in baseline_dict:
        bstart, bend = baseline_dict[baseline_key]
    else:
        bstart, bend = 0.0, min(5.0, float(times[-1]) if len(times) else 5.0)

    i0 = int(np.searchsorted(times, bstart))
    i1 = int(np.searchsorted(times, bend))
    i1 = max(i1, i0 + 1)
    return slice(i0, i1), times

from matplotlib.transforms import Bbox
# ---------- main window-function ----------
def _interactive_window(df, channels, side, sf, results_dict, window_title, init_params, baseline_dict,
                        setup, task, limb_label, png_path=None):
    """
    opens one matplotlib-wondow with n subplots and the sliders.
    Results (last slider-position + on/offsets) are saved in results_dict.
    """
    # baseline
    baseline_key = f"{limb_label}_{setup}_{task}"
    baseline_slice, sync_times = _get_baseline_segment_by_key(
        df, baseline_dict, baseline_key, sf
    )

    n_ch = len(channels)
    fig_h = 3.6 * n_ch
    fig, axs = plt.subplots(n_ch, 1, figsize=(12, fig_h), sharex=True)
    axs = np.atleast_1d(axs)
    plt.subplots_adjust(bottom=0.12 + 0.10 * n_ch)
    fig.canvas.manager.set_window_title(window_title)

    def _on_close(_evt):
        if png_path:
            try:
                # Rendern
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()  # type: ignore

                # Nur die EMG-Axes
                axes_list = np.atleast_1d(axs)

                # Inch-BBox aus allen EMG-Axes
                bboxes_inch = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                               for ax in axes_list]
                tight_bbox = Bbox.union(bboxes_inch)

                # Debug-Ausgabe
                w_in, h_in = tight_bbox.width, tight_bbox.height
                print(f"[PNG DEBUG] Subplot-BBox (inches): {w_in:.2f} x {h_in:.2f}")

                # DPI-Sicherheitscheck
                target_dpi = 200
                max_pixels = 10000
                if w_in * target_dpi > max_pixels or h_in * target_dpi > max_pixels:
                    scale_w = max_pixels / (w_in * target_dpi)
                    scale_h = max_pixels / (h_in * target_dpi)
                    scale = min(scale_w, scale_h)
                    target_dpi = int(target_dpi * scale)
                    print(f"[PNG WARN] zu groß – DPI reduziert auf {target_dpi}")

                fig.savefig(
                    png_path,
                    dpi=target_dpi,
                    bbox_inches=tight_bbox.expanded(1.02, 1.05),
                    pad_inches=0.05
                )
                print(f"[PNG] gespeichert (nur Subplots): {png_path}")

            except Exception as e:
                print(f"[PNG] Fehler beim Speichern ({png_path}): {e}")

    fig.canvas.mpl_connect("close_event", _on_close)

    # lists for saving
    threshold_lines = []
    onset_lines = [[] for _ in channels]
    offset_lines = [[] for _ in channels]
    mu_sigma = []
    signals = []
    colnames = []

    # slider-container
    k_sliders, dur_sliders, pause_sliders = [], [], []

    # for each channel: set axes and sliders
    for i, ch in enumerate(channels):
        col = _find_column(df, ch)
        colnames.append(col)
        ax = axs[i]
        if col is None:
            ax.text(0.5, 0.5, f"{ch} (Spalte fehlt)", ha="center", va="center")
            ax.set_title(ch)
            threshold_lines.append(None)
            k_sliders.append(None); dur_sliders.append(None); pause_sliders.append(None)
            mu_sigma.append((np.nan, np.nan))
            signals.append(np.zeros(len(df)))
            continue

        signal = df[col].to_numpy()
        signals.append(signal)

        # baseline_vals if needed
        baseline_signal = signal[baseline_slice]
        mu, sigma = float(np.mean(baseline_signal)), float(np.std(baseline_signal, ddof=0))
        mu_sigma.append((mu, sigma))

        # plot
        ax.plot(sync_times, signal, label=ch)
        ax.set_title(f"{ch}")
        ax.legend(loc='upper right')
        (thr_line,) = ax.plot([], [], ls='--', c='r', label='Threshold')
        threshold_lines.append(thr_line)

        # Slider
        # layout: for each channel a "row" of sliders
        y0 = 0.05 + i * 0.09
        axk   = plt.axes([0.10, y0, 0.25, 0.03])
        axdur = plt.axes([0.40, y0, 0.20, 0.03])
        axpau = plt.axes([0.65, y0, 0.20, 0.03])

        k0   = init_params.get(ch, {}).get("k", 5.0)
        dur0 = init_params.get(ch, {}).get("min_dur", 0.0)
        pau0 = init_params.get(ch, {}).get("min_pause", 0.0)

        sk   = Slider(axk,   f'{ch} k',        0.0, 100.0, valinit=k0,   valstep=0.5)
        sdur = Slider(axdur, 'Dauer (s)',      0.0,   5.0, valinit=dur0, valstep=0.05)
        spau = Slider(axpau, 'Pause (s)',      0.0,   5.0, valinit=pau0, valstep=0.05)

        k_sliders.append(sk); dur_sliders.append(sdur); pause_sliders.append(spau)

    # --- computing function ---
    def recompute_and_draw(_=None):
        for i, ch in enumerate(channels):
            col = colnames[i]
            if col is None:
                continue

            signal = signals[i]
            mu, sigma = mu_sigma[i]
            k = k_sliders[i].val
            min_duration_s = dur_sliders[i].val
            min_pause_s    = pause_sliders[i].val

            # threshold
            threshold = mu + k * sigma
            threshold_lines[i].set_data(sync_times, np.full_like(sync_times, threshold, dtype=float))

            # binarize + min_duration
            above = signal > threshold
            labels_array, n_lbl = _label_runs(above)
            valid = np.zeros_like(above, dtype=bool)
            min_len = int(round(min_duration_s * sf))
            if min_len <= 1:
                valid = above.copy()
            else:
                for lbl in range(1, n_lbl + 1):
                    idx = np.where(labels_array == lbl)[0]
                    if idx.size >= min_len:
                        valid[idx] = True

            # on-/offsets
            diff = np.diff(valid.astype(int))
            onsets  = np.where(diff ==  1)[0] + 1
            offsets = np.where(diff == -1)[0] + 1

            onsets, offsets = take_out_short_off_onset(onsets, offsets, min_pause_s, sf)

            # delete old lines
            for l in onset_lines[i] + offset_lines[i]:
                try: l.remove()
                except Exception: pass
            onset_lines[i].clear(); offset_lines[i].clear()

            # draw new lines
            for x in onsets:
                onset_lines[i].append(axs[i].axvline(sync_times[x], ls="--", c="g"))
            for x in offsets:
                offset_lines[i].append(axs[i].axvline(sync_times[x], ls="--", c="k"))

            # store results (samples + seconds)
            # seconds via sync_times
            if np.issubdtype(sync_times.dtype, np.number):
                on_sec  = [float(sync_times[x]) for x in onsets]
                off_sec = [float(sync_times[x]) for x in offsets]
            else:
                on_sec  = [float(x)/sf for x in onsets]
                off_sec = [float(x)/sf for x in offsets]

            pairs_samples = new_on_offsets(list(map(int, onsets)), list(map(int, offsets)), end_index=len(signal) - 1)
            pairs_seconds = new_on_offsets(on_sec, off_sec, end_index=float(sync_times[-1]))

            results_dict[ch] = {
                "k_value": float(k),
                "min_dur": float(min_duration_s),
                "min_pause": float(min_pause_s),
                "on_offsets_samples": pairs_samples,
                "on_offsets": pairs_seconds
            }

        fig.canvas.draw_idle()

    # sliders
    for s in k_sliders + dur_sliders + pause_sliders:
        if s is not None:
            s.on_changed(recompute_and_draw)

    # first run
    recompute_and_draw()

    # titel
    plt.suptitle(window_title, fontsize=14)

    plt.show()  # blocking; after closing, starts again
    # Ergebnisse stehen bereits in results_dict (letzter Zustand)
    # results are already in results_dict (last status)


# ---------- main function ----------
def interactive_move_detection(
    filepaths,
    output_dir,
    fig_dir,
    sf,
    baseline_dict=None,
    init_params=None
):
    """
    Parameters
    ----------
    filepaths : list[str]
        list of h5 files (one file per recording).
    output_dir : str
        target folder for the JSONs (one JSON per file).
    sf : float
        Sampling-Rate (Hz).
    baseline_dict : dict|None
        z.B. {'L': (start_s, end_s), 'R': (start_s, end_s)}.
        Fallback: first 5 seconds.
    init_params : dict|None
        optional baseline vals per channel, e.g.:
        {
          "brachioradialis_L": {"k": 5, "min_dur": 0.5, "min_pause": 0.2},
          "SVM_R": {"k": 8, "min_dur": 0.3, "min_pause": 0.15}
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    for fp in filepaths:
        base = os.path.basename(fp)
        base_noext, _ = os.path.splitext(base)
        sub, setup, task = _parse_meta_from_basename(base_noext)
        basename_lower = base.lower()

        # ------------ Setup erkennen ------------
        if "setupa" in basename_lower or "setup_a" in basename_lower:
            # 2 windows à 3 Subplots
            windows = [
                (["brachioradialis_L_tkeo", "deltoideus_L_tkeo", "SVM_L_smooth_rms"], "L", f"{base_noext} – SetupA – LEFT_ARM", "arm_L",
                 os.path.join(fig_dir, f"sub-{sub}_setupA_{task}_armL.png")),
                (["brachioradialis_R_tkeo", "deltoideus_R_tkeo", "SVM_R_smooth_rms"], "R", f"{base_noext} – SetupA – RIGHT_ARM", "arm_R",
                 os.path.join(fig_dir, f"sub-{sub}_setupA_{task}_armR.png"))
            ]
        elif "setupb" in basename_lower or "setup_b" in basename_lower:
            windows = [
                (["brachioradialis_L_tkeo", "deltoideus_L_tkeo", "SVM_L_smooth_rms"], "L", f"{base_noext} – SetupB – LEFT_ARM", "arm_L",
                 os.path.join(fig_dir, f"sub-{sub}_setupB_{task}_armL.png")),
                (["tibialisAnterior_L_tkeo", "SVM_R_smooth_rms"], "R", f"{base_noext} – SetupB – LEFT_LEG", "leg_L",
                 os.path.join(fig_dir, f"sub-{sub}_setupB_{task}_legL.png"))
            ]
        else:
            print(f"[WARN] Konnte Setup in Dateinamen nicht erkennen: {base} (nehme SetupA an)")
            windows = [
                (["brachioradialis_L", "deltoideus_L", "SVM_L"], "L", f"{base_noext} – Setup? – LEFT_ARM", "arm_L",
                 os.path.join(fig_dir, f"sub-{sub}_setupUnknown_{task}_armL.png")),
                (["brachioradialis_R", "deltoideus_R", "SVM_R"], "R", f"{base_noext} – Setup? – RIGHT_ARM", "arm_R",
                 os.path.join(fig_dir, f"sub-{sub}_setupUnknown_{task}_armR.png"))
            ]

        # ------------ load data ------------
        try:
            df = pd.DataFrame(pd.read_hdf(fp, key="data"))
        except Exception as e:
            print(f"[ERROR] Konnte {fp} nicht lesen: {e}")
            continue

        df = funcs.add_tkeo(df, ["brachioradialis_L", "deltoideus_L", "brachioradialis_R", "deltoideus_R",
                                 "tibialisAnterior_L", "tibialisAnterior_R"], window_size=20)
        df = funcs.apply_savgol_rms_acc(df, ["SVM_L", "SVM_R"])

        # ------------ interactive windows ------------
        results_all = {}  # here, we save per folder all channels
        limb_label = None
        for channels, side, title, limb_label, png_path in windows:
            _interactive_window(
                df=df,
                channels=channels,
                side=side,
                sf=sf,
                results_dict=results_all,
                window_title=title,
                init_params=init_params or {},
                baseline_dict=baseline_dict or {},
                setup=setup,
                task=task,
                limb_label=limb_label,
                png_path=png_path
            )

        # ------------ write JSON (one per file) ------------
        # channels, that miss in the df → results_all hold only the ones that are present
        out_path = os.path.join(output_dir, f"{base_noext[:-10]}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results_all, f, ensure_ascii=False, indent=2)
            print(f"[OK] JSON Gespeichert: {out_path}")
        except Exception as e:
            print(f"[ERROR] Konnte JSON nicht schreiben ({out_path}): {e}")





def reprocess_one_file(
    filepath,
    output_dir,
    fig_dir,
    sf,
    baseline_dict=None,
    init_params=None,
    write_mode="overwrite"  # "overwrite" | "versioned" | "merge"
):
    """
    opens only that one file, shows both windows and
    writes the corrects json + png for that one file (use for correcting misdetected files for example)
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    base = os.path.basename(filepath)
    base_noext, _ = os.path.splitext(base)

    # get metadata from filename
    sub, setup, task = _parse_meta_from_basename(base_noext)

    try:
        df = pd.read_hdf(filepath)
    except Exception as e:
        print(f"[ERROR] Konnte {filepath} nicht lesen: {e}")
        return

    df = funcs.add_tkeo(df, ["brachioradialis_L", "deltoideus_L", "brachioradialis_R", "deltoideus_R",
                             "tibialisAnterior_L", "tibialisAnterior_R"], window_size=20)
    df = funcs.apply_savgol_rms_acc(df, ["SVM_L", "SVM_R"])

    # window definitions
    if setup == "setupA":
        windows = [
            (["brachioradialis_L_tkeo", "deltoideus_L_tkeo", "SVM_L_smooth_rms"], "L",
             f"{base_noext} – SetupA – LEFT",
             "arm_L",
             os.path.join(fig_dir, f"sub-{sub}_setupA_{task}_armL.png")),
            (["brachioradialis_R_tkeo", "deltoideus_R_tkeo", "SVM_R_smooth_rms"], "R",
             f"{base_noext} – SetupA – RIGHT",
             "arm_R",
             os.path.join(fig_dir, f"sub-{sub}_setupA_{task}_armR.png")),
        ]
    elif setup == "setupB":
        windows = [
            (["brachioradialis_L_tkeo", "deltoideus_L_tkeo", "SVM_L_smooth_rms"], "L",
             f"{base_noext} – SetupB – LEFT",
             "arm_L",
             os.path.join(fig_dir, f"sub-{sub}_setupB_{task}_armL.png")),
            (["tibialisAnterior_L_tkeo", "SVM_R_smooth_rms"], "R",
             f"{base_noext} – SetupB – SECOND",
             "leg_L",
             os.path.join(fig_dir, f"sub-{sub}_setupB_{task}_legL.png")),
        ]
    else:
        windows = [
            (["brachioradialis_L_tkeo", "deltoideus_L_tkeo", "SVM_L_smooth_rms"], "L",
             f"{base_noext} – Setup? – LEFT",
             "arm_L",
             os.path.join(fig_dir, f"sub-{sub}_setupUnknown_{task}_armL.png")),
            (["brachioradialis_R_tkeo", "deltoideus_R_tkeo", "SVM_R_smooth_rms"], "R",
             f"{base_noext} – Setup? – RIGHT",
             "arm_R",
             os.path.join(fig_dir, f"sub-{sub}_setupUnknown_{task}_legL.png")),
        ]

    # for versions (optional!)
    def _versioned_path(path):
        if write_mode != "versioned" or not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        v = 2
        while True:
            cand = f"{stem}_v{v}{ext}"
            if not os.path.exists(cand):
                return cand
            v += 1

    # interactive window -> save results
    results_all = {}
    for channels, side, title, limb_label, png_path in windows:
        _interactive_window(
            df=df,
            channels=channels,
            side=side,
            sf=sf,
            results_dict=results_all,
            window_title=title,
            init_params=init_params or {},
            baseline_dict=baseline_dict or {},
            setup=setup,
            task=task,
            limb_label=limb_label,
            png_path=_versioned_path(png_path)
        )

    # write json only for this file
    out_path = os.path.join(output_dir, f"{base_noext}.json")
    if write_mode == "merge" and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
        old.update(results_all)
        payload = old
    else:
        # overwrite / versioned
        if write_mode == "versioned" and os.path.exists(out_path):
            stem, ext = os.path.splitext(out_path)
            v = 2
            while True:
                cand = f"{stem}_v{v}{ext}"
                if not os.path.exists(cand):
                    out_path = cand
                    break
                v += 1
        payload = results_all

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON gespeichert: {out_path}")



