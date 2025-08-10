import numpy as np
import pandas as pd
from hyperframe.frame import DataFrame
import os
from scipy.signal import butter, sosfiltfilt
from combined_analysis_bachelor.code.functions_for_pipeline import add_tkeo_add_envelope, \
    filtered_and_notched
from utils.find_paths import get_onedrive_path
from utils.get_sub_dir import get_sub_folder_dir



def create_processed_files(source_directory, target_directory,
                           acc_filter_parameters, emg_filter_parameters, sf):

    """takes in the raw (trimmed) files, generates SVM columns for ACC data, applies filtering, generates new filenames and saves files
    to target directory

    input:
    - source_directory
    - targes_directory
    - acc_filter_parameters & emg_filter_parameters : lists that hold the desired lfreq and hfreq for filtering

    returns the files that hold the processed data in the target directory
    """
    ### ==== get all filepaths ==== ###
    filepaths = []
    for file in os.listdir(source_directory):
        if file.endswith(".h5"):
            filepath = f"{source_directory}/{file}"
            filepaths.append(filepath)

    print(f"retrieved data: {filepaths}")

    ### ===== looping trough files ===== ###
    for filepath in filepaths:

        # ----- read in raw file ----- #
        raw = pd.read_hdf(filepath, key="data")
        raw_df = pd.DataFrame(raw) # make sure its a df, not an object
        print(raw_df.columns)

        # --- create euclidean norm column per side --- #
        #svm_left = np.sqrt(raw_df["acc_x_hand_L"]**2 + raw_df["acc_y_hand_L"]**2 +  raw_df["acc_z_hand_L"]**2)
        #svm_right = np.sqrt(raw_df["acc_x_leg_L"]**2 + raw_df["acc_y_leg_L"]**2 +  raw_df["acc_z_leg_L"]**2)
        svm_right = np.sqrt(raw_df["acc_x_hand_R"] ** 2 + raw_df["acc_y_hand_R"] ** 2 + raw_df["acc_z_hand_R"] ** 2)
        #raw_df["SVM_L"] = svm_left
        #raw_df["SVM_leg_L"] = svm_right
        raw_df["SVM_R"] = svm_right


        # ----- perform filtering ----- #
        emg_cols = raw_df.columns[raw_df.columns.str.contains('brachioradialis|deltoideus|tibialis',
                                                                               case=False, regex=True)].tolist()
        acc_cols = raw_df.columns[raw_df.columns.str.contains("SVM|hand",
                                                             case=False, regex=True)].tolist() # zurück ändern??

        #raw_df = raw_df.drop(["acc_x_hand_R", "acc_y_hand_R", "acc_z_hand_R"], axis=1) # , "acc_x_leg_L", "acc_y_leg_L", "acc_z_leg_L"]) # zurück ändern?
        filtered_df = filtered_and_notched(raw_df, acc_cols, emg_cols, acc_filter_parameters, emg_filter_parameters, sf)


        # ----- creating new file ----- #
        filename = os.path.basename(filepath)
        print(filename)
        if filename.endswith(".h5"):
            filename = filename[:-6]
        filename += "processed.h5"

        target_filepath = os.path.join(target_directory, filename)


        print(f"Saving to: {target_filepath}")

        filtered_df.to_hdf(target_filepath, key="data", mode="w")

    print("\n✅ Every File processed and saved!")

#create_processed_files(source_dir, target_dir, [1,20], [20,450], 1000)