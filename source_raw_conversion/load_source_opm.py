# import packages
import pandas as pd
import numpy as np
import os
import mne

# load custom
from utils.load_utils import get_onedrive_path
from source_raw_conversion.time_syncing import cut_data_to_task_timing



def extract_opm_sourcedata(sub_config, ACQ, TASK,
                           STORE_TRIGGER_TIMES=True,):

    SUB = sub_config["subject_id"][-2:]

    # define paths
    sub_opm_path = os.path.join(get_onedrive_path('source_data'),
                                f'sub-{SUB}', 'opm')
    opm_folders = os.listdir(sub_opm_path)  # get available files
    
    # get meg file for task and acquisition
    sel_folder = [f for f in opm_folders if ACQ in f.lower() and TASK in f.lower()][0]

    sel_file = [f for f in os.listdir(os.path.join(sub_opm_path, sel_folder))
                if f.endswith(sub_config["meg_file_ext"])][0]
    
    fpath = os.path.join(sub_opm_path, sel_folder, sel_file)

    if sub_config['rec_location'] == 'PTB':
        # use PTB system specifics
        lvm_df = read_lvm_to_df(meg_file_path=fpath)

        (
            meg_sfreq,
            megtimes,
            megdata,
            meg_chnames,
            meg_trigger
        ) = get_data_from_lvm_df(lvm_df=lvm_df)
    
    ### store trigger times in meg-timing
    if STORE_TRIGGER_TIMES:
        fpath = os.path.join(get_onedrive_path('raw_data'),
                             f'sub-{SUB}', 'opm',
                             f'sub-{SUB}_{TASK}_{ACQ}_opm_triggertimes.npy')
        # define threshold as 50% of max
        thr = np.nanmax(meg_trigger)
        # get indices and times of trigger presses
        trigger_idx = np.where(np.diff(meg_trigger) > (.5 * thr))[0]
        trigger_times = np.array(megtimes[trigger_idx])
        np.save(fpath, trigger_times)
        print(f'stored opm TRIGGERS in {fpath}')


    return megtimes, megdata, meg_sfreq, meg_chnames, meg_trigger



def read_lvm_to_df(meg_file_path):
    
    # LabVIEW .lvm usually has a header of variable length, often ending with "***End_of_Header***"
    with open(meg_file_path, "r") as f:
        lines = f.readlines()

    # Find header end
    for i, line in enumerate(lines):

        if "End_of_Header" in line:
            header_end = i
            break

    # Load into DataFrame
    df = pd.read_csv(
        meg_file_path,
        sep="\t",          # LVM is tab-separated
        skiprows=header_end+1,
        engine="python"
    )

    return df


def find_megtrigger_in_AI(megdata, megchnames, SYSTEM='PTB',):
    """
    finds trigger channel in MEG data based on largest max value
    in AI (analog input) channels, returns trigger timeseries
    """
    # PTB and FieldLine systems have different analog codes in chnames
    AI_CODE = {'PTB': 'AI', 'FL': None}

    # find trigger AI (analog input) channel based on largest max
    trigger_ai_idx = np.argmax(
        [np.nanmax(megdata[:, i]) for i, ch in enumerate(megchnames)
        if ch.startswith(AI_CODE[SYSTEM])]
    )
    # find matching chname
    trigger_chname = [
        ch for ch in megchnames if ch.startswith(AI_CODE[SYSTEM])
    ][trigger_ai_idx]
    # find index in all data
    trigger_idx_global = np.where(
        [ch == trigger_chname for ch in megchnames]
    )[0][0]

    # assign trigger timeseries
    trigger_series = megdata[:, trigger_idx_global]

    return trigger_series



def get_data_from_lvm_df(lvm_df):
    """
    For lvm file, channels contain 64 sensors, X1 - X64, Y1 - Y64,
    Z1 - Z64, -> all meg channels;
    D1 - D10: all empty;
    AI 1 - AI 15: analog input, contains trigger channel;
    MUX_Counter1/2:
    
    """
    # find relevant rows in csv file

    for i in np.arange(lvm_df.shape[0]):

        str_list = lvm_df.iloc[i, 0].split(',')

        if str_list[0] == 'Delta_X':
            MEG_SFREQ = int(1 / float(str_list[1]))
            
        elif str_list[0] == 'X_Value':
            ch_names = str_list[1:-1]  # skip X_Value row name, skip COmment last col
            I_ROW_VALUESTART = i + 1
            break
    
    # extract full channel data
    temp_times = []
    temp_values = []

    for i in np.arange(I_ROW_VALUESTART, lvm_df.shape[0]):
        # get values from comma split string
        str_list = lvm_df.iloc[i, 0].split(',')
        # store timestamp, store values
        temp_times.append(float(str_list[0]))
        values = np.array([float(v) for v in str_list[1:]])
        # select channels present for this sub (based on config)
        temp_values.append(values)

    
    meg_data = np.array(temp_values)
    times = np.array(temp_times)

    # extract trigger signal
    trigger_ch = find_megtrigger_in_AI(megdata=meg_data, megchnames=ch_names,
                                       SYSTEM='PTB',)



    return MEG_SFREQ, times, meg_data, ch_names, trigger_ch



def select_and_store_axis_data(
    sub_config, ACQ, TASK, AX,
    meg_data=None, meg_times=None, meg_chnames=None,
    MEG_SFREQ=None, sensor_reg=None, sub_meta=None, 
    STORE: bool = False, LOAD: bool = True,
):

    assert AX in ['X', 'Y', 'Z'], f'AX not X Y Z ({AX})'

    if LOAD:
        path = os.path.join(get_onedrive_path('raw_data'),
                            sub_config["subject_id"], 'opm',
        )
        fname = f'{sub_config["subject_id"]}_{TASK}_{ACQ}_opm_{AX}channels_helmetIdxSorted.npy'
        # load if existing
        if fname in os.listdir(path):
            ax_data = np.load(os.path.join(path, fname))
            fname = f'{sub_config["subject_id"]}_{TASK}_{ACQ}_opm_timestamps.npy'
            meg_times = np.load(os.path.join(path, fname))

            return ax_data, meg_times

    # only select channels from given axis
    axcol_sel_bool = [
        c in sensor_reg[['X_ch', 'Y_ch', 'Z_ch']].values and AX in c
        for c in meg_chnames
    ]
    axis_ch_order = np.argsort(sensor_reg[f'idx_64order'])  # sort ax-channels to order of helmet sensors

    # select channels for specific axis and sort according to sensor reg
    ax_data = meg_data[:, axcol_sel_bool]
    ax_data = ax_data[:, axis_ch_order]


    # cut data based on task beginning and end to get real-task-timings
    ax_data, meg_times = cut_data_to_task_timing(ax_data, meg_times, sub_meta,
                                                 TASK=TASK, ACQ=ACQ, SFREQ=MEG_SFREQ,
                                                 ASSUME_TSTART_OPMt0=True,)

    if STORE:
        np.save(os.path.join(path, fname), ax_data, allow_pickle=True)
        fname = f'{sub_config["subject_id"]}_{TASK}_{ACQ}_opm_timestamps.npy'
        np.save(os.path.join(path, fname), meg_times, allow_pickle=True)


    return ax_data, meg_times





def load_sensor_coords(sub,):

    if len(sub) == 2: sub = f'sub-{sub}'

    subdata_dir = os.path.join(
        get_onedrive_path('source_data'),
        sub, 'opm',
    )
    # get geometry
    geo_folder = os.path.join(subdata_dir, 'geometry')
    geo_file = [f for f in os.listdir(geo_folder)
                if f.endswith('txt') and 'coord' in f.lower()][0]

    coord = pd.read_csv(os.path.join(geo_folder, geo_file),
                        header=None, index_col=3,)
    coord.index.name = 'sensor'
    coord.columns = ['X', 'Y', 'Z']
    coord['X'] = [v.split('ECHO: ')[1] for v in coord['X']]
        
    return coord


def get_ptb_sensor_id(string_id):
    """
    Sensor indices are from 1 to 64, first 64-X,
    then 6-Y, and 64-Z. 64-order is based on blocks
    of 8, sorted by alphabet: A1, A2, A3, ..., A8, B1, etc.
    """
    abc_group = string_id[0]
    number = int(string_id[1])

    if abc_group == 'A':
        sensor_idx = number
    
    elif abc_group == 'B':
        sensor_idx = number + 8
    
    elif abc_group == 'C':
        sensor_idx = number + 16
    
    else:
        raise ValueError('ABC sensor is outside A, C, C')

    return sensor_idx
    

def get_sensor_info(sub_config, STORE: bool = False):
    
    SUB = sub_config["subject_id"][-2:]
    
    idx_sensors = list(sub_config['meg_sensors'].keys())
    abcidx_sensors = list(sub_config['meg_sensors'].values())
    # sort
    sort_sensors = np.argsort(idx_sensors)
    idx_sensors = [idx_sensors[i] for i in sort_sensors]
    abcidx_sensors = [abcidx_sensors[i] for i in sort_sensors]
    sensor_orders = [get_ptb_sensor_id(v) for v in abcidx_sensors]

    sensor_reg = pd.DataFrame(index=[int(i) for i in idx_sensors])
    sensor_reg.index.name = 'helmet_idx'
    sensor_reg['idx_abc'] = abcidx_sensors
    sensor_reg['idx_64order'] = sensor_orders
    for a in ['X', 'Y', 'Z']:
        sensor_reg[f'{a}_ch'] = [f'{a}{i}' for i in sensor_orders]

    
    # add anatomical coordinates
    coord = load_sensor_coords(sub=SUB)

    for i in sensor_reg.index:
        for a in coord.keys():
            sensor_reg.loc[i, f'{a}_coord'] = coord.loc[i, a]
    
    if STORE:
        fname = f'opm_sensor_info_{sub_config["subject_id"]}.xlsx'
        fpath = os.path.join(
            get_onedrive_path('raw_data'),
            f'sub-{SUB}', 'opm', fname
        )
        sensor_reg.to_excel(fpath, header=True, index=True)


    return sensor_reg
    

def load_raw_opm_into_mne(meg_data, sub_config,
                          sfreq=None, ch_names=None, AX=None,):
    """
    requires availibilty of sensor coordniates in meters with
    nasion, peri-auricular left/right xyz in some coord system
    Input:
    """

    sensor_reg = get_sensor_info(sub_config,)

    geo_coord = load_sensor_coords(sub_config['subject_id'],)

    if not ch_names:
        ch_names = [f'{i}{AX}' for i in sensor_reg.index]

    if sub_config['subject_id'] == 'sub-03': sfreq = 750  # hardcoded from ptb

    # ensure correct direction of data
    if meg_data.shape[0] > meg_data.shape[1]:
        meg_data = meg_data.T

    assert len(ch_names) == meg_data.shape[0], 'ch names and data shape not matching'

    ### create info object
    ch_types = ["mag"] * meg_data.shape[0]

    ## MNE doesnt allow manual "MAG" position changing (bcs MEG usually doesnt need/allows this)
    # option 1: replace mag with eeg type
    # ch_types = ["eeg"] * meg_data.shape[0]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )
    
    # option 2: write directly into info object, including orientations
    coords = {c: sensor_reg.loc[i, ['X_coord', 'Y_coord', 'Z_coord']].values
              for c, i in zip(ch_names, sensor_reg.index)}
    
    for ch in info['chs']:
        name = ch['ch_name']
        if name in coords:
            # set xyz
            pos = np.array(coords[name], dtype=float)
            ch["loc"][:3] = pos
            # include orientation
            sensor = name.split(AX)[0]
            i_sensor = np.where(geo_coord.index == int(sensor))[0][0]
            if AX == 'Z': i_orient = 1
            else: i_orient = 2
            ch["loc"][3:6] = geo_coord.iloc[i_sensor + i_orient].values


    ### create raw object
    raw = mne.io.RawArray(meg_data, info)

    ### create montage

    # coords needs to be dict {chname: [x, y, z]}
    if sub_config["meg_loc_refsystem"] == "rp-lp-na":

        # coords = {c: sensor_reg.loc[i, ['X_coord', 'Y_coord', 'Z_coord']].values
        #           for c, i in zip(ch_names, sensor_reg.index)}
        montage = mne.channels.make_dig_montage(
            # ch_pos=coords,  # given via info["chs"]
            nasion=sub_config['xyz_NA'],
            lpa=sub_config['xyz_LP'],
            rpa=sub_config['xyz_RP'],
            coord_frame="head"
        )
        info.set_montage(montage)
    
    else:
        raise ValueError('create montage for new coord system')



    # if STORE_fif:
    #     raw.save("my_meg_raw.fif", overwrite=True)

    return raw