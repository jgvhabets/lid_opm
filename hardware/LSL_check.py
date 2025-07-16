import pyxdf

data, header = pyxdf.load_xdf("C:/Users/User/Downloads/sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
print(f"Number of streams: {len(data)}")
for stream in data:
    print(f"Stream '{stream['info']['name'][0]}' has {len(stream['time_series'])} samples.")