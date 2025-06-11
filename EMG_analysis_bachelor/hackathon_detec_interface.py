import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog,font, Button, Label
import threading
import tkinter.font as tkFont

import scipy as sp
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from EMG_analysis_bachelor.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, notched, \
    filtered, create_df, envelope


# import data
#EMG_ACC_data = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/"
 #                                  "PTB_measurement_14.04/Bonato_Federico_2025-04-14_13-02-56.cnt", preload=True)
EMG_only_test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/hackathon/"
                                   "test_first_2025-04-07_14-10-04_forearm_upperSide.cnt", preload=True)
# custom_order_names = ["BIP6", "BIP1", "BIP2", "BIP7", "BIP8"]
# custom_order_names = ["BIP7", "BIP8"]
location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x"}
ACC = ["BIP6", "BIP1", "BIP2"]
# emg_idx, acc_idx = get_ch_indices(custom_order_names, EMG, ACC)

#def plot_movement_detection():
#    plt.figure(figsize=(10, 4))
#    plt.plot(times, signal, "b", label="Envelope")
#    plt.axhline(threshold, color="r", linestyle="--", label="Threshold")
#    for t in start_times:
#        plt.axvline(t, color="g", linestyle="--", label="Start" if t == start_times[0] else "")
#    for t in end_times:
#        plt.axvline(t, color="k", linestyle="--", label="End" if t == end_times[0] else "")
#    plt.legend()
#    plt.xlabel("Time (s)")
#    plt.ylabel("Envelope / Movement")
#    plt.title("Contraction Detection")
#    canvas.draw()


#mne.Epochs
#als df speichern? csv

emg = ["BIP7"]
emg_1 = ["BIP7"]
#ACC = ["BIP6", "BIP1", "BIP2"]

# --- Initial Window ---
m = tk.Tk()
m.title("EMG Movement Detection")
m.geometry("700x700")

second_m = None
progress_var = None
progress_bar = None
status_label = None

custom_font = tkFont.Font(family="Arial", size=25)
custom_font1 = tkFont.Font(family="Arial", size=16)

# ---------- Funktionen ----------

def processing():
    global emg_envelope
    def update_progress(step, description):
        progress_var.set(step * 16.66)
        status_label.config(text=description)
        second_m.update_idletasks()

    try:
        update_progress(1, "get raw data...")
        data, times = EMG_only_test[emg, :]
        data[0] *= -1
        raw_df = create_df(data, emg, times)

        update_progress(3, "notch-filtering...")
        notched_df = notched(raw_df, emg, times)

        update_progress(4, "bandpass-filtering...")
        notched_and_filtered_df = filtered(notched_df, emg, ACC, emg_1, times)

        update_progress(5, "rectifying...")
        rectified_df = abs(notched_and_filtered_df)

        update_progress(6, "normalising...")
        emg_normalized_df = normalize_emg(rectified_df, emg, emg_1, times)

        update_progress(6.5, "getting envelope...")
        emg_envelope = envelope(emg_normalized_df, emg, emg_1, times, 3)

        status_label.config(text="processing done!", fg="green")
    except Exception as e:
        status_label.config(text=f"Error: {str(e)}", fg="red")
    Button(second_m, text="detect and plot movement", command=movement_detection).pack(pady=10)

def movement_detection():
   global times, thresh, task_signal, start_times, end_times, contractions
   task_start_end = [1000, 15000] # pretending that that was trigger signal of start and end of task
   forearm_envelope = emg_envelope["BIP7"]
   task_signal = forearm_envelope[task_start_end[0]:task_start_end[1]]

   sfreq = 1000
   times = np.arange(len(task_signal)) / sfreq

   thresh = np.mean(task_signal[:1000]) + np.max(zscore(task_signal[:1000])) * np.std(task_signal[:1000])
   movement = task_signal > thresh
   movement = movement.to_numpy()

   starts = []
   ends = []

   for i in range(1, len(movement)):
       if not movement[i-1] and movement[i]:
           starts.append(i)
       elif movement[i-1] and not movement[i]:
           ends.append(i)

   # if recording starts or ends with contraction
   if movement[0]:
       starts = [0] + starts
   if movement[-1]:
       ends = ends + [len(movement)-1]

   # to seconds
   start_times = np.array(starts) / sfreq
   end_times = np.array(ends) / sfreq

   contractions = []
   for i in range(0, len(start_times)):
       if len(start_times) == len(end_times):
           contraction = [start_times[i], end_times[i]]
           contractions.append(contraction)
       else:
           print("no equal amount of starts and ends")

   #Button(second_m, text="plot signal", command=plot_movement_detection).pack()
   second_m.withdraw()
   plot_movement_detection()

def plot_movement_detection():
    global third_m, canvas, task_signal, thresh, times, start_times, end_times
    third_m = tk.Toplevel()
    third_m.title("EMG Plotting")
    third_m.geometry("600x400")
    Label(third_m, text="Movement Detection", font=custom_font).pack()

    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master = third_m)
    canvas.get_tk_widget().pack(side="top",fill='both',expand=True)

    toolbar = NavigationToolbar2Tk(canvas, third_m, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(anchor="w", side=tk.RIGHT, fill=tk.X)

    plt.plot(times, task_signal, "b", label="Envelope")
    plt.axhline(thresh, color="r", linestyle="--", label="Threshold")
    for t in start_times:
        plt.axvline(t, color="g", linestyle="--", label="Start" if t == start_times[0] else "")
    for t in end_times:
        plt.axvline(t, color="k", linestyle="--", label="End" if t == end_times[0] else "")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope / Movement")
    plt.title("Contraction Detection")
    canvas.draw()
    save_contractions()

def save_contractions():
    global third_m, contractions, start_times
    Label(third_m, text=f"{len(start_times)} Contractions detected: {contractions}", font=custom_font1).pack()

def start_processing():
    progress_var.set(0)
    status_label.config(text="start processing...", fg="black")
    threading.Thread(target=processing, daemon=True).start()

def select_folder():
    global directory
    directory = filedialog.askdirectory(title="Select Participant Folder")
    if directory:
        folder_label.config(text=f"Selected folder: {directory}")
        if not continue_button.winfo_ismapped():
            continue_button.pack(side=tk.RIGHT, padx=10, pady=10, anchor=tk.SE)
    else:
        folder_label.config(text="no folder selected.")

def clear_window():
    for widget in m.winfo_children():
        widget.destroy()
    show_channel_selection()

def show_channel_selection():
    global value_inside, channel_amount
    tk.Label(m, text="How many channels did you use?", font=custom_font1).pack(pady=20)
    options_list = [str(i) for i in range(1, 13)]
    value_inside = tk.StringVar(m)
    value_inside.set("1")
    tk.OptionMenu(m, value_inside, *options_list).pack()
    tk.Button(m, text='Submit', command=save_channel_option).pack(pady=10)

def save_channel_option():
    global channel_amount
    channel_amount = int(value_inside.get())
    for widget in m.winfo_children():
        widget.destroy()
    put_in_location()

def put_in_location():
    tk.Label(m, text="Type in your channel-location assignment", font=custom_font1).pack(pady=10)
    tk.Label(m, text="Format : BIP7, right forearm").pack()
    global entries
    entries = []
    for i in range(channel_amount):
        frame = tk.Frame(m)
        frame.pack(pady=5)
        tk.Label(frame, text=f"Channel {i + 1}:").pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=30)
        entry.pack(side=tk.LEFT)
        entries.append(entry)
    tk.Button(m, text='Save Locations', command=save_channel_locations, font=custom_font1).pack(pady=20)

def save_channel_locations():
    global channel_locations
    channel_locations = {}
    for entry in entries:
        input_text = entry.get().strip()
        if "," in input_text:
            channel, location = [x.strip() for x in input_text.split(",", 1)]
            channel_locations[channel] = location
        else:
            error_label = tk.Label(m, text="Error: Use 'channel name, location'", fg="red")
            error_label.pack(pady=5)
            entry.delete(0, tk.END)
            entry.focus_set()
            entry.config(bg="#FFF0F0")
            return

    tk.Label(m, text=f"Saved: {channel_locations}").pack()
    m.withdraw()
    create_processing_window()

def create_processing_window():
    global second_m, progress_var, progress_bar, status_label
    second_m = tk.Toplevel()
    second_m.title("EMG Processing")
    second_m.geometry("600x400")

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(second_m, variable=progress_var, maximum=100, length=400)
    progress_bar.pack(pady=20)

    status_label = tk.Label(second_m, text="ready for processing...")
    status_label.pack()

    tk.Button(second_m, text="start processing", command=start_processing).pack(pady=10)

# ---------- Startfenster ----------
tk.Label(m, text="Select the folder of your participant", font=custom_font1).pack(pady=20)
directory = ""
tk.Button(m, text="Browse", command=select_folder, font=custom_font1).pack(pady=10)
folder_label = tk.Label(m, text="", font=tkFont.Font(family="Arial", size=10))
folder_label.pack(pady=20)

continue_button = tk.Button(m, text="Continue", command=clear_window, font=custom_font1)

channel_amount = None
channel_locations = {}
entries = []

m.mainloop()

