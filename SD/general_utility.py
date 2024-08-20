import pandas as pd
import numpy as np
import mne

def label_epochs(num_epochs, epoch_length, annotations_df_bi):
    """
    Label each epoch based on whether it overlaps with seizure intervals.
    
    Parameters:
    num_epochs (int): Total number of epochs.
    epoch_length (int): Length of each epoch in seconds.
    annotations_df_bi (pd.DataFrame): DataFrame containing start and stop times, and labels 
                                      for intervals (e.g., 'seiz' for seizure).
    
    Returns:
    list: Labels for each epoch (1 for seizure, 0 for no seizure).
    """
    # Validate inputs
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ValueError("num_epochs should be a positive integer.")
    
    if not isinstance(epoch_length, int) or epoch_length <= 0:
        raise ValueError("epoch_length should be a positive integer.")
    
    if not isinstance(annotations_df_bi, pd.DataFrame):
        raise TypeError("annotations_df_bi should be a pandas DataFrame.")
    
    required_columns = {'start_time', 'stop_time', 'label'}
    if not required_columns.issubset(annotations_df_bi.columns):
        missing_cols = required_columns - set(annotations_df_bi.columns)
        raise ValueError(f"annotations_df_bi is missing required columns: {', '.join(missing_cols)}")

    # Initialize labels array
    labels = np.zeros(num_epochs, dtype=int)
    
    try:
        start_times = annotations_df_bi['start_time'].values.astype(int)
        stop_times = annotations_df_bi['stop_time'].values.astype(int)
        labels_df = annotations_df_bi['label'].values
    except ValueError as e:
        raise ValueError("Error in converting start_time or stop_time to integers.") from e
    
    # Label epochs based on overlaps with seizure intervals
    for i in range(num_epochs):
        epoch_start = i * epoch_length
        epoch_end = (i + 1) * epoch_length
        
        # Check each interval
        for start, stop, label in zip(start_times, stop_times, labels_df):
            if not isinstance(label, str):
                raise ValueError("Labels in annotations_df_bi['label'] should be strings.")
            
            if label == 'seiz' and epoch_start < stop and epoch_end > start:
                labels[i] = 1
                break

    return labels

def read_edf_file(file_path, annotations_path_bi, epoch_length=30, epoch_overlap=0):
    """
    Read data from an EDF file and the lables from CSV file , segmenting it into epochs.

    Parameters:
    - file_path: str, path to the EDF file.
    - annotations_path_bi: str, path to the CSV file containing annotations.
    - epoch_length: int, length of each epoch in seconds (default is 30 seconds).
    - epoch_overlap: int, overlap between epochs in seconds (default is 0 seconds).

    Returns:
    - epochs number 
    - EEG data 
    - EEG Channel names, string
    - sampling frequency 
    - epochs_labels: array (1, n_epochs) containing the labels for each epoch.
    """
    
    # Load annotations
    annotations_df_bi = pd.read_csv(annotations_path_bi, skiprows=5)
    # Read raw EEG data
    raw = mne.io.read_raw_edf(file_path, preload=True)
    EEG_data = raw.get_data()  
    fs = raw.info['sfreq']  # Sampling frequency
    n_samples_per_epoch = int(fs * epoch_length)
    channel_names = [name.replace('EEG ', '') for name in raw.ch_names]
    n_epochs = int((EEG_data.shape[1] - epoch_overlap) / (n_samples_per_epoch - epoch_overlap))
    # Close raw data
    raw.close()
    labels = label_epochs(n_epochs, epoch_length, annotations_df_bi)
    
    return n_epochs, EEG_data, labels, channel_names, fs