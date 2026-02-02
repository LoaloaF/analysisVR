import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import median_abs_deviation
from analytics_processing.modality_loading import session_modality_from_nas


def calculate_bias_histogram(session_dir, n_samples, ampl_ids=None,
                            chunk_duration_sec=0.2, 
                            downsample_after_sec=None, skip_interval_sec=None,
                            batch_size=50):
    """
    Calculate ADC value histogram for raw ephys traces over time.
    
    Args:
        session_dir: Path to the session directory
        n_samples: Total number of samples in the file (required)
        ampl_ids: Array of amplifier IDs (row indices) to use, or None for all channels
        chunk_duration_sec: Duration of each analysis chunk in seconds (default: 0.2s)
        downsample_after_sec: After this duration, start skipping intervals (optional)
        skip_interval_sec: Duration to skip between chunks after downsample_after_sec (optional)
        batch_size: Number of chunks to process at once (default: 50, speeds up I/O)
        
    Returns:
        aggr_histogram: Array of shape (n_chunks, 1024) - histogram of ADC values per chunk
        clipping_counts: Array of shape (n_chunks, 2) - (n_zero_clips, n_max_clips) per chunk
    """
    sampling_rate = 20_000  # Hz
    chunk_size = int(chunk_duration_sec * sampling_rate)
    
    if downsample_after_sec is not None:
        downsample_after = int(downsample_after_sec * sampling_rate)
        skip_interval = int(skip_interval_sec * sampling_rate) if skip_interval_sec else 0
    else:
        downsample_after = None
        skip_interval = 0
    
    aggr_histogram = []
    clipping_counts = []
    
    chunk_idx = 0
    chunk_num = 0
    
    total_duration_min = n_samples / (sampling_rate * 60)
    print(f"Processing {total_duration_min:.1f} minutes of data...")
    print(f"Total chunks to process: ~{n_samples // chunk_size}")
    
    while chunk_idx + chunk_size <= n_samples:
        # Collect batch of chunk indices
        batch_indices = []
        for _ in range(batch_size):
            if chunk_idx + chunk_size > n_samples:
                break
            
            batch_indices.append(chunk_idx)
            
            # Increment chunk_idx for next iteration
            chunk_idx += chunk_size
            
            # Skip intervals if past downsample threshold
            if downsample_after is not None and chunk_idx >= downsample_after:
                chunk_idx += skip_interval
        
        if not batch_indices:
            break
        
        # Read all chunks in batch at once using session_modality_from_nas
        batch_start = batch_indices[0]
        batch_end = batch_indices[-1] + chunk_size
        batch_data = session_modality_from_nas(session_dir, 'raw_ephys_traces', 
                                               columns=slice(batch_start, batch_end))
        
        # Filter channels if ampl_ids provided
        if ampl_ids is not None:
            batch_data = batch_data[ampl_ids, :]
        
        batch_data = batch_data.astype(np.int32)  # Convert to int32 for safe operations
        
        # Process each chunk in the batch
        for i, chunk_start_in_batch in enumerate(batch_indices):
            offset = chunk_start_in_batch - batch_start
            chunk_data = batch_data[:, offset:offset + chunk_size]
            
            # Compute mean per channel (using int32 to avoid overflow)
            mean_per_channel = chunk_data.sum(axis=1) // chunk_size
            
            # Count clipping (assuming 10-bit ADC: 0-1023 range)
            n_zero_clips = (mean_per_channel == 0).sum()
            n_max_clips = (mean_per_channel == 1023).sum()
            clipping_counts.append((n_zero_clips, n_max_clips))
            
            # Calculate histogram of ADC values across channels
            hist, _ = np.histogram(mean_per_channel, bins=1024, range=[0, 1024])
            aggr_histogram.append(hist)
            
            chunk_num += 1
        
        if chunk_num % 500 == 0:
            elapsed_time_min = batch_indices[-1] / (sampling_rate * 60)
            print(f"\rProcessed {chunk_num} chunks: {elapsed_time_min:.1f} / {total_duration_min:.1f} min", end='', flush=True)
    
    print(f"\nDone! Processed {chunk_num} chunks total.")
    return np.array(aggr_histogram), np.array(clipping_counts)


def render_bias_heatmap(aggr_histogram, chunk_duration_sec=0.2, 
                       xlim_start=None, title="ADC Bias Drift"):
    """
    Create heatmap visualization of ADC value distribution over time.
    
    Args:
        aggr_histogram: Array of shape (n_chunks, 1024) from calculate_bias_histogram
        chunk_duration_sec: Duration of each chunk in seconds
        xlim_start: Start time for x-axis limit (in chunk units), or None for full range
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Transpose for proper orientation (ADC values on y-axis, time on x-axis)
    img = ax.imshow(aggr_histogram.T, aspect='auto', vmax=1, origin='lower', cmap='gray_r')
    
    # Colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0 channels', '>1 channel'])
    cbar.set_label('Number of channels at ADC value')
    
    # Axis labels
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('ADC Value (0-1023)')
    ax.set_title(title)
    
    # X-axis: convert chunk index to minutes
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{(x * chunk_duration_sec / 60):.1f}" for x in xticks])
    
    if xlim_start is not None:
        ax.set_xlim(xlim_start, aggr_histogram.shape[0])
    else:
        ax.set_xlim(0, aggr_histogram.shape[0])
    
    ax.set_ylim(0, 1024)
    
    # Set specific y-axis ticks at powers of 2
    ax.set_yticks([0, 256, 512, 768, 1024])
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add horizontal dashed grid lines (thicker, behind plot)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='white', linewidth=1.5)
    ax.set_axisbelow(True)  # Draw grid behind image
    
    plt.tight_layout()
    return fig


def render_clipping_plot(clipping_counts, chunk_duration_sec=0.2,
                        title="Signal Clipping Over Time"):
    """
    Create stacked area plot showing proportion of channels at clipping values.
    
    Args:
        clipping_counts: Array of shape (n_chunks, 2) from calculate_bias_histogram
        chunk_duration_sec: Duration of each chunk in seconds
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure
    """
    n_channels = 1024  # Total possible channels (or use actual channel count)
    
    # Calculate proportions
    n_zero_clips = clipping_counts[:, 0]
    n_max_clips = clipping_counts[:, 1]
    n_normal = n_channels - (n_zero_clips + n_max_clips)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    time_axis = np.arange(clipping_counts.shape[0])
    
    ax.stackplot(time_axis, n_zero_clips, n_normal, n_max_clips,
                labels=['≤ 0 (clipped low)', 'Normal range', '≥ 1023 (clipped high)'],
                colors=['red', 'green', 'orange'], alpha=0.7)
    
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of channels')
    
    # X-axis: convert chunk index to seconds
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x * chunk_duration_sec)}" for x in xticks])
    
    ax.set_xlim(0, clipping_counts.shape[0])
    ax.set_ylim(0, n_channels)
    
    plt.tight_layout()
    return fig


def calculate_rms_noise(session_dir, n_samples, ampl_ids=None, n_chunks=10, 
                       chunk_duration_sec=1.0):
    """
    Calculate RMS noise for each channel after bandpass filtering.
    
    Samples multiple chunks across the recording, applies bandpass filter (300-5000 Hz)
    like spike sorting preprocessing, then calculates RMS per channel.
    
    Args:
        session_dir: Path to the session directory
        n_samples: Total number of samples in the file
        ampl_ids: Array of amplifier IDs (row indices) to use, or None for all channels
        n_chunks: Number of chunks to sample across the recording (default: 10)
        chunk_duration_sec: Duration of each chunk in seconds (default: 1.0s)
        
    Returns:
        rms_per_channel: Array of shape (n_channels,) - RMS noise averaged over chunks
        rms_all_chunks: Array of shape (n_chunks, n_channels) - RMS per chunk for violin plot
    """
    sampling_rate = 20_000  # Hz
    chunk_size = int(chunk_duration_sec * sampling_rate)
    
    # Design bandpass filter (Butterworth, order 4, 300-5000 Hz)
    nyquist = sampling_rate / 2
    low_freq = 300 / nyquist
    high_freq = 5000 / nyquist
    sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', output='sos')
    
    # Sample chunk positions evenly across the recording
    chunk_positions = np.linspace(0, n_samples - chunk_size, n_chunks, dtype=int)
    
    print(f"Calculating RMS noise from {n_chunks} chunks...")
    
    rms_all_chunks = []
    
    for i, chunk_start in enumerate(chunk_positions):
        chunk_end = chunk_start + chunk_size
        
        # Read chunk
        chunk_data = session_modality_from_nas(session_dir, 'raw_ephys_traces',
                                              columns=slice(chunk_start, chunk_end))
        
        # Filter channels if ampl_ids provided
        if ampl_ids is not None:
            chunk_data = chunk_data[ampl_ids, :]
        # Apply bandpass filter to each channel
        filtered_data = np.zeros_like(chunk_data, dtype=np.float32)
        for ch in range(chunk_data.shape[0]):
            filtered_data[ch, :] = signal.sosfiltfilt(sos, chunk_data[ch, :])
        
        # Calculate RMS per channel for this chunk
        rms = np.sqrt(np.mean(filtered_data**2, axis=1))
        rms_all_chunks.append(rms)
        
        print(f"\rProcessed chunk {i+1}/{n_chunks}", end='', flush=True)
    
    print("\nDone!")
    
    rms_all_chunks = np.array(rms_all_chunks)  # Shape: (n_chunks, n_channels)
    rms_per_channel = np.mean(rms_all_chunks, axis=0)  # Average over chunks
    
    return rms_per_channel, rms_all_chunks


def render_rms_violin(rms_all_chunks_1, rms_all_chunks_2=None, 
                     color1='#5A8FC6', color2='#CCCCCC',
                     label1='Session 1', label2='Session 2',
                     title="RMS Noise Distribution"):
    """
    Create violin plot of RMS noise distribution across channels.
    Can plot one or two sessions side by side.
    
    Args:
        rms_all_chunks_1: Array of shape (n_chunks, n_channels) from calculate_rms_noise
        rms_all_chunks_2: Optional second array for comparison (default: None)
        color1: Color for first violin plot (default: '#5A8FC6' - blue)
        color2: Color for second violin plot (default: '#CCCCCC' - light gray)
        label1: Label for first session (default: 'Session 1')
        label2: Label for second session (default: 'Session 2')
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Determine if we're plotting one or two sessions
    single_session = rms_all_chunks_2 is None
    
    fig, ax = plt.subplots(figsize=(4 if single_session else 5, 8))
    
    # Flatten all RMS values into distributions
    rms_flat_1 = rms_all_chunks_1.flatten()
    
    if single_session:
        positions = [1]
        data = [rms_flat_1]
        colors = [color1]
        labels = [label1]
    else:
        rms_flat_2 = rms_all_chunks_2.flatten()
        positions = [1, 2]
        data = [rms_flat_1, rms_flat_2]
        colors = [color1, color2]
        labels = [label1, label2]
    
    # Create violin plots without mean/median/extrema
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                         showmeans=False, showmedians=False, showextrema=False)
    
    # Style the violins with uniform colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(colors[i])
        pc.set_alpha(1.0)
    
    # Calculate and display statistics
    stats_y_pos = 0.97
    for i, (rms_flat, label, color) in enumerate(zip(data, labels, colors)):
        mean_rms = np.mean(rms_flat)
        median_rms = np.median(rms_flat)
        std_rms = np.std(rms_flat)
        
        stats_text = f"{label}:\nMean: {mean_rms:.2f} µV\nMedian: {median_rms:.2f} µV\nStd: {std_rms:.2f} µV"
        
        x_pos = 0.98 if single_session else (0.48 if i == 0 else 0.98)
        ax.text(x_pos, stats_y_pos - (i * 0.15 if single_session else 0), stats_text, 
               transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
    
    ax.set_ylabel('RMS Noise (µV)')
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    return fig


def render_sample_traces(session_dir, n_samples, ampl_ids=None, n_channels_plot=5, 
                        chunk_duration_sec=0.1, n_chunks=3):
    """
    Plot sample filtered traces from random channels to visualize noise characteristics.
    
    Args:
        session_dir: Path to the session directory
        n_samples: Total number of samples in the file
        ampl_ids: Array of amplifier IDs (row indices) to use, or None for all channels
        n_channels_plot: Number of random channels to plot (default: 5)
        chunk_duration_sec: Duration of each trace to plot in seconds (default: 0.1s)
        n_chunks: Number of time chunks to sample (default: 3)
        
    Returns:
        matplotlib figure
    """
    sampling_rate = 20_000  # Hz
    chunk_size = int(chunk_duration_sec * sampling_rate)
    
    # Design bandpass filter (same as RMS calculation)
    nyquist = sampling_rate / 2
    low_freq = 300 / nyquist
    high_freq = 5000 / nyquist
    sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', output='sos')
    
    # Sample chunk positions across the recording
    chunk_positions = np.linspace(0, n_samples - chunk_size, n_chunks, dtype=int)
    
    fig, axes = plt.subplots(n_channels_plot, 1, figsize=(10, 8), sharex=True)
    if n_channels_plot == 1:
        axes = [axes]
    
    # Read first chunk to determine number of channels
    chunk_data = session_modality_from_nas(session_dir, 'raw_ephys_traces',
                                          columns=slice(chunk_positions[0], 
                                                       chunk_positions[0] + chunk_size))
    if ampl_ids is not None:
        chunk_data = chunk_data[ampl_ids, :]
    print(chunk_data)
    n_channels_total = chunk_data.shape[0]
    
    # Select random channels to plot
    selected_channels = np.random.choice(n_channels_total, 
                                        size=min(n_channels_plot, n_channels_total), 
                                        replace=False)
    
    # Time axis in milliseconds
    time_ms = np.arange(chunk_size) / sampling_rate * 1000
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_chunks))
    
    for ch_idx, ch in enumerate(selected_channels):
        print(ch)
        ax = axes[ch_idx]
        
        all_traces = []
        
        for chunk_i, chunk_start in enumerate(chunk_positions):
            chunk_end = chunk_start + chunk_size
            
            # Read chunk
            chunk_data = session_modality_from_nas(session_dir, 'raw_ephys_traces',
                                                  columns=slice(chunk_start, chunk_end))
            
            if ampl_ids is not None:
                chunk_data = chunk_data[ampl_ids, :]
                
            
            
            # Apply bandpass filter
            filtered_trace = signal.sosfiltfilt(sos, chunk_data[ch, :])
            print(chunk_data[ch, :])
            print(filtered_trace)
            # filtered_trace = chunk_data[ch, :].astype(np.float32)
            all_traces.append(filtered_trace)
            
            # Plot trace
            ax.plot(time_ms, filtered_trace, color=colors[chunk_i], 
                   alpha=0.7, linewidth=0.8, label=f'Chunk {chunk_i+1}')
        
        # Calculate and plot std bands
        all_traces = np.array(all_traces)
        mean_trace = np.mean(all_traces, axis=0)
        std_trace = np.std(all_traces, axis=0)
        global_std = np.std(all_traces)
        global_mad = median_abs_deviation(all_traces.flatten(), scale='normal')
        
        # Plot std bands
        ax.axhline(global_std, color='red', linestyle='--', linewidth=1, 
                  label=f'±1 STD ({global_std:.1f} µV)')
        ax.axhline(-global_std, color='red', linestyle='--', linewidth=1)
        
        # Plot MAD bands
        ax.axhline(global_mad, color='orange', linestyle=':', linewidth=1.5, 
                  label=f'±1 MAD ({global_mad:.1f} µV)')
        ax.axhline(-global_mad, color='orange', linestyle=':', linewidth=1.5)
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Styling
        ax.set_ylabel(f'Ch {ch}\n(µV)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if ch_idx == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=False)
    
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Sample Filtered Traces (300-5000 Hz Bandpass)', fontsize=12)
    plt.tight_layout()
    
    return fig


def render_channel_stack_comparison(session_dir_tethered, session_dir_logger,
                                    n_samples_tethered, n_samples_logger,
                                    ampl_ids_tethered, ampl_ids_logger,
                                    chunk_idx=0, start_channel=0, n_channels=60,
                                    chunk_duration_sec=0.5):
    """
    Plot stacked traces from many channels for a single time chunk, comparing tethered vs logger.
    
    Each channel is plotted with a y-offset. Two panels show tethered (left) and logger (right).
    RMS values are annotated for each trace.
    
    Args:
        session_dir_tethered: Path to tethered session directory
        session_dir_logger: Path to logger session directory
        n_samples_tethered: Total samples in tethered session
        n_samples_logger: Total samples in logger session
        ampl_ids_tethered: Amplifier IDs for tethered session
        ampl_ids_logger: Amplifier IDs for logger session
        chunk_idx: Which chunk to display (0=start, 1, 2, etc.)
        start_channel: Starting channel index (e.g., 60 to plot channels 60-120)
        n_channels: Number of channels to plot (default: 60)
        chunk_duration_sec: Duration of chunk in seconds (default: 0.5s)
        
    Returns:
        matplotlib figure
    """
    sampling_rate = 20_000  # Hz
    chunk_size = int(chunk_duration_sec * sampling_rate)
    
    # Design bandpass filter
    nyquist = sampling_rate / 2
    low_freq = 300 / nyquist
    high_freq = 5000 / nyquist
    sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', output='sos')
    
    # Determine chunk position
    chunk_start_tethered = min(chunk_idx * chunk_size, n_samples_tethered - chunk_size)
    chunk_end_tethered = chunk_start_tethered + chunk_size
    
    chunk_start_logger = min(chunk_idx * chunk_size, n_samples_logger - chunk_size)
    chunk_end_logger = chunk_start_logger + chunk_size
    
    # Read data for both sessions
    data_tethered = session_modality_from_nas(session_dir_tethered, 'raw_ephys_traces',
                                             columns=slice(chunk_start_tethered, chunk_end_tethered))
    data_logger = session_modality_from_nas(session_dir_logger, 'raw_ephys_traces',
                                           columns=slice(chunk_start_logger, chunk_end_logger))
    
    # Apply channel filtering
    if ampl_ids_tethered is not None:
        data_tethered = data_tethered[ampl_ids_tethered, :]
    if ampl_ids_logger is not None:
        data_logger = data_logger[ampl_ids_logger, :]
    
    # Select channel range
    end_channel = min(start_channel + n_channels, data_tethered.shape[0], data_logger.shape[0])
    data_tethered = data_tethered[start_channel:end_channel, :]
    data_logger = data_logger[start_channel:end_channel, :]
    n_channels_actual = data_tethered.shape[0]
    
    # Time axis in milliseconds
    time_ms = np.arange(chunk_size) / sampling_rate * 1000
    
    # Create figure with two panels
    fig, (ax_tethered, ax_logger) = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    
    # Y-offset for stacking traces (adaptive based on data range)
    y_offset = 100  # µV spacing between channels
    
    # Process and plot tethered session
    for ch_idx in range(n_channels_actual):
        # Apply bandpass filter
        filtered_trace = signal.sosfiltfilt(sos, data_tethered[ch_idx, :])
        # filtered_trace = data_tethered[ch_idx, :]
        
        # Calculate RMS
        rms = np.sqrt(np.mean(filtered_trace**2))
        
        # Plot with y-offset
        y_offset_val = ch_idx * y_offset
        ax_tethered.plot(time_ms, filtered_trace + y_offset_val, 
                        color='#5A8FC6', alpha=0.3, linewidth=0.5)
        
        # Annotate RMS every 10th channel
        if ch_idx % 10 == 0:
            ax_tethered.text(time_ms[-1] + 2, y_offset_val, f'{rms:.1f}', 
                           fontsize=7, va='center', color='#5A8FC6')
    
    # Process and plot logger session
    for ch_idx in range(n_channels_actual):
        # Apply bandpass filter
        filtered_trace = signal.sosfiltfilt(sos, data_logger[ch_idx, :])
        # filtered_trace = data_logger[ch_idx, :]
        
        # Calculate RMS
        rms = np.sqrt(np.mean(filtered_trace**2))
        
        # Plot with y-offset
        y_offset_val = ch_idx * y_offset
        ax_logger.plot(time_ms, filtered_trace + y_offset_val, 
                      color='#888888', alpha=0.3, linewidth=0.5)
        
        # Annotate RMS every 10th channel
        if ch_idx % 10 == 0:
            ax_logger.text(time_ms[-1] + 2, y_offset_val, f'{rms:.1f}', 
                          fontsize=7, va='center', color='#888888')
    
    # Styling
    ax_tethered.set_xlabel('Time (ms)')
    ax_tethered.set_ylabel(f'Channel (offset) | Channels {start_channel}-{end_channel-1}')
    ax_tethered.set_title(f'Tethered Session\nChunk {chunk_idx} ({chunk_duration_sec}s)')
    ax_tethered.spines['top'].set_visible(False)
    ax_tethered.spines['right'].set_visible(False)
    ax_tethered.grid(True, linestyle='--', alpha=0.2)
    
    ax_logger.set_xlabel('Time (ms)')
    ax_logger.set_title(f'Logger Session\nChunk {chunk_idx} ({chunk_duration_sec}s)')
    ax_logger.spines['top'].set_visible(False)
    ax_logger.spines['right'].set_visible(False)
    ax_logger.grid(True, linestyle='--', alpha=0.2)
    
    # Add RMS annotation label
    fig.text(0.98, 0.5, 'RMS (µV)', rotation=270, va='center', fontsize=8)
    
    plt.tight_layout()
    return fig
