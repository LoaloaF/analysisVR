#!/bin/bash
# filepath: /home/vrmaster/Projects/VirtualReality/analysisVR/run_svm_parallel.sh

sessions=(
    "2024-11-14_15-01_rYL006_P1100_LinearTrackStop_30min"
    "2024-11-14_16-40_rYL006_P1100_LinearTrackStop_21min"
    "2024-11-15_15-48_rYL006_P1100_LinearTrackStop_35min"
    "2024-11-20_17-46_rYL006_P1100_LinearTrackStop_22min"
    "2024-11-21_17-22_rYL006_P1100_LinearTrackStop_25min"
    "2024-11-22_16-01_rYL006_P1100_LinearTrackStop_24min"
    "2024-11-25_16-25_rYL006_P1100_LinearTrackStop_18min"
    "2024-11-26_16-39_rYL006_P1100_LinearTrackStop_25min"
    "2024-11-27_16-11_rYL006_P1100_LinearTrackStop_31min"
    "2024-11-28_17-41_rYL006_P1100_LinearTrackStop_30min"
    "2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min"
    "2024-12-02_16-09_rYL006_P1100_LinearTrackStop_28min"
    "2024-12-03_16-23_rYL006_P1100_LinearTrackStop_30min"
    "2024-12-04_18-06_rYL006_P1100_LinearTrackStop_30min"
    "2024-12-06_16-49_rYL006_P1100_LinearTrackStop_25min"
    "2024-12-09_17-45_rYL006_P1100_LinearTrackStop_26min"
    "2024-12-10_17-20_rYL006_P1100_LinearTrackStop_30min"
    "2024-12-11_17-42_rYL006_P1100_LinearTrackStop_30min"
    "2024-12-12_16-13_rYL006_P1100_LinearTrackStop_26min"
    "2024-12-13_17-10_rYL006_P1100_LinearTrackStop_30min"
    "2025-01-14_18-08_rYL006_P1100_LinearTrackStop_30min"
    "2025-01-15_17-18_rYL006_P1100_LinearTrackStop_30min"
    "2025-01-16_17-47_rYL006_P1100_LinearTrackStop_30min"
    "2025-01-17_16-55_rYL006_P1100_LinearTrackStop_30min"
    "2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min"
    "2025-01-22_17-51_rYL006_P1100_LinearTrackStop_5min"
    "2025-01-23_16-48_rYL006_P1100_LinearTrackStop_42min"
    "2025-01-24_12-24_rYL006_P1100_LinearTrackStop_41min"
    "2025-01-24_19-37_rYL006_P1100_LinearTrackStop_40min"
    "2025-01-25_10-55_rYL006_P1100_LinearTrackStop_60min"
    "2025-01-25_21-29_rYL006_P1100_LinearTrackStop_61min"
    "2025-01-26_13-45_rYL006_P1100_LinearTrackStop_67min"
    "2025-01-26_21-48_rYL006_P1100_LinearTrackStop_55min"
    "2025-01-27_13-39_rYL006_P1100_LinearTrackStop_73min"
)

# Make the script executable with: chmod +x run_svm_parallel.sh
for session in "${sessions[@]}"; do
    echo "Starting process for session: $session"
    python run_pipeline.py SVMCueOutcomeChoicePred --animal_ids 6 --mode recompute --paradigm_ids 1100 --session_names "$session" &
done

# Wait for all background processes to complete
wait
echo "All processes completed"