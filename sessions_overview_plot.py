import pandas as pd
from analysis_core import init_analysis
from data_loading.paradigm_loading import get_paradigm_modality
import matplotlib.pyplot as plt

from analysis_core import PARADIGM_COLORS, PARADIGM_VISIBLE_NAMES

def get_all_sessions_metadata(excl_paradigms=[]):
    paradigms = [p for p in range(0,1200,100) if p not in excl_paradigms]
    
    all_data = []
    for p in paradigms:
        animal_parsing_kwargs={"skip_sessions":["2024-07-29_16-58_rYL001_P0800_LinearTrack_41min", # bell vel doesn't match up with unity frames
                                                "2024-07-26_14-57_rYL003_P0800_LinearTrack_27min", # trial_id 1-5 duplicated
                                                "2024-08-09_18-20_rYL003_P0800_LinearTrack_21min", # has only 1 trial
        ]}
        session_parsing_kwargs={"dict2pandas":True}
        modality_parsing_kwargs={"columns": ['start_time', 'duration']}

        data = get_paradigm_modality(paradigm_id=p, modality='metadata', cache='from', 
                                    animal_parsing_kwargs=animal_parsing_kwargs,
                                    session_parsing_kwargs=session_parsing_kwargs,
                                    modality_parsing_kwargs=modality_parsing_kwargs,
                                )
        all_data.append(data)
    all_data = pd.concat(all_data)
    return all_data
    # pd.to_pickle(all_data, "all_sessions_metadata.pkl")
    # all_data.to_excel("all_sessions_metadata.xlsx")


def rewrite_metadata(metadata):
    print(metadata)
    metadata.to_excel("./assets/all_sessions_metadata.xlsx")
    metadata.sort_values(["start_time"], inplace=True)
    print(metadata)
    metadata.to_excel("./assets/all_sessions_metadata_sorted.xlsx")

def draw_all_sessions(metadata):
    print('\n\n\n')
    metadata = metadata.droplevel((3, 4))
    start_time_dt = pd.DatetimeIndex(pd.to_datetime(metadata['start_time'], format="%Y-%m-%d_%H-%M"))
    
    metadata.loc[:, "session_start"] = start_time_dt
    metadata.loc[:, "session_length_min"] = pd.to_timedelta(metadata['duration_minutes'], unit='m')
    metadata.loc[:, "session_length_min"].fillna(pd.to_timedelta(15, unit='m'), inplace=True)
    metadata.loc[:, "session_stop"] = start_time_dt + metadata['session_length_min']
    metadata = metadata.iloc[:, -3:]  # Keep only the relevant columns
    metadata.sort_index(level=('animal_id'), inplace=True)
    print(metadata)

    animals = metadata.index.get_level_values("animal_id").unique().tolist()

    # Setup the plot
    fig, axes = plt.subplots(figsize=(19, 3), nrows=len(animals), sharex=True)
    fig.subplots_adjust(hspace=0, left=0.05, right=0.96, top=0.98, bottom=0.1)

    
    for idx, row in metadata.iterrows():
        parad, anim = idx[0], idx[1]
        if anim in (7,5) and parad == 800:
            parad = 801
        
        # Extract day and time
        day = row['session_start'].normalize()  # Start of the day (datetime.date)
        start_time = row['session_start'].time()  # Time of day
        stop_time = row['session_stop'].time()  # Time of day
        
        # Convert time to seconds since start of day for plotting
        start_seconds = pd.to_timedelta(start_time.hour, unit='h') + pd.to_timedelta(start_time.minute, unit='m')
        stop_seconds = pd.to_timedelta(stop_time.hour, unit='h') + pd.to_timedelta(stop_time.minute, unit='m')
        
        # Select the appropriate axis for this animal
        ax = axes[animals.index(anim)]
        
        # Plot the session as a vertical line spanning start to stop
        x, y = [day, day], [start_seconds.total_seconds(), stop_seconds.total_seconds()]
        ax.plot(x, y, 'k', lw=4, color=PARADIGM_COLORS[parad])

        ax.set_ylim(21*3600, 9*3600)  # 0 to 24 hours in seconds
        ax.tick_params(axis='y', left=False, labelleft=False)
        
    
    for i, ax in enumerate(axes):
        ax.set_ylabel(f"rat {animals[i]:03}", rotation=0, labelpad=10, 
                      fontsize=12, ha='right', va='center')
        [ax.spines[spine].set_visible(False) for spine in ['top', 'right', 'left']]
        ax.spines['bottom'].set_linewidth(1)
        if i == 0:
            ax.spines['top'].set_visible(True)
            ax.tick_params(axis='y', left=False, right=True, labelright=True, 
                           labelleft=False)
            ax.set_yticks([10*3600, 15*3600, 20*3600])
            ax.set_yticklabels(["10am", "3pm", "8pm"], fontsize=9)
            
    
    for p in [0,100,200,400]:
        axes[-1].plot([], [], 'k', lw=4, color=PARADIGM_COLORS[p], 
                 label=PARADIGM_VISIBLE_NAMES[p])
    l1 = axes[-1].legend(loc='lower left', fontsize=9, ncols=2 )
    axes[-1].add_artist(l1)

    # Second set of paradigms
    for p in [500, 900, 1000, 800, 801, 1100]:
        axes[2].plot([], [], 'k', lw=4, color=PARADIGM_COLORS[p], 
                     label=PARADIGM_VISIBLE_NAMES[p])
    legend2 = axes[2].legend(loc='upper right', fontsize=9, ncols=2, 
                             bbox_to_anchor=(.95, .99), bbox_transform=fig.transFigure)
    axes[2].add_artist(legend2)  # Add the second legend 
    
    
    
    plt.savefig("./plots/all_sessions_timeline.svg")
    plt.show()
    
    

if __name__ == "__main__":
    init_analysis("INFO")
    all_data = get_all_sessions_metadata(excl_paradigms=[100,300,600,700])
    rewrite_metadata(all_data)
    draw_all_sessions(all_data)
    