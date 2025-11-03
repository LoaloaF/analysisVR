# !/usr/bin/env python3 
# executable - set up paths for import
import os
import sys
# to setup import paths add project root dir to sys.path (with baseVR dir in it)
sys.path.append(os.path.join(os.getcwd(), ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

from dashsrc.plot_components.plots import plot_unit_fr_stability
from dashsrc.plot_components.plots import plot_EnsambleEncondings
from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger
from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args

from dashsrc.plot_components.plots import plot_SessionWaveforms
from dashsrc.plot_components.plots import plot_EnsembleChoiceEncoding
from dashsrc.plot_components.plot_wrappers.data_selection import group_filter_data


output_dir = "./outputs/lm3"
data = {}
nas_dir = C.device_paths()[0]
Logger().init_logger(None, None, logging_level="INFO")

import plotly.io as pio
pio.renderers.default = "browser"


animal_ids = [6]
# animal_ids = [10]
session_ids = None
paradigm_ids = [1100]
excl_session_names = ['2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min', '2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min', '2024-12-11_17-42_rYL006_P1100_LinearTrackStop_30min']
cols = ["session_spike_count", "session_nsamples", "cluster_id", "unit_snr", "unit_Vpp"]
width = 700
height = 700
group_by = None

data['Spikes'] = analytics.get_analytics('Spikes', mode='set',
                                                      #  columns = ['amplitude_uV', 'cluster_id'],
                                                       paradigm_ids=paradigm_ids,
                                                       animal_ids=animal_ids,
                                                       excl_session_names=excl_session_names,
                                                       session_ids=session_ids)

data['SpikeClusterMetadata'] = analytics.get_analytics('SpikeClusterMetadata', mode='set',
                                                      #  columns = cols,
                                                       paradigm_ids=paradigm_ids,
                                                       animal_ids=animal_ids,
                                                       excl_session_names=excl_session_names,
                                                       session_ids=session_ids)

# HPC
data['SpikeClusterMetadata'] = data['SpikeClusterMetadata'][data['SpikeClusterMetadata'].cluster_id<=20]
# mPFC
data['SpikeClusterMetadata'] = data['SpikeClusterMetadata'][data['SpikeClusterMetadata'].cluster_id<20]
fig = plot_unit_fr_stability.render_plot_heatmap(data['SpikeClusterMetadata'])
# fullfname = f'{output_dir}/unit_fr_stability.svg'
fullfname = f'{output_dir}/unit_fr_stability_greater1Hz.svg'
# fig.write_image(fullfname, width=width, height=height, scale=1)
# execute = f"code {fullfname}"
# os.system(execute)
# fig.show()



# sess_Vpp = data['Spikes'].reset_index()[['session_id','cluster_id','amplitude_uV']].groupby(['session_id','cluster_id']).mean()

# highlight_cluster = 46
# fig = plot_unit_fr_stability.render_plot_amplitude(data['SpikeClusterMetadata'], sess_Vpp, highlight_cluster)
# fullfname = f'{output_dir}/unit_fr_stability_ampl46.svg'
# fig.write_image(fullfname, width=800, height=400, scale=1)
# execute = f"code {fullfname}"
# os.system(execute)
# fig.show()

# highlight_cluster = 15
# fig = plot_unit_fr_stability.render_plot_amplitude(data['SpikeClusterMetadata'], sess_Vpp, highlight_cluster)
# fullfname = f'{output_dir}/unit_fr_stability_ampl15.svg'
# fig.write_image(fullfname, width=800, height=400, scale=1)
# execute = f"code {fullfname}"
# os.system(execute)
# fig.show()



# data["SessionMetadata"] = analytics.get_analytics('SessionMetadata', mode='set',
#                                          paradigm_ids=paradigm_ids,
#                                          animal_ids=animal_ids,
#                                          excl_session_names=excl_session_names,
#                                          session_ids=session_ids)
# fig = plot_SessionWaveforms.render_plot(data['SpikeClusterMetadata'], data['SessionMetadata'],
#                                         width, height,)
# fig.show()

















animal_ids = [6]
session_ids = None
# paradigm_ids = [1100, 0]
# excl_session_names = ['2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min', 
#                       '2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min',
#                       '2024-12-02_16-09_rYL006_P1100_LinearTrackStop_28min']
# cols = ["session_spike_count", "session_nsamples", "cluster_id", "unit_snr", "unit_Vpp"]
width = 700
height = 700
analytic = 'Ensemble40msProjEventAligned'
data[analytic] = analytics.get_analytics(analytic, mode='set',
                                         paradigm_ids=None,
                                         animal_ids=animal_ids,
                                        #  excl_session_names=excl_session_names,
                                         session_ids=session_ids)
data[analytic].set_index(['session_id', 't0', 'interval_t'], inplace=True,)
invalid_session_ids = 10, 24, 25
data[analytic] = data[analytic][~data[analytic].index.get_level_values('session_id').isin(invalid_session_ids)]

ens_selection = ['Assembly012', 'Assembly013', 'Assembly014', 'Assembly015', 'Assembly016', 'Assembly017',
                 'Assembly018', 'Assembly019', 'Assembly020', 'Assembly021'] 
event_selection = ['enter_afterCueZone', 'enter_reward1Zone', 'enter_reward2Zone']
session_slider = [7, 23]
valid_sessions = [sid for sid in range(session_slider[0], session_slider[1] + 1) 
                  if sid in data[analytic].index.unique('session_id')]


data[analytic] = data[analytic].loc[valid_sessions,]
data[analytic] = data[analytic][data[analytic].t0_event_name.isin(event_selection)]
drp_ens_cols = [c for c in data[analytic].columns if c.startswith('Assembly') and c not in ens_selection]
data[analytic].drop(columns=drp_ens_cols, inplace=True)

# # simplification, 0 or one instead of 1, 1+ 0 rewards
data[analytic].loc[:, 'trial_outcome'] = data[analytic].loc[:, 'trial_outcome'].astype(bool).astype(int)
print(data[analytic].index.unique('session_id'))

outcome_filter = ['1 R', '1+ R', 'no R']
cue_filter = ['Cue1 trials', 'Cue2 trials']
part_of_session_filter = ['1/3', '2/3', '3/3']
r1_choice_filter = ['stop', 'skip']
r2_choice_filter = ['stop', 'skip']
group_by = 'Cue' #'R1 choice' #, 'R1 choice', 'R2 choice'
data[analytic], group_by_values = group_filter_data(data[analytic], outcome_filter=outcome_filter,
                                                    cue_filter=cue_filter,
                                                    trial_filter=part_of_session_filter,
                                                    r1_choice_filter=r1_choice_filter,
                                                    r2_choice_filter=r2_choice_filter,
                                                    group_by=group_by)
print(group_by_values)
# print(data[analytic])

# fig = plot_EnsambleEncondings.render_plot(data[analytic], group_by, group_by_values,
#                                           ens_selection=ens_selection,)
#                                     # width=width, height=height)

# #  takes super long to render
# fullfname = f'{output_dir}/cue1vs2_delayperiod_ens12.pdf'
# fig.write_image(fullfname, width=fig.layout.width, height=fig.layout.height, scale=1)
# execute = f"code {fullfname}"
# os.system(execute)
# fig.show()



# not used thresholds for drawing activation strengths for each session, and each time point
plot_set = {
            9:(1.2,1.4,1.4),
            12:(1.2,1.4,1.4),
            13:(1.5,1.5,1.8),
            14:(1.5,1.7,2),
            15:(1.45,1.8,1.55),
            16:(1.45,1.8,1.55),
            17:(1.45,1.8,1.55),
            18:(1.45,1.8,1.55),
            19:(1.45,1.8,1.55),
            20:(1.45,1.8,1.55),
            21:(1.45,1.8,1.55),
            }

for ens in ens_selection:
  for cue in ['Cue2', 'Cue1']:
      #   for s_id in data[analytic].index.unique('session_id'):
      for s_id, (post_cue_thr, bef_R1_thr, bef_R2_thr) in plot_set.items():
              drp_ens_cols = [c for c in data[analytic].columns if c.startswith('Assembly') and c != ens]
              dat = data[analytic].drop(columns=drp_ens_cols)
              dat = data[analytic].xs(s_id, level='session_id', drop_level=False)
              fig = plot_EnsembleChoiceEncoding.render_plot(dat, ens_selection[ens], which_cue=cue,
                                                  #   post_cue_thr=post_cue_thr, bef_R1_thr=bef_R1_thr, bef_R2_thr=bef_R2_thr,
                                                    window=(15,35))
              fullfname = f'{output_dir}/sankeys/curated_session_choices_s{s_id}_ens{ens}_{cue}.svg'
              fig.write_image(fullfname, width=fig.layout.width, height=fig.layout.height, scale=1)
              fig.write_image(fullfname.replace('svg', 'png'), width=fig.layout.width, height=fig.layout.height, scale=10)
              # os.system(f"code {fullfname.replace('svg', 'png')}")
              fig.show()
              print(f"Saved {fullfname}")
              # exit()


# below glues the plots above together in one panel

import os
import glob
from PIL import Image
import re

output_dir = "./outputs/lm3"
png_files = sorted(glob.glob(f'{output_dir}/sankeys/curated_session_choices_*.png'))

# Regex to extract session and cue from filename
pattern = re.compile(r'curated_session_choices_(\d+)_(Cue\d)\.png')

# Organize images by session
sessions = {}
for fname in png_files:
    match = pattern.search(os.path.basename(fname))
    if match:
        session_id = int(match.group(1))
        cue = match.group(2)
        sessions.setdefault(session_id, {})[cue] = fname

# Only keep sessions with both cues
sessions = {sid: cues for sid, cues in sessions.items() if 'Cue1' in cues and 'Cue2' in cues}

if not sessions:
    print("No complete session pairs found for concatenation.")
else:
    session_imgs = []
    for sid in sorted(sessions):
        imgs = [Image.open(sessions[sid]['Cue2']), Image.open(sessions[sid]['Cue1'])]  # top: Cue2, bottom: Cue1
        # Resize both images to the same width (min width)
        min_width = min(img.width for img in imgs)
        imgs_resized = [img.resize((min_width, img.height), Image.LANCZOS) for img in imgs]
        # Stack vertically
        total_height = sum(img.height for img in imgs_resized)
        stacked_img = Image.new('RGB', (min_width, total_height))
        y_offset = 0
        for img in imgs_resized:
            stacked_img.paste(img, (0, y_offset))
            y_offset += img.height
        session_imgs.append(stacked_img)

    # Resize all session columns to the same height (min height)
    min_height = min(img.height for img in session_imgs)
    session_imgs_resized = [img.resize((img.width, min_height), Image.LANCZOS) for img in session_imgs]
    # Concatenate horizontally
    total_width = sum(img.width for img in session_imgs_resized)
    final_img = Image.new('RGB', (total_width, min_height))
    x_offset = 0
    for img in session_imgs_resized:
        final_img.paste(img, (x_offset, 0))
        x_offset += img.width

    output_filename = os.path.join(output_dir, 'all_session_choices_stacked_curated.png')
    final_img.save(output_filename)
    print("Stacked PNG saved to:", output_filename)