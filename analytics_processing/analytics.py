from datetime import datetime
import os
from time import sleep

import pandas as pd
import numpy as np

from CustomLogger import CustomLogger as Logger

import analytics_processing.sessions_from_nas_parsing as sp
import analytics_processing.analytics_constants as C

# analytics computations related, allowed to fail for extiernal imports that only read analytics
try:
    from pyarrow import parquet as pq
    import analytics_processing.agg_modalities2analytic as m2a
    import analytics_processing.integr_analytics as integr_analytics
    import ephys_preprocessing.postproc_mea1k_ephys as ephys
except ImportError as e:
    print("Warning: Failed to import packages required for analytics computation modules:", e)


def _get_sess_analytic_fname(session_dir, analytic):
    full_path = os.path.join(session_dir, "session_analytics")
    if not os.path.exists(full_path):
        print("Creating analytics directory for session ", os.path.basename(session_dir))
        os.makedirs(full_path)
    fullfname = os.path.join(full_path, analytic+".parquet")
    return fullfname

def _get_animal_analytic_fname(animal_dir, analytic):
    full_path = os.path.join(animal_dir, "animal_analytics")
    if not os.path.exists(full_path):
        print("Creating analytics directory.")
        os.makedirs(full_path)
    fullfname = os.path.join(full_path, analytic+".parquet")
    return fullfname

def _get_available_analytics(session_fullfname):
    analytics_dir = os.path.join(os.path.dirname(session_fullfname), "session_analytics")
    available_analytics = []
    
    if not os.path.exists(analytics_dir):
        return None

    for analytics_fname in os.listdir(analytics_dir):
        if not analytics_fname.endswith(".parquet") or analytics_fname.startswith(".") or analytics_fname.startswith("AnalyticsOverview"):
            continue
        
        fp = os.path.join(analytics_dir, analytics_fname)
        # file modification time
        mtime = os.path.getmtime(fp)
        # human readable ISO
        dt = datetime.fromtimestamp(mtime)
        mtime_iso = f"{dt.strftime('%b')} {dt.day}. {dt.year}, {dt.strftime('%H:%M')}"

        # read parquet file metadata
        schema = pq.read_schema(fp)
        col_names = sorted(schema.names)
        # get total rows from file metadata (fast)
        pf = pq.ParquetFile(fp)
        n_rows = pf.metadata.num_rows
        available_analytics.append({
                "analytic": analytics_fname[:-8],
                "nrows": int(n_rows),
                "ncols": len(col_names),
                "column_names": col_names if not len(col_names) > 200 else col_names[:100] + ['  ...  '] + col_names[-100:],
                "last_modified": mtime_iso,
            }
        )
    available_analytics = pd.DataFrame.from_records(available_analytics)
    return available_analytics

def _compute_animal_analytic(analytic, all_sessions_ffnames):
    L = Logger()
    
    if analytic == "ConcatenatedPCs40ms":
        all_fr_Hz = get_analytics(analytic="FiringRate40msHz",
                                 session_names=sp.fullfnames2snames(all_sessions_ffnames))
        if all_fr_Hz is None:
            L.logger.warning("Missing lower level analytic")
            return None
        data = ephys.get_ConcatenatedPCA40ms(all_fr_Hz)
        # TODO
        # schema = C.SCHEMA_ConcatenatedPCs40ms
        
    # elif analytic == "ConcatenatedEnsambles40ms":
    elif set(analytic.split("-")) == {"ConcatenatedEnsambles40ms", "ConcatenatedEnsambleProj40ms"}:
        
        all_fr_hz = get_analytics(analytic="FiringRate40msHz",
                                session_names=sp.fullfnames2snames(all_sessions_ffnames))
        if all_fr_hz is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        PCs = get_analytics(analytic="ConcatenatedPCs40ms",
                            session_names=[os.path.basename(s_ffname)[:-5]
                                           for s_ffname in all_sessions_ffnames],)
        if PCs is None:
            L.logger.warning("Missing lower level analytic `ConcatenatedPCs40ms`")
            return None
        
        # assembly_templates, assembly_activity
        data = ephys.get_ConcatenatedEnsambles40ms(PCs, all_fr_hz)
        
    elif analytic == "SessionPCs40msCAs":
        PCs = get_analytics(analytic='SessionPCs40ms',
                            session_names=sp.fullfnames2snames(all_sessions_ffnames))
        if PCs is None:
            L.logger.warning("Missing lower level analytic `SessionPCs40ms`")
        
        data = ephys.get_SessionPCs40msCAs(PCs)
        
    # TODO should be sesssion analytic...    
    elif analytic ==  "Ensemble40msProjEventAligned":
        ensemble_proj = get_analytics('ConcatenatedEnsambleProj40ms',
                                      session_names=sp.fullfnames2snames(all_sessions_ffnames))

        if ensemble_proj is None:
            L.logger.warning("Missing lower level analytic")
            return None

        # kinematics = get_analytics('TrackKinematics', 
        kinematics = get_analytics('BehaviorFramewise', 
                                   columns=['frame_ephys_timestamp', 'frame_raw', 'frame_yaw', 
                                            'frame_pitch', 'frame_acceleration', 'frame_velocity',
                                            'lick_count', 'frame_position', 'reward-sound_count',
                                            'reward-valve-open_count', 'cue', 'trial_id', 'trial_outcome',
                                            'choice_R1', 'choice_R2',
                                            # 'reward-removed_count',
                                              ],
                                   session_names=sp.fullfnames2snames(all_sessions_ffnames))
        if kinematics is None:
            L.logger.warning("Missing lower level analytic")
            return None

        data = ephys.get_Ensemble40msProjEventAligned(ensemble_proj, kinematics)
        # data_table = C.ENSAMBLE_40MS_PROJ_ENCODING_TABLE
        
        
        
        
    
        
    return data
    
    
    
    
    
    
    
    
    
    
def _compute_sess_analytic(analytic, session_fullfname):
    L = Logger()
    session_name = os.path.basename(session_fullfname)[:-5]
    # session_names = [os.path.basename(s_ffname)[:-5] for s_ffname in all_sess_ffnames]
    L.logger.debug(f"Computing {analytic} for {session_name}:")
    
    # TODO still check this one, i still don't like sessionOverview handling
    if analytic == "SessionMetadata":
        data = m2a.get_SesssionMetadata(session_fullfname)
        schema = C.SESSION_METADATA_TABLE
    
    elif analytic == "TrackKinematics":
        data = m2a.get_TrackKinematics(session_fullfname)
        schema = C.SCHEMA_TrackKinematics
    
    elif analytic == "BehaviorTrialwise":
        track_kinematics = get_analytics(analytic="TrackKinematics",
                                         columns=['trial_id', 'frame_pc_timestamp',
                                                  'frame_ephys_timestamp', 'fps',
                                                  'track_zone', 'frame_RawYawPitch_abs_vel_sum',
                                                  'frame_position', 
                                                  ],
                                         session_names=[session_name])
        if track_kinematics is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = m2a.get_BehaviorTrialwise(session_fullfname, track_kinematics)
        # schema = C.SCHEMA_BehaviorTrialwise
    
    elif analytic == "BehaviorEvents":
        cols = ['trial_id', 'trial_start_pc_timestamp', 'trial_end_pc_timestamp',
                'cue', 'trial_outcome', 'choice_R1', 'choice_R2']
        trialwise = get_analytics(analytic="BehaviorTrialwise", columns=cols,
                                  session_names=[session_name])
        if trialwise is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = m2a.get_BehaviorEvents(session_fullfname, trialwise)
        schema = C.SCHEMA_BehaviorEvents
        
    elif analytic == "BehaviorPose":
        data = m2a.get_BehaviorPose(session_fullfname)
    
    elif analytic == "BehaviorFramewise":
        # 1. track kinematics
        track_kinematics = get_analytics(analytic="TrackKinematics",
                                            session_names=[session_name],)
        if track_kinematics is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        # 2. trial wise data
        trialwise = get_analytics(analytic="BehaviorTrialwise", session_names=[session_name])
        if trialwise is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        # 3. behavior events
        events = get_analytics(analytic="BehaviorEvents",
                               session_names=[session_name],
                               columns=["event_pc_timestamp", "event_ephys_timestamp", 
                                        "event_value", "event_name_full",])
        if events is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        pose_data = get_analytics(analytic="BehaviorPose",
                                  columns=['image_ephys_timestamp', 'image_pc_timestamp', 
                                           'head_angle', 'head_angle_vel',
                                           'movement_energy_smooth5'],
                                  session_names=[session_name],)
        if pose_data is None:
            L.logger.warning("Missing lower level analytic, but continueing")
    
        data = integr_analytics.get_BehaviorFramewise(track_kinematics, trialwise, 
                                                      events, pose_data)
        schema = C.SCHEMA_BehaviorFramewise
    
    elif analytic == "BehaviorTrackwise":
        framewise = get_analytics(analytic="BehaviorFramewise",
                                  session_names=[session_name])
        if framewise is None:
            L.logger.warning("Missing lower level analytic")
            return None
        data = integr_analytics.get_BehaviorTrackwise(framewise)
        # schema = C.SCHEMA_BehaviorTrackwise
        schema = None
    
    
    elif analytic == "TrialWiseT0Events40ms":
        behavior = get_analytics(analytic="Behavior40msAligned",
                                 session_names=[session_name],)
        if behavior is None:
            L.logger.warning("Missing lower level analytic")
            return None
        data = integr_analytics.get_TrialWiseT0Events40ms(behavior)
        # schema = C.SCHEMA_T0EVENTS
    
    
    
    
    
    
    
    
    
    
    
    
    ### ========================= EPHYS  BASIC  ===========================
    
    elif set(analytic.split("-")) == {'SpikeClusterMetadata', 'Spikes'}:
        sp_clust_metadata, spikes = ephys.extract_jrc_spikes(session_fullfname)
        data = spikes, sp_clust_metadata
        if data[0] is None:
            return None
    elif analytic == "FiringRate40msHz":
        spikes = get_analytics('Spikes', session_names=[session_name],
                               columns=['ephys_timestamp', 'cluster_id_str'])
        if spikes is None:
            L.logger.warning("Missing lower level analytic")
            return None
        all_cluster_names = get_analytics('SpikeClusterMetadata', session_names=[session_name],
                                          columns=['cluster_id_str', ])
        if spikes is None:
            L.logger.warning("Missing lower level analytic")
            return None
        data = ephys.get_FiringRate40msHz(spikes.reset_index(drop=True),all_cluster_names)
        data_table = dict.fromkeys(data.columns, C.FIRING_RATE_40MS_HZ_ONE_DTYPE)
    
    elif analytic == "FiringRate40msZ":
        fr_hz = get_analytics('FiringRate40msHz', session_names=[session_name],)
                               
        if fr_hz is None:
            return None
        data = ephys.get_FiringRate40msZ(fr_hz)
        data_table = dict.fromkeys(data.columns, C.FIRING_RATE_40MS_Z_ONE_DTYPE)
    
    elif analytic == "FiringRateTrackwiseHz":
        fr_data = get_analytics('FiringRate40msHz', session_names=[session_name])
        if fr_data is None:
            L.logger.warning("Missing lower level analytic")
            return None
        track_behavior_data = get_analytics('BehaviorTrackwise', session_names=[session_name],)
                                        # columns=['frame_z_position', 'frame_pc_timestamp'])
        if track_behavior_data is None:
            L.logger.warning("Missing lower level analytic")
            return None
        data = ephys.get_FiringRateTrackwiseHz(fr_data, track_behavior_data)
        data_table = dict.fromkeys(data.columns, C.FIRING_RATE_TRACKWISE_HZ_ONE_DTYPE)
        
    
    
    
    
    
    
    ### ========================= EPHYS  MORE  ===========================
        
    elif analytic == "SessionPCs40ms-SessionPCsProj40ms":
        fr_z = get_analytics('FiringRate40msZ', session_names=[session_name])
        if fr_z is None:
            L.logger.warning("Missing lower level analytic")
            return None
        # return PCs, Z_proj
        data = ephys.get_sessionPCA(fr_z)    

    # elif set(analytic.split("-")) == {"PCsZonewise", "PCsZoneEmbeddings"}:
    #     trackfr_data = get_analytics('FiringRateTrackwiseHz', session_names=[session_name])
    #     track_behavior_data = get_analytics('BehaviorTrackwise', session_names=[session_name],)
    #                                     # columns=['frame_z_position', 'frame_pc_timestamp'])
    #     if trackfr_data is None:
    #         return None
    #     # combo analytic TODO, do it properly, order is not guaranteed too, rename
    #     data = ephys.get_PCsZonewise(trackfr_data, track_behavior_data)
    #     # data1_table, data2_table = C.PCS_ZONEWISE_TABLE, None
    
    # elif set(analytic.split("-")) == {"CCsZonewise", "CCsZonewiseAngles"}:
    #     #  TODO rename to PCsZonewise, not bases
    #     session_subspace_basis = get_analytics('PCsZonewise', session_names=[session_name]) 
    #     all_subspace_basis = get_analytics('PCsZonewise', session_names=session_names)
    #     if session_subspace_basis is None or all_subspace_basis is None:
    #         L.logger.warning("Missing lower level analytic")
    #         return None
    #     data = ephys.get_PCsSubspaceAngles(session_subspace_basis, all_subspace_basis)
    
    elif analytic == "SVMCueOutcomeChoicePred":
        fr_z = get_analytics('FiringRate40msZ', session_names=[session_name])
        if fr_z is None:
            return None
        cols = ['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
                 'to_ephys_timestamp', 'frame_position',]
        beh = get_analytics('Behavior40msAligned', session_names=[session_name],
                            columns=cols)
        if beh is None:
            return None
        
        t0_events = get_analytics('TrialWiseT0Events40ms', session_names=[session_name],)

        data = ephys.get_SVMCueOutcomeChoicePred(fr_z, beh, t0_events)
        # data_table = C.SVM_CUE_OUTCOME_CHOICE_PRED_TABLE

        if data is None:
            L.logger.warning("Failed to compute SVM Cue Outcome Choice Prediction")
            return None
        
    elif analytic == "PVCueCorr":
        trackfr_data = get_analytics('FiringRateTrackwiseHz', session_names=[session_name])
        track_behavior_data = get_analytics('BehaviorTrackwise', session_names=[session_name],)
        if track_behavior_data is None:
            L.logger.warning("Missing lower level analytic")
            return None
        if trackfr_data is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = ephys.get_PVCueCorr(trackfr_data, track_behavior_data)
    
    # elif analytic == "FiringRateTrackbinsZ":
    #     fr_data = get_analytics('FiringRate40msZ', session_names=[session_name])
    #     track_behavior_data = get_analytics('UnityTrackwise', session_names=[session_name],)
    #                                     # columns=['frame_z_position', 'frame_pc_timestamp'])
    #     if fr_data is None:
    #         return None
    #     data = ephys.get_FiringRateTrackbinsHz(fr_data, track_behavior_data)
    #     print(data)
    #     data_table = dict.fromkeys(data.columns, C.FIRING_RATE_TRACKBINS_Z_ONE_DTYPE)
    
    elif analytic == "Behavior40msAligned":
        fr_raw = get_analytics('FiringRate40msHz', session_names=[session_name])
        if fr_raw is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        cols = ["frame_ephys_timestamp", "frame_pc_timestamp",
                # generally useful
                "trial_id", "cue", "trial_outcome", "choice_R1", "choice_R2",
                # forward velocity and acc
               'frame_raw_500msMedian', 'frame_raw_abs_acc_500msMedian', 
               # off rotations velocity
               'frame_YawPitch_abs_vel_sum_500msMedian', 'frame_YawPitch_abs_acc_sum_500msMedian', 
               # sum for reward threshold and acc
               'frame_RawYawPitch_abs_vel_sum_500msMedian', 'frame_RawYawPitch_abs_acc_sum_500msMedian',
               # derived metrics
               'frame_forward_prop',
               'forward_vs_rotation_corr',
               # count based events
               'lick_detected', 'reward-sound_detected', 'reward-valve-open_detected', # 'reward-removed_detected', can be missing TODO, fix
                # position info
                "frame_position", "track_zone", "track_zone_int", 'cue_visible',
                # action from camera pose
                "head_angle",
                "head_angle_vel",
                "movement_energy_smooth5",
        ]
        behavior = get_analytics('BehaviorFramewise', session_names=[session_name], 
                                 columns=cols)
        if behavior is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = integr_analytics.get_Behavior40msAligned(fr_raw, behavior)
        # schema = C.SCHEMA_BEHAVIOR_FR_40MS
        
        
    elif analytic == "AnalyticsOverview":
        data = _get_available_analytics(session_fullfname)
        schema = C.SESSION_ANALYTICS_OVERVIEW_TABLE
    
    
    else:
        raise ValueError(f"Unknown analytic: {analytic}")

    #TODO fix later
    # if analytic != "UnityTrialwiseMetrics" and data is not None:
    #     data = data.reindex(columns=data_table.keys())
    #     data = data.astype(data_table)
    
 
    if not isinstance(data, tuple):   
    
        try:
            data = data.astype(schema)
        except Exception as e:
            pass
            # data = data.astype(data_table)
        
        if data is not None:
            L.logger.debug(f"Computed analytic {analytic}:\n{data}\n{data.dtypes}")

    return data

# def _extract_id_from_sessionname(session_name):
#     session_name_split = session_name.split("_")
#     anim_name, parad_name = session_name_split[2], session_name_split[3]
#     return int(anim_name[-3:]), int(parad_name[1:]), 0

def get_analytics(analytic, mode="set", paradigm_ids=None, animal_ids=None, 
                  session_ids=None, session_names=None, excl_session_names=None,
                  from_date=None, to_date=None, columns=None,):
    L = Logger()
    
    sessionlist_fullfnames, ids = sp.sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, 
                                                                      session_ids, session_names, 
                                                                      excl_session_names,
                                                                      from_date, to_date)
    
    ANIMAL_ANALYTICS = ('ConcatenatedEnsambles40ms-ConcatenatedEnsambleProj40ms', 
                        'ConcatenatedPCs40ms', 'ConcatenatedEnsambleProj40ms',
                        'ConcatenatedEnsambles40ms',
                        'SessionPCs40msCAs', 'Ensemble40msProjEventAligned',
                        'Ensamble40msProjEncodings')
    aggr = []
    
    if analytic in ANIMAL_ANALYTICS:
        L.logger.debug(f"Processing {analytic}, for n={len(sessionlist_fullfnames)} sessions")
        animal_dir = os.path.join(os.path.dirname(sessionlist_fullfnames[0]), "..", "..")
        
        # Handle combo analytics for animal-level analytics
        if "-" in analytic:
            combo_analytic_fnames = []
            for analyt in analytic.split('-'):
                analytic_fname = _get_animal_analytic_fname(animal_dir, analytic=analyt)
                combo_analytic_fnames.append(analytic_fname)
            main_analytic_fname = combo_analytic_fnames[0]  # Use first analytic for existence check
        else:
            analytic_fname = _get_animal_analytic_fname(animal_dir, analytic=analytic)
            main_analytic_fname = analytic_fname
        
        if mode.endswith('compute'):
            if os.path.exists(main_analytic_fname) and mode != "recompute":
                L.logger.info(f"Output exists, skipping.")
                return None
            
            data = _compute_animal_analytic(analytic, sessionlist_fullfnames)
            if data is not None:
                if "-" in analytic and isinstance(data, tuple):
                    # combo analytics return a tuple of dataframes and save them separately
                    for analytic_fname, dat in zip(combo_analytic_fnames, data):
                        dat.to_parquet(analytic_fname, index=False, engine='pyarrow')
                else:
                    # single analytic
                    data.to_parquet(main_analytic_fname, index=False, engine='pyarrow')
            else:
                L.logger.warning(f"Failed to compute {analytic} for animal")
            return data

        elif mode == "set":
            print(main_analytic_fname)
        if not os.path.exists(main_analytic_fname):
            L.logger.info(f"Analytic `{analytic}` not does not exist for")
        data = pd.read_parquet(main_analytic_fname, columns=columns)
        # check when file was last modified
        last_mod_time = datetime.fromtimestamp(os.path.getmtime(main_analytic_fname))
        L.logger.info(f"Loaded animal-level analytic `{analytic}` last modified on {last_mod_time.isoformat()}")
        return data
        
    
    
    for session_fullfname, identif in zip(sessionlist_fullfnames, ids):
        L.logger.debug(f"Processing {analytic}, {identif} {os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        analytic_fname = _get_sess_analytic_fname(os.path.dirname(session_fullfname),
                                               analytic=analytic)
        
        if mode.endswith('compute'):
            # here, the user can specify multiple analytics, set mode is limited to one analytic
            combo_analytic_fnames = []
            for analyt in analytic.split('-'):
                analytic_fname = _get_sess_analytic_fname(os.path.dirname(session_fullfname),
                                                       analytic=analyt)
                combo_analytic_fnames.append(analytic_fname)
            if len(combo_analytic_fnames) > 1:
                L.logger.debug(f"Combo analytic {analytic} with files: "
                               f"{[os.path.basename(fname) for fname in combo_analytic_fnames]}")
            
            # for double analyitcs doens't skip second analyitc if exists
            if os.path.exists(analytic_fname) and mode != "recompute":
                L.logger.info(f"Output exists, skipping.")
                continue
            
            data = _compute_sess_analytic(analytic, session_fullfname)
            if data is not None and not isinstance(data, tuple):
                # default, single iteration of the lopp above, no combo analytics
                data.to_parquet(analytic_fname, index=False, engine='pyarrow')
            
            elif data is not None and isinstance(data, tuple):
                # combo analytics return a tuple of dataframes and save them separately
                for analytic_fname, dat in zip(combo_analytic_fnames, data):
                    dat.to_parquet(analytic_fname, index=False, engine='pyarrow')
            else:
                L.logger.warning(f"Failed to compute {analytic} for {identif}")
            L.spacer("debug")
            
        
        elif mode == "set":
            if not os.path.exists(analytic_fname):
                L.logger.info(f"Analytic `{analytic}` not does not exist for"
                              f" {identif}, compute first, or check for typo")
                continue
            data = pd.read_parquet(analytic_fname, columns=columns)
            midx = [(*identif, i) for i in range(data.shape[0])]
            names = ["paradigm_id", "animal_id", "session_id", "entry_id"]
            data.index = pd.MultiIndex.from_tuples(midx, names=names)
            
            # recover datatypes not preserved by parquet, like interval
            # if analytic == 'TrialWiseT0Events40ms':
            #     print(data.loc[:, 'pre_cue_interval'])
            #     print(data.loc[:, 'nextto_cue_interval'])
            #     exit()
            # for col in data.select_dtypes(include=['interval']).columns:
            #     data[col] = data[col].apply(lambda x: pd.Interval(x.left, x.right, closed=x.closed))
            
            aggr.append(data)
            
        elif mode == "available":
            if os.path.exists(analytic_fname):
                last_mod_time = datetime.fromtimestamp(os.path.getmtime(analytic_fname))
                aggr.append(list(identif)+[last_mod_time.strftime('%Y-%m-%d %H:%M')])
        
        elif mode == 'clear':
            if os.path.exists(analytic_fname):
                L.logger.warning(f"Permantly DELETING {analytic_fname} in 1s !")
                sleep(1)
                os.remove(analytic_fname)
            else:
                L.logger.warning(f"File {analytic_fname} does not exist, skipping.")
        
        else:
            raise ValueError(f"Unknown mode {mode}")
    
    if mode == "set":
        if len(aggr) == 0:
            return None
        aggr = pd.concat(aggr)
        
        # session_ids = aggr.index.get_level_values("session_id").tolist()
        session_ids = aggr.index.unique("session_id").tolist()
        paradigm_ids = aggr.index.unique("paradigm_id").tolist()
        animal_ids = aggr.index.unique("animal_id").tolist()
        mid_iloc = aggr.shape[0] // 2
        L.spacer("debug")
        L.logger.info(f"Returning {analytic} for {len(session_ids)} sessions.")
        L.logger.debug(f"Paradigm_ids: {paradigm_ids}, Animal_ids: {animal_ids}"
                       f"\n{aggr}\n{aggr.iloc[mid_iloc:mid_iloc+1].T}")
        return aggr
    elif mode == "available":
        aggr = np.array(aggr)
        L.logger.debug(f"Returning {analytic}:\n{aggr}")
        return aggr
