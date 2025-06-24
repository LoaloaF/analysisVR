import os
from time import sleep
from collections import OrderedDict

import pandas as pd
import numpy as np

from CustomLogger import CustomLogger as Logger

import analytics_processing.analytics_constants as C
import analytics_processing.agg_modalities2analytic as m2a
import analytics_processing.integr_analytics as integr_analytics
import analytics_processing.sessions_from_nas_parsing as sp

import ephys_preprocessing.postproc_mea1k_ephys as ephys

def _get_analytics_fname(session_dir, analytic):
    full_path = os.path.join(session_dir, "session_analytics")
    if not os.path.exists(full_path):
        print("Creating analytics directory for session ", os.path.basename(session_dir))
        os.makedirs(full_path)
    fullfname = os.path.join(full_path, analytic+".parquet")
    return fullfname

def _compute_analytic(analytic, session_fullfname, all_sess_ffnames):
    L = Logger()
    session_name = os.path.basename(session_fullfname)[:-5]
    session_names = [os.path.basename(s_ffname)[:-5] for s_ffname in all_sess_ffnames]
    L.logger.debug(f"Computing {analytic} for {session_name}:")
    
    # TODO still check this one, i still don't like sessionOverview handling
    if analytic == "SessionMetadata":
        data = m2a.get_SesssionMetadata(session_fullfname)
        data_table = C.SESSION_METADATA_TABLE
    
    elif analytic == "TrackKinematics":
        data = m2a.get_TrackKinematics(session_fullfname)
        schema = C.SCHEMA_TrackKinematics
    
    elif analytic == "BehaviorTrialwise":
        track_kinematics = get_analytics(analytic="TrackKinematics",
                                         columns=['trial_id', 'frame_pc_timestamp',
                                                  'frame_ephys_timestamp',
                                                  'track_zone',
                                                  'frame_position', 'frame_velocity',
                                                  'frame_acceleration', 
                                                  ],
                                         session_names=[session_name])
        if track_kinematics is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = m2a.get_BehaviorTrialwise(session_fullfname, track_kinematics)
        data_table = C.SCHEMA_BehaviorTrialwise
    
    elif analytic == "BehaviorEvents":
        cols = ['trial_id', 'trial_start_pc_timestamp', 'trial_end_pc_timestamp',
                'cue', 'trial_outcome', 'choice_R1', 'choice_R2']
        trialwise = get_analytics(analytic="BehaviorTrialwise", columns=cols,
                                  session_names=[session_name])
        if trialwise is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = m2a.get_BehaviorEvents(session_fullfname, trialwise)
        data_table = C.SCHEMA_BehaviorEvents
        
    # TODO impelemnt 
    elif analytic == "BehaviorPose":
        cols = ['trial_id', 'trial_start_pc_timestamp', 'trial_end_pc_timestamp',
                'cue', 'trial_outcome', 'choice_R1', 'choice_R2']
        trialwise = get_analytics(analytic="BehaviorTrialwise", columns=cols,
                                  session_names=[session_name])
        if trialwise is None:
            L.logger.warning("Missing lower level analytic")
            return None
        
        data = m2a.get_BehaviorPose(session_fullfname, trialwise)
        data_table = C.SCHEMA_BehaviorPose
    
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
        
        # TODO: poses
        # # 4. facecam poses
        # pose_data = get_analytics(analytic="BehaviorPoses",
        #                           session_names=[session_name],)
    
        data = integr_analytics.get_BehaviorFramewise(track_kinematics, trialwise, 
                                                      events, ) # pose_data)
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
    
    
    
    # elif analytic == "Unity":
    #     data = m2a.get_Unity(session_fullfname)
    #     data_table = C.UNITY_TABLE
    
    # elif analytic == "BehaviorEvents":
    #     data = m2a.get_BehaviorEvents(session_fullfname)
    #     data_table = C.BEHAVIOR_EVENT_TABLE
        
    # elif analytic == "FacecamPoses":
    #     data = m2a.get_FacecamPoses(session_fullfname)
    #     int_columns = ['facecam_image_pc_timestamp', 'facecam_image_ephys_timestamp']
    #     data_table = OrderedDict((col, pd.Int64Dtype() if col in int_columns 
    #                               else pd.Float32Dtype()) for col in data.columns)

    # elif analytic == "UnityFramewise":
    #     data = m2a.get_UnityFramewise(session_fullfname)
    #     portenta_data = get_analytics(analytic="Portenta",
    #                                   session_names=[session_name])
        
    #     # integrate the behavior events and facecam poses
    #     data = integr_analytics.merge_behavior_events_with_frames(data, portenta_data)
    #     # data = integr_analytics.merge_facecam_poses_with_frames(data, pose_data)

    #     data_table = C.UNITY_FAMEWISE_TABLE
    #     # update the data_table with the facecam pose columns
    #     for col in data.columns:
    #         if col.startswith("facecam_pose_"):
    #             data_table[col] = pd.Float32Dtype()

    # elif analytic == "UnityTrackwise":
    #     unity_framewise = get_analytics(analytic="UnityFramewise",
    #                                     session_names=[session_name])
 
    #     data = integr_analytics.get_UnityTrackwise(unity_framewise)
        
    #     data_table = C.UNITY_TRACKWISE_TABLE
    #     # update the data_table with the facecam pose columns
    #     for col in data.columns:
    #         if col.startswith("facecam_pose_"):
    #             data_table[col] = pd.Float32Dtype()
    
    # elif analytic == "UnityTrialwiseMetrics":
    #     data = m2a.get_UnityTrialwiseMetrics(session_fullfname)
    #     print(data['trial_outcome'])
    #     print()
    #     print()
    #     print()
    #     print()
    #     # exit()
    #     # unity_framewise = get_analytics(analytic="UnityFramewise", 
    #     #                                 sessionlist_fullfnames=[session_fullfname])
    #     # data = integr_analytics.get_UnityTrialwiseMetrics(unity_framewise)
    #     # data_table = C.UNITY_TRIALWISE_METRICS_TABLE
    
    # elif analytic == "MultiUnits":
    #     data = m2a.get_MultiUnits(session_fullfname)
    #     data_table = C.MULTI_UNITS_TABLE
        
    elif analytic == "SpikeClusterMetadata":
        data = ephys.get_SpikeClusterMetadata(session_fullfname)
        data_table = C.SPIKES_CLUSTER_METADATA_TABLE
    
    elif analytic == "Spikes":
        sp_clust_metadata = get_analytics('SpikeClusterMetadata',
                                          session_names=[session_name],
                                          columns=['cluster_id_ssbatch', 'cluster_id', 
                                                   'cluster_color', 'cluster_type',
                                                   'unit_count', 'ss_batch_id',
                                                   'session_nsamples']
                                          )
        if sp_clust_metadata is None:
            return None
        data = ephys.get_Spikes(session_fullfname, sp_clust_metadata)
        data_table = C.SPIKES_TABLE
        # print(get_analytics('Spikes', session_names=[session_name],
        #                        columns=['sample_id', 'cluster_id']))
    
    elif analytic == "FiringRate40msHz":
        spikes = get_analytics('Spikes', session_names=[session_name],
                               columns=['ephys_timestamp', 'cluster_id'])
        if spikes is None:
            return None
        data = ephys.get_FiringRate40msHz(spikes.reset_index(drop=True))
        data_table = dict.fromkeys(data.columns, C.FIRING_RATE_40MS_HZ_ONE_DTYPE)
    
    # elif analytic == "FiringRate40msZ":
    #     fr_hz = get_analytics('FiringRate40msHz', session_names=[session_name],)
                               
    #     if fr_hz is None:
    #         return None
    #     data = ephys.get_FiringRate40msZ(fr_hz)
    #     data_table = dict.fromkeys(data.columns, C.FIRING_RATE_40MS_Z_ONE_DTYPE)
    
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
    
    elif set(analytic.split("-")) == {"PCsZonewise", "PCsZoneEmbeddings"}:
        trackfr_data = get_analytics('FiringRateTrackwiseHz', session_names=[session_name])
        track_behavior_data = get_analytics('BehaviorTrackwise', session_names=[session_name],)
                                        # columns=['frame_z_position', 'frame_pc_timestamp'])
        if trackfr_data is None:
            return None
        # combo analytic TODO, do it properly, order is not guaranteed too, rename
        data = ephys.get_PCsZonewise(trackfr_data, track_behavior_data)
        # data1_table, data2_table = C.PCS_ZONEWISE_TABLE, None
    
    elif set(analytic.split("-")) == {"CCsZonewise", "CCsZonewiseAngles"}:
        #  TODO rename to PCsZonewise, not bases
        session_subspace_basis = get_analytics('PCsZonewise', session_names=[session_name]) 
        all_subspace_basis = get_analytics('PCsZonewise', session_names=session_names)
        if session_subspace_basis is None or all_subspace_basis is None:
            L.logger.warning("Missing lower level analytic")
            return None
        data = ephys.get_PCsSubspaceAngles(session_subspace_basis, all_subspace_basis)
    
    elif analytic == "SVMCueOutcomeChoicePred":
        PCsZoneEmbeddings = get_analytics('PCsZoneEmbeddings', session_names=[session_name])
        if PCsZoneEmbeddings is None:
            return None
        data = ephys.get_SVMCueOutcomeChoicePred(PCsZoneEmbeddings)
        data_table = C.SVM_CUE_OUTCOME_CHOICE_PRED_TABLE

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
    
    aggr = []
    for session_fullfname, identif in zip(sessionlist_fullfnames, ids):
        L.logger.debug(f"Processing {analytic}, {identif} {os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
                                               analytic=analytic)
        
        if mode.endswith('compute'):
            # here, the user can specify multiple analytics, set mode is limited to one analytic
            combo_analytic_fnames = []
            for analyt in analytic.split('-'):
                analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
                                                       analytic=analyt)
                if os.path.exists(analytics_fname) and mode != "recompute":
                    L.logger.info(f"Output exists, skipping.")
                    continue
                combo_analytic_fnames.append(analytics_fname)
            
            data = _compute_analytic(analytic, session_fullfname, 
                                     all_sess_ffnames=sessionlist_fullfnames)
            if data is not None and not isinstance(data, tuple):
                # default, single iteration of the lopp above, no combo analytics
                data.to_parquet(analytics_fname, index=False, engine='pyarrow')
            
            elif data is not None and isinstance(data, tuple):
                # combo analytics return a tuple of dataframes and save them separately
                for analytics_fname, dat in zip(combo_analytic_fnames, data):
                    dat.to_parquet(analytics_fname, index=False, engine='pyarrow')
            else:
                L.logger.warning(f"Failed to compute {analytic} for {identif}")
            L.spacer("debug")
            
        
        elif mode == "set":
            if not os.path.exists(analytics_fname):
                L.logger.info(f"Analytic `{analytic}` not does not exist for"
                              f" {identif}, compute first, or check for typo")
                continue
            data = pd.read_parquet(analytics_fname, columns=columns)
            midx = [(*identif, i) for i in range(data.shape[0])]
            names = ["paradigm_id", "animal_id", "session_id", "entry_id"]
            data.index = pd.MultiIndex.from_tuples(midx, names=names)
            aggr.append(data)
            
        elif mode == "available":
            if os.path.exists(analytics_fname):
                aggr.append(identif)
        
        elif mode == 'clear':
            if os.path.exists(analytics_fname):
                L.logger.warning(f"Permantly DELETING {analytics_fname} in 1s !")
                sleep(1)
                os.remove(analytics_fname)
            else:
                L.logger.warning(f"File {analytics_fname} does not exist, skipping.")
        
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