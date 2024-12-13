import json
import os
import ast

from CustomLogger import CustomLogger as Logger

from analysis_core import PROJECT_DIR
# from analysis_utils import str2list

def extract_metadata(metadata, session_name):
    L = Logger()
    metadata_parsed = {"trial_id": 1, "session_name": session_name}
    
    # =================================================
    # useful metadata fields, unpacked as single values
    # =================================================
    
    paradigm_name = metadata.get("paradigm_name")
    if paradigm_name is not None and paradigm_name.shape[0]:
        metadata_parsed["paradigm_name"] = paradigm_name.item().replace(".xlsx", "")
        metadata_parsed["paradigm_id"] = int(paradigm_name.item()[1:5])

    animal = metadata.get("animal_name")
    if animal is not None and animal.shape[0]: 
        metadata_parsed["animal_name"] = animal.item()
        metadata_parsed["animal_id"] = int(animal.item()[-3:])

    animal_weight = metadata.get("animal_weight")
    if animal_weight is not None and animal_weight.shape[0]: 
        metadata_parsed["animal_weight"] = animal_weight.item()
        if metadata_parsed["animal_weight"] == "None":
            metadata_parsed["animal_weight"] = None
        else:
            metadata_parsed["animal_weight"] = int(metadata_parsed["animal_weight"])
            if metadata_parsed["animal_weight"] < 240: # accidental wrong input
                metadata_parsed["animal_weight"] = None

    start_time = metadata.get("start_time")
    if start_time is not None and start_time.shape[0]: 
        metadata_parsed["start_time"] = start_time.item()

    stop_time = metadata.get("stop_time")
    if stop_time is not None and stop_time.shape[0]: 
        metadata_parsed["stop_time"] = stop_time.item()

    duration = metadata.get("duration")
    if duration is not None and duration.shape[0]: 
        metadata_parsed["duration_minutes"] = duration.item()[:-3]
        if metadata_parsed["duration_minutes"] == "":
            metadata_parsed["duration_minutes"] = None
        else:
            metadata_parsed["duration_minutes"] = int(metadata_parsed["duration_minutes"])

    notes = metadata.get("notes")
    if notes is not None and notes.shape[0]: 
        metadata_parsed["notes"] = notes.item()
        if metadata_parsed["notes"] == "Free notes here":
            metadata_parsed["notes"] = None
        if metadata_parsed["notes"] is not None:
            metadata_parsed["notes"] = metadata_parsed["notes"].replace('\'', ', ').replace('[', '').replace(']', '')
    
    rewardPostSoundDelay = metadata.get("rewardPostSoundDelay")
    if rewardPostSoundDelay is not None and rewardPostSoundDelay.shape[0]:
        metadata_parsed["rewardPostSoundDelay"] = rewardPostSoundDelay.item()
        
    rewardAmount = metadata.get("rewardAmount")
    if rewardAmount is not None and rewardAmount.shape[0]:
        metadata_parsed["rewardAmount"] = rewardAmount.item()
    
    successSequenceLength = metadata.get("successSequenceLength")
    if successSequenceLength is not None and successSequenceLength.shape[0]:
        metadata_parsed["successSequenceLength"] = successSequenceLength.item()
    
    # =============================================
    # list-like  metadata fields, packed as strings
    # =============================================
    
    trialPackageVariables = metadata.get("trialPackageVariables")
    if trialPackageVariables is not None and trialPackageVariables.shape[0]:
        metadata_parsed["trialPackageVariables"] = str2list(trialPackageVariables.item())
        
    trialPackageVariablesDefault = metadata.get("trialPackageVariablesDefault")
    if trialPackageVariablesDefault is not None and trialPackageVariablesDefault.shape[0]:
        metadata_parsed["trialPackageVariablesDefault"] = str2list(trialPackageVariablesDefault.item())
        
    trialPackageVariablesFullNames = metadata.get("trialPackageVariablesFullNames")
    if trialPackageVariablesFullNames is not None and trialPackageVariablesFullNames.shape[0]:
        metadata_parsed["trialPackageVariablesFullNames"] = str2list(trialPackageVariablesFullNames.item())
        
    # =================================
    # JSON based nested metadata fields
    # =================================
    
    configuration = metadata.get("configuration")
    if configuration is not None and configuration.shape[0]:
        try:
            metadata_parsed['configuration'] = json.loads(configuration.item())
        except json.JSONDecodeError:
            L.logger.error(f"Failed to parse configuration: {configuration.item()}")
            metadata_parsed['configuration'] = None
    
    env_metadata = metadata.get("env_metadata")
    if env_metadata is not None and env_metadata.shape[0]:
        try:
            metadata_parsed['env_metadata'] = json.loads(env_metadata.item())
        except json.JSONDecodeError:
            L.logger.error(f"Failed to parse env_metadata: {env_metadata.item()}")
            metadata_parsed['env_metadata'] = None
            
            # env_metadata is important, patch for 1-3 P0800 sessions with default
            if metadata_parsed['paradigm_id'] == 800:
                L.logger.warning("P0800 metadata is faulty, patching with default values")
                metadata_parsed['env_metadata'] = _get_default_P0800_env_metadata()

    fsm_metadata = metadata.get("fsm_metadata")
    if fsm_metadata is not None and fsm_metadata.shape[0]:
        try:
            metadata_parsed['fsm_metadata'] = json.loads(fsm_metadata.item())
        except json.JSONDecodeError:
            L.logger.error(f"Failed to parse fsm_metadata: {fsm_metadata.item()}")
            metadata_parsed['fsm_metadata'] = None
    
    log_file_content = metadata.get("log_file_content")
    if log_file_content is not None and log_file_content.shape[0]:
        try:
            metadata_parsed['log_file_content'] = json.loads(log_file_content.item())
        except json.JSONDecodeError:
            L.logger.error(f"Failed to parse log_file_content: {log_file_content.item()}")
            metadata_parsed['log_file_content'] = None
        
    # ===============================================================
    # old metadata format, get nested fields from nested metadata key
    # ===============================================================
    
    if (nested_metadata := metadata.get("metadata")) is not None: 
        L.logger.debug("Parsing old-format metadata")
        nested_metadata = json.loads(nested_metadata.item())
        metadata_parsed.update({'env_metadata': {
            "pillars": nested_metadata.get("pillars"),
            "pillar_details": nested_metadata.get("pillar_details"),
            "envX_size": nested_metadata.get("envX_size"),
            "envY_size": nested_metadata.get("envY_size"),
            "base_length": nested_metadata.get("base_length"),
            "wallzone_size": nested_metadata.get("wallzone_size"),
            "wallzone_collider_size": nested_metadata.get("wallzone_collider_size"),}})
            
        metadata_parsed.update({'fsm_metadata': {
            "paradigms_states": nested_metadata.get("paradigms_states"),
            "paradigms_transitions": nested_metadata.get("paradigms_transitions"),
            "paradigms_decisions": nested_metadata.get("paradigms_decisions"),
            "paradigms_actions": nested_metadata.get("paradigms_actions"),
        }})
        
        metadata_parsed.update({'log_file_content': {
            "log_file_content": nested_metadata.get("log_files"),
        }})
    
    if metadata_parsed['paradigm_id'] in (800, 1100): 
        track_details = env_metadata2track_details(metadata_parsed['env_metadata'])
        metadata_parsed['track_details'] = json.dumps(track_details, indent='  ')
        
    # add keys that are still in metadata but not in metadata_parsed, to see if something interesting is missing
    metadata_parsed['GAP'] = None
    for key in metadata.keys():
        if key not in metadata_parsed:
            metadata_parsed[key] = metadata[key].item()

    return metadata_parsed

def minimal_metad_from_session_name(session_name):
    infos = session_name.split("_")
    data = {
        "trial_id": 0,
        "session_name": session_name,
        "start_time": infos[0] + "_" + infos[1],
        "animal_name": infos[2],
        "animal_id": int(infos[2][-3:]),
        "paradigm_name": infos[3] + "_" + infos[4],
        "paradigm_id": int(infos[3][1:]),
        "duration_minutes": int(infos[5][:-3]) if infos[5]!="min" else None, # number can be missing
    }
    return data

def env_metadata2track_details(env_metadata):
    pillar_ids = list(set([val["id"] for val in env_metadata["pillars"].values()]))
    n_pillar_types = len(pillar_ids)
    pillars_posY = {}
    for i in pillar_ids:
        pillar_pos = [val["y"] for val in env_metadata["pillars"].values() if val['id'] == i][0]
        # transform excel coordinates to unity coordinates
        pillars_posY[f"pillar{i}"] = env_metadata["envX_size"]/2 - pillar_pos
    
    # the most up-to-date version of the paradigm 800 has 17 pillar types
    # P1100 has 19 pillar types, two more for cue textures in reward zone
    if n_pillar_types == 17 or n_pillar_types == 19:
        # gget the start and stop unity cooredinates for each region
        zone_positions = {
            # 'start_zone': (-169, pillars_posY["pillar5"]),
            'start_zone': (pillars_posY["pillar5"]-40, pillars_posY["pillar5"]),
            'cue1_visible': (pillars_posY["pillar5"], pillars_posY["pillar6"]),
            'cue1': (pillars_posY["pillar6"], pillars_posY["pillar7"]),
            'cue1_passed': (pillars_posY["pillar7"], pillars_posY["pillar8"]),
            'between_cues': (pillars_posY["pillar8"], pillars_posY["pillar9"]),
            'cue2_visible': (pillars_posY["pillar9"], pillars_posY["pillar10"]),
            'cue2': (pillars_posY["pillar10"], pillars_posY["pillar11"]),
            'cue2_passed': (pillars_posY["pillar11"], pillars_posY["pillar12"]),
            'before_reward1': (pillars_posY["pillar12"], pillars_posY["pillar13"]),
            'reward1': (pillars_posY["pillar13"], pillars_posY["pillar14"]),
            'before_reward2': (pillars_posY["pillar14"], pillars_posY["pillar15"]),
            'reward2': (pillars_posY["pillar15"], pillars_posY["pillar16"]),
            'post_reward': (pillars_posY["pillar16"], env_metadata["envX_size"]/2),
        }
        
    # P800 very old, used to have 11 pillar types
    # initial version of paradigm
    elif n_pillar_types == 11:
        if env_metadata["envX_size"] == 480:
            # print("offset:", -env_metadata["envX_size"]/2)
            # early case when we start at 0 and cue at first position
            start_pos = -env_metadata["envX_size"]/2
        elif env_metadata["envX_size"] == 540:
            # print("offset:", -169)
            # # later case when we start at -169 and cue at second position
            # start_pos = -169
            start_pos = -env_metadata["envX_size"]/2
        zone_positions = {
            'start_zone': (start_pos, pillars_posY["pillar5"]),
            'cue1_visible': (pillars_posY["pillar5"], pillars_posY["pillar5"]+40),
            'cue1': (pillars_posY["pillar5"]+40, pillars_posY["pillar5"]+80),
            'cue1_passed': (pillars_posY["pillar5"]+80, pillars_posY["pillar6"]),
            'between_cues': (pillars_posY["pillar6"], pillars_posY["pillar7"]),
            'cue2_visible': (pillars_posY["pillar7"], pillars_posY["pillar7"]+40),
            'cue2': (pillars_posY["pillar7"]+40, pillars_posY["pillar7"]+80),
            'cue2_passed': (pillars_posY["pillar7"]+80, pillars_posY["pillar8"]),
            'before_reward1': (pillars_posY["pillar8"], pillars_posY["pillar8"]+40),
            'reward1': (pillars_posY["pillar8"]+40, pillars_posY["pillar9"]),
            'before_reward2': (pillars_posY["pillar9"], pillars_posY["pillar9"]+40),
            'reward2': (pillars_posY["pillar9"]+40, pillars_posY["pillar10"]),
            'post_reward': (pillars_posY["pillar10"], env_metadata["envX_size"]/2),
        }
    else:
        raise ValueError(f"Unknown number of pillar types {n_pillar_types}, not 11 or 17")
    
    # these are the regions that have relevant pillar details (actual pillars)
    special_zones_pillar_indices = {"cue1": '1', "cue2": '2', "reward1": '3', 
                                    "reward2": '4'}
    
    track_details = {}
    for zone in zone_positions.keys():
        track_zone_details = {"start_pos": zone_positions[zone][0], 
                              "end_pos": zone_positions[zone][1]}
        if zone in special_zones_pillar_indices:
            pillar_idx = special_zones_pillar_indices[zone]
            track_zone_details["radius"] = env_metadata["pillar_details"][pillar_idx]["pillarRadius"]
            track_zone_details["height"] = env_metadata["pillar_details"][pillar_idx]["pillarHeight"]
            track_zone_details["z_pos"] = env_metadata["pillar_details"][pillar_idx]["pillarZposition"]
            track_zone_details["texture"] = env_metadata["pillar_details"][pillar_idx]["pillarTexture"]
            track_zone_details["transparency"] = env_metadata["pillar_details"][pillar_idx]["pillarTransparency"]
            track_zone_details["reward_radius"] = env_metadata["pillar_details"][pillar_idx]["pillarRewardRadius"]
            track_zone_details["show_ground"] = env_metadata["pillar_details"][pillar_idx]["pillarShowGround"]
        track_details[zone] = track_zone_details
    return track_details


def _get_default_P0800_env_metadata():
    path = os.path.join(PROJECT_DIR, "analysisVR", "data_loading")
    with open(os.path.join(path,"/P0800_env_metadata.json"), 'r') as f:
        metadata = json.load(f)
    return metadata

def str2list(string):
    """
    Convert a string representation of a list to an actual list.

    Parameters:
    - string (str): The string representation of the list.

    Returns:
    - list: The actual list.
    """
    try:
        # Use ast.literal_eval to safely evaluate the string
        result = ast.literal_eval(string)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("The provided string does not represent a list.")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid string representation of a list: {string}") from e