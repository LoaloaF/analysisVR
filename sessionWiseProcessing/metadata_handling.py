import pandas as pd
import json 

def _parse_metadata(metadata):
    paradigm_name = metadata.get("paradigm_name")
    if paradigm_name is not None and paradigm_name.shape[0]:
        paradigm_name = paradigm_name.item()
        paradigm_id = int(paradigm_name[1:5])
    
    animal = metadata.get("animal_name")
    if animal is not None and animal.shape[0]: 
        animal = animal.item()
    
    animal_weight = metadata.get("animal_weight")
    if animal_weight is not None and animal_weight.shape[0]: 
        animal_weight = animal_weight.item()
    
    start_time = metadata.get("start_time")
    if start_time is not None and start_time.shape[0]: 
        start_time = start_time.item()
    
    stop_time = metadata.get("stop_time")
    if stop_time is not None and stop_time.shape[0]: 
        stop_time = stop_time.item()
    
    duration = metadata.get("duration")
    if duration is not None and duration.shape[0]: 
        duration = duration.item()
    
    notes = metadata.get("notes")
    if notes is not None and notes.shape[0]: 
        notes = "".join(notes.item())
    
    metadata = json.loads(metadata.metadata.item())
    
    metadata_parsed = {
        "paradigm_name": paradigm_name,
        "paradigm_id": paradigm_id,
        "animal": animal,
        "animal_weight": animal_weight,
        "start_time": start_time,
        "stop_time": stop_time,
        "duration": duration,
        "notes": notes,
        "pillars": metadata.get("pillars"),
        "pillar_details": metadata.get("pillar_details"),
        "envX_size": metadata.get("envX_size"),
        "envY_size": metadata.get("envY_size"),
        "base_length": metadata.get("base_length"),
        "wallzone_size": metadata.get("wallzone_size"),
        "wallzone_collider_size": metadata.get("wallzone_collider_size"),
        "paradigms_states": metadata.get("paradigms_states"),
        "paradigms_transitions": metadata.get("paradigms_transitions"),
        "paradigms_decisions": metadata.get("paradigms_decisions"),
        "paradigms_actions": metadata.get("paradigms_actions"),
    }
    return metadata_parsed

session_fullfname = "/mnt/SpatialSequenceLearning/RUN_rYL001/rYL001_P0800/2024-08-28_16-50_rYL001_P0800_LinearTrack.xlsx_17min/2024-08-28_16-50_rYL001_P0800_LinearTrack.xlsx_17min.hdf5"
metadata = pd.read_hdf(session_fullfname, key="metadata")

metadata = _parse_metadata(metadata)

if metadata["paradigm_id"] == 800:
    pillars_posY = {}
    
    if len(metadata["pillar_details"]) == 17:
        # the most up-to-date version of the paradigm
        for i in range(1, 17):
            pillar_pos = [value["y"] for key, value in metadata["pillars"].items() if value['id'] == i][0]
            # To transfer excel coordinates to unity coordinates
            pillars_posY[f"pillar{i}"] = metadata["envX_size"]/2 - pillar_pos
        
        regions_pos = {
            'start_zone': (-169,pillars_posY["pillar5"]),
            'cue1_visible': (pillars_posY["pillar5"],pillars_posY["pillar6"]),
            'cue1': (pillars_posY["pillar6"],pillars_posY["pillar7"]),
            'cue1_passed': (pillars_posY["pillar7"],pillars_posY["pillar8"]),
            'between_cues': (pillars_posY["pillar8"],pillars_posY["pillar9"]),
            'cue2_visible': (pillars_posY["pillar9"],pillars_posY["pillar10"]),
            'cue2': (pillars_posY["pillar10"],pillars_posY["pillar11"]),
            'cue2_passed': (pillars_posY["pillar11"],pillars_posY["pillar12"]),
            'before_reward1': (pillars_posY["pillar12"],pillars_posY["pillar13"]),
            'reward1': (pillars_posY["pillar13"],pillars_posY["pillar14"]),
            'before_reward2': (pillars_posY["pillar14"],pillars_posY["pillar15"]),
            'reward2': (pillars_posY["pillar15"],pillars_posY["pillar16"]),
            'post_reward': (pillars_posY["pillar16"],metadata["envX_size"]/2),
        }
        
    elif len(metadata["pillar_details"]) == 11:
        for i in range(1, 11):
            pillar_pos = [value["y"] for key, value in metadata["pillars"].items() if value['id'] == i][0]
            # To transfer excel coordinates to unity coordinates
            pillars_posY[f"pillar{i}"] = metadata["envX_size"]/2 - pillar_pos
        
        if metadata["envX_size"] == 440:
            # early case when we start at 0 and cue at first position
            start_pos = -metadata["envX_size"]/2
        elif metadata["envX_size"] == 540:
            # later case when we start at -169 and cue at second position
            start_pos = -169
        
        regions_pos = {
            'start_zone': (start_pos,pillars_posY["pillar5"]),
            'cue1_visible': (pillars_posY["pillar5"],pillars_posY["pillar5"]+40),
            'cue1': (pillars_posY["pillar5"]+40,pillars_posY["pillar5"]+80),
            'cue1_passed': (pillars_posY["pillar5"]+80,pillars_posY["pillar6"]),
            'between_cues': (pillars_posY["pillar6"],pillars_posY["pillar7"]),
            'cue2_visible': (pillars_posY["pillar7"],pillars_posY["pillar7"]+40),
            'cue2': (pillars_posY["pillar7"]+40,pillars_posY["pillar7"]+80),
            'cue2_passed': (pillars_posY["pillar7"]+80,pillars_posY["pillar8"]),
            'before_reward1': (pillars_posY["pillar8"],pillars_posY["pillar8"]+40),
            'reward1': (pillars_posY["pillar8"]+40,pillars_posY["pillar9"]),
            'before_reward2': (pillars_posY["pillar9"],pillars_posY["pillar9"]+40),
            'reward2': (pillars_posY["pillar9"]+40,pillars_posY["pillar10"]),
            'post_reward': (pillars_posY["pillar10"],metadata["envX_size"]/2),
        }
    else:
        print("Unknown condition, not supported")
        pass
    
    region = {}
    regions_name = list(regions_pos.keys())
    
    special_regions_pillar_idx = {
        # these are the regions that have pillar details
        "cue1": '1',
        "cue2": '2',
        "reward1": '3',
        "reward2": '4'
    }
    
    for each_region in regions_name:
        region[each_region] = {
            "start_pos": regions_pos[each_region][0],
            "end_pos": regions_pos[each_region][1]
        }
        if each_region in special_regions_pillar_idx:
            # these are the information I put in the metadata, maybe redundant
            region[each_region]["radius"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarRadius"]
            region[each_region]["height"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarHeight"]
            region[each_region]["z_pos"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarZposition"]
            region[each_region]["texture"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarTexture"]
            region[each_region]["transparency"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarTransparency"]
            region[each_region]["reward_radius"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarRewardRadius"]
            region[each_region]["show_ground"] = metadata["pillar_details"][special_regions_pillar_idx[each_region]]["pillarShowGround"]
        # below I am not sure if I should put None or leave it empty
        # else:
        #     region[each_region]["radius"] = None
        #     region[each_region]["height"] = None
        #     region[each_region]["z_pos"] = None
        #     region[each_region]["texture"] = None
        #     region[each_region]["transparency"] = None
        #     region[each_region]["reward_radius"] = None
        #     region[each_region]["show_ground"] = None
            
    
    metadata["regions"] = region
