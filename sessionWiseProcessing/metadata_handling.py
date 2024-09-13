import pandas as pd

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

session_fullfname = "path/to/session.h5"
metadata = pd.read_hdf(session_fullfname, key="metadata")

metadata = _parse_metadata(metadata)
if metadata["paradigm_id"] == 800:
    if len(metadata["pillar_details"]) == 16:
        # 16 pillars version
        regions = {
            'start_zone': (160,240),
            'cue1_visible': (240,320),
            # ... and so on, but use metadata["pillar_details"] to get the exact values
        }
    elif len(metadata["pillar_details"]) == 8:
        pass
        # 8 pillars version
    else:
        # other casese?
        pass
    
    print(regions)
    metadata["regions"] = regions
