import numpy as np

    
def group_filter_data(data, outcome_filter=['1 R', '1+ R', 'no R'], 
                      cue_filter=['Early R', 'Late R'], 
                      trial_filter=['1/3', '2/3', '3/3'], group_by="None"):
    group_by_values = None
    def group_filter_session(sess_d, outcome_filter, cue_filter, 
                             trial_filter, group_by="None"):
        # last session determines this var, should be the same across all sessions
        nonlocal group_by_values
        group_values = {}
        # outcome filtering
        one_r_outcomes = [1,11,21,31,41,51,10,20,30,40,50]
        if '1 R' in outcome_filter:
            # group_values['1 R'] = [1]
            group_values['1 R'] = one_r_outcomes
        if '1+ R' in outcome_filter:
            group_values['1+ R'] = [i for i in range(1,56) if i not in one_r_outcomes]
        if 'no R' in outcome_filter:
            group_values['no R'] = [0]
        sess_d = sess_d[sess_d['trial_outcome'].isin(np.concatenate(list(group_values.values())))]
        if group_by == 'Outcome':
            group_by_values = group_values
        
        # cue filtering
        group_values = {}
        if 'Cue1 trials' in cue_filter:
            group_values['Early R'] = [1]
        if 'Cue2 trials' in cue_filter:
            group_values['Late R'] = [2]
        sess_d = sess_d[sess_d['cue'].isin(np.concatenate(list(group_values.values())))]
        if group_by == 'Cue':
            group_by_values = group_values
            
        # trial filtering
        group_values = {}
        # get the 1st, 2nd, 3rd proportion of trials/ split in thirds
        trial_groups = np.array_split(sess_d['trial_id'].unique(), 3)
        if "1/3" in trial_filter:
            group_values["1/3"] = trial_groups[0]
        if "2/3" in trial_filter:
            group_values["2/3"] = trial_groups[1]
        if "3/3" in trial_filter:
            group_values["3/3"] = trial_groups[2]
        incl_trials = np.concatenate([tg for tg in group_values.values()])
        sess_d = sess_d[sess_d['trial_id'].isin(incl_trials)]
        if group_by == 'Part of session':
            group_by_values = group_values
            
        # # choice filtering
        # group_values = {}
        # if 'Cue1 trials' in cue_filter:
        #     group_values['Early R'] = [1]
        # if 'Cue2 trials' in cue_filter:
        #     group_values['Late R'] = [2]
        # sess_d = sess_d[sess_d['cue'].isin(np.concatenate(list(group_values.values())))]
        # if group_by == 'Cue':
        #     group_by_values = group_values
            
            
        return sess_d
    data = data.groupby(level='session_id').apply(group_filter_session, outcome_filter, 
                                                  cue_filter, trial_filter, 
                                                  group_by)
    # groupby prepends the groupby column to the index (duplicate session_id), remove it
    data = data.reset_index(level=0, drop=True) 
    return data, group_by_values