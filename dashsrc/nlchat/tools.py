"""NLChat tool layer: plot-generating functions + their registry.

Pure functions the LLM calls. Each reuses the existing analysisVR pipeline
(get_analytics -> group_filter_data -> plot_X.render_plot) and returns
(plotly_fig, text_summary). TOOL_SCHEMAS is the registry the agent advertises to
the model; the same structure a future MCP server would expose.

Every tool accepts a `global_data` kwarg: if the GUI already loaded the analytic
into memory it's used directly; otherwise the tool loads it from the NAS via
get_analytics (so it works headless too).
"""

import numpy as np
import pandas as pd

from analytics_processing import analytics
from ..plot_components.plot_wrappers.data_selection import group_filter_data
from ..plot_components.plots import plot_EvolvingStayDecision


# --------------------------------------------------------------------------- #
# Registry: what the model is allowed to call (Anthropic-style input_schema;
# the agent converts it to whatever the backend needs). One dict per plot.
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {
        "name": "generate_evolving_stay_decision",
        "description": (
            "Plot how an animal's stop/skip choice strategy evolves across trials "
            "and sessions, split by cue. This is the plot behind the GUI's "
            "'EvolvingStayDecision' / 'EvolvingStayTime' buttons. Uses the "
            "BehaviorTrialwise analytic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "animal_id": {"type": "integer", "description": "Animal id, e.g. 6."},
                "paradigm_id": {"type": "integer",
                                "description": "Paradigm id, e.g. 1100 (1D track)."},
                "mode_span": {"type": "integer",
                              "description": "Rolling window (trials) for the smoothed "
                                             "strategy line. Default 8."},
                "outcome": {"type": "array",
                            "items": {"type": "string", "enum": ["1 R", "1+ R", "no R"]},
                            "description": "Outcome splits to include."},
                "cue": {"type": "array",
                        "items": {"type": "string", "enum": ["Cue1", "Cue2"]},
                        "description": "Cue splits to include."},
                "part_of_session": {"type": "array",
                                    "items": {"type": "string",
                                              "enum": ["1/3", "2/3", "3/3"]},
                                    "description": "Which thirds of each session."},
            },
            "required": ["animal_id", "paradigm_id"],
        },
    },
]

# analytics each tool needs (mirrors get_vis_name_req_data for the GUI)
TOOL_REQUIRED_ANALYTICS = {
    "generate_evolving_stay_decision": ("BehaviorTrialwise",),
}


# --------------------------------------------------------------------------- #
# Tool implementations: plain args -> (fig, text). No Dash, no LLM.
# --------------------------------------------------------------------------- #
def _load_behavior_trialwise(paradigm_id, animal_id, global_data=None):
    """Use already-loaded data if present (GUI path); else read from NAS."""
    if global_data is not None and global_data.get("BehaviorTrialwise") is not None:
        d = global_data["BehaviorTrialwise"]
        return d.loc[pd.IndexSlice[paradigm_id, animal_id, :, :]]
    return analytics.get_analytics(
        "BehaviorTrialwise", mode="set",
        paradigm_ids=[paradigm_id], animal_ids=[animal_id],
    )


def _stay_decision_stats(filtered):
    """Grounded numbers the LLM can cite, using the data's own correctness flag
    (choice_R{cue}_correct) on each cue's trials: fraction correct in the first
    third vs the last third of trials, so the summary describes learning with real
    percentages. Trials are already ordered by session then trial."""
    parts = []
    for cue in (1, 2):
        cd = filtered[filtered["cue"] == cue]
        vals = cd[f"choice_R{cue}_correct"].dropna().to_numpy()
        n = len(vals)
        if n == 0:
            continue
        third = max(1, n // 3)
        early = 100 * (vals[:third] == 1).mean()
        late = 100 * (vals[-third:] == 1).mean()
        overall = 100 * (vals == 1).mean()
        parts.append(
            f"Cue{cue} (n={n}): correct choice at R{cue} {early:.0f}% in first third "
            f"-> {late:.0f}% in last third (overall {overall:.0f}%)"
        )
    return "; ".join(parts) if parts else "no per-cue trials to summarize"


def generate_evolving_stay_decision(animal_id, paradigm_id, mode_span=8,
                                    outcome=None, cue=None, part_of_session=None,
                                    global_data=None):
    data = _load_behavior_trialwise(paradigm_id, animal_id, global_data)
    if data is None or data.empty:
        return None, (f"No BehaviorTrialwise rows for animal {animal_id}, "
                      f"paradigm {paradigm_id}.")

    # translate the LLM-facing cue vocabulary to group_filter_data's vocabulary
    cue_map = {"Cue1": "Cue1 trials", "Cue2": "Cue2 trials"}
    cue_filter = [cue_map[c] for c in cue] if cue else ["Cue1 trials", "Cue2 trials"]

    filtered, _ = group_filter_data(
        data,
        outcome_filter=list(outcome) if outcome else ["1 R", "1+ R", "no R"],
        cue_filter=cue_filter,
        trial_filter=list(part_of_session) if part_of_session else ["1/3", "2/3", "3/3"],
        r1_choice_filter=["stop", "skip"],
        r2_choice_filter=["stop", "skip"],
    )
    if filtered.empty:
        return None, "No trials remained after filtering."

    fig = plot_EvolvingStayDecision.render_plot(filtered, mode_span, 1000, 1100)
    n_sessions = filtered.index.get_level_values("session_id").nunique()
    stats = _stay_decision_stats(filtered)
    text = (f"Evolving stop/skip decision for animal {animal_id}, paradigm "
            f"{paradigm_id}: {len(filtered)} trials across {n_sessions} sessions. "
            f"Stats -- {stats}.")
    return fig, text


TOOL_IMPLS = {
    "generate_evolving_stay_decision": generate_evolving_stay_decision,
}
