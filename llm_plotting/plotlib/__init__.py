"""Shared plotting substrate for the analysisVR figure notebooks.

The layering this package sits at the bottom of (imports only ever point downward):

    plotlib/            bins and axes    - topic-agnostic, no domain semantics
    *_plotting_lib.py   fits and neurons - figure composers, one module per analysis line
    *_analysis.py       the NAS and this week's question - dev notebooks
    lm<N>_plots.py      the poster - calls only, no definitions

Nothing in here knows what an SVM fit, a CCGP direction or a cue is; that vocabulary
belongs one layer up. Nothing in here does I/O or has import-time side effects either,
with the single deliberate exception of `session_data`, whose whole job is loading.
"""
