"""Shared plotting configuration for all pred_error / spacetime_viz notebooks."""

import seaborn as sns

_DEEP = sns.color_palette("deep")

# Fixed color per model key — consistent across all dataset notebooks.
# fmt: off
MODEL_COLORS = {
    "s_onsagernet":         _DEEP[3],  # red
    "res_onsagernet":       _DEEP[1],  # orange
    "res_onsagernet_2d":    _DEEP[1],  # orange
    "onsagernet1d_lowrank": _DEEP[5],  # brown
    "fno":                  _DEEP[2],  # green
    "kdv":                  _DEEP[0],  # blue
    "ac":                   _DEEP[0],  # blue
    "ac_2d":                _DEEP[0],  # blue
    "kdv_realV":            _DEEP[4],  # purple
    "ac_realV":             _DEEP[4],  # purple
    "ac_2d_realV":          _DEEP[4],  # purple
}
# fmt: on
