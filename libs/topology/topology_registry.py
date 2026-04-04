"""Registry for known circuit topology templates."""

from __future__ import annotations


TOPOLOGY_REGISTRY = {
    "ota2": ["two_stage_ota", "folded_cascode"],
    "ldo": ["pmos_ldo", "nmos_ldo"],
    "bandgap": ["brokaw", "sub_bandgap"],
}
