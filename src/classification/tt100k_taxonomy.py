"""
Tsinghua-Tencent 100K (TT100K) Class Taxonomy & Cross-Domain Ontology.
Maintained by Member D (Integration & Pipeline Lead).

Provides:
1. 221-Class TT100K Ontology (Prohibitory, Warning, Indicatory, Mandatory).
2. Semantic descriptions for Chinese & International road sign standards.
3. Bidirectional Cross-Domain Mapping between TT100K and GTSRB benchmarks.
4. Multi-Domain Consensus verification logic.
"""

from typing import Dict, Optional, Tuple, List
from src.schema import SignCategory

# 221 TT100K Class Codes (Source: Official Tsinghua-Tencent CVPR 2016 dataset)
TT100K_CLASSES: List[str] = [
    "pl5", "pl10", "pl15", "pl20", "pl25", "pl30", "pl40", "pl50", "pl60", "pl70",
    "pl80", "pl90", "pl100", "pl110", "pl120", "pm5", "pm10", "pm13", "pm15", "pm20",
    "pm25", "pm30", "pm35", "pm40", "pm46", "pm50", "pm55", "pm8", "pn", "pne",
    "ph4", "ph4.5", "ph5", "ps", "pg", "ph1.5", "ph2", "ph2.1", "ph2.2", "ph2.4",
    "ph2.5", "ph2.8", "ph2.9", "ph3", "ph3.2", "ph3.5", "ph3.8", "ph4.2", "ph4.3", "ph4.8",
    "ph5.3", "ph5.5", "pb", "pr10", "pr100", "pr20", "pr30", "pr40", "pr45", "pr50",
    "pr60", "pr70", "pr80", "pr90", "p1", "p2", "p3", "p4", "p5", "p6",
    "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16",
    "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p25", "p26",
    "p27", "p28", "pa8", "pa10", "pa12", "pa13", "pa14", "pb5", "pc", "pg",
    "ph1", "ph1.3", "ph1.5", "ph2", "ph3", "ph4", "ph5", "pi", "pl0", "pl4",
    "pl5", "pl8", "pl10", "pl15", "pl20", "pl25", "pl30", "pl35", "pl40", "pl50",
    "pl60", "pl65", "pl70", "pl80", "pl90", "pl100", "pl110", "pl120", "pm2", "pm8",
    "pm10", "pm13", "pm15", "pm20", "pm25", "pm30", "pm35", "pm40", "pm46", "pm50",
    "pm55", "pn", "pne", "po", "pr10", "pr100", "pr20", "pr30", "pr40", "pr45",
    "pr50", "pr60", "pr70", "pr80", "ps", "w1", "w2", "w3", "w5", "w8",
    "w10", "w12", "w13", "w16", "w18", "w20", "w21", "w22", "w24", "w28",
    "w30", "w31", "w32", "w34", "w35", "w37", "w38", "w41", "w42", "w43",
    "w44", "w45", "w46", "w47", "w48", "w49", "w50", "w51", "w52", "w53",
    "w54", "w55", "w56", "w57", "w58", "w59", "w60", "w62", "w63", "w66",
    "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10",
    "i11", "i12", "i13", "i14", "i15", "il60", "il80", "il100", "il110", "io",
    "ip"
]

# Human-readable descriptive names for key TT100K sign codes
TT100K_DESCRIPTIONS: Dict[str, str] = {
    # Speed limits (pl)
    "pl5": "Speed limit (5km/h)", "pl10": "Speed limit (10km/h)", "pl15": "Speed limit (15km/h)",
    "pl20": "Speed limit (20km/h)", "pl25": "Speed limit (25km/h)", "pl30": "Speed limit (30km/h)",
    "pl40": "Speed limit (40km/h)", "pl50": "Speed limit (50km/h)", "pl60": "Speed limit (60km/h)",
    "pl70": "Speed limit (70km/h)", "pl80": "Speed limit (80km/h)", "pl90": "Speed limit (90km/h)",
    "pl100": "Speed limit (100km/h)", "pl110": "Speed limit (110km/h)", "pl120": "Speed limit (120km/h)",
    
    # Prohibitory (pn, pne, p, pa, pr)
    "pn": "No passing / Overtaking prohibited",
    "pne": "End of no passing",
    "p1": "No entry for motor vehicles",
    "p2": "No entry for trucks",
    "p3": "No entry for tractors",
    "p6": "No parking",
    "p10": "No left turn",
    "p11": "No right turn",
    "p12": "No U-turn",
    "p14": "No entry",
    "pa14": "No entry / Restricted zone",
    "p19": "Stop sign",
    "ps": "Stop and give way",
    "pg": "Yield / Give way",

    # Warning / Danger (w)
    "w12": "Dangerous curve to the left",
    "w13": "Dangerous curve to the right",
    "w16": "Double curve",
    "w20": "Road narrows",
    "w21": "Road narrows on left",
    "w22": "Road narrows on right",
    "w30": "Pedestrians crossing",
    "w32": "Children crossing",
    "w34": "Bicycles crossing",
    "w35": "Wild animals crossing",
    "w41": "Bumpy road / Uneven surface",
    "w45": "Roundabout ahead",
    "w46": "Traffic signals ahead",
    "w55": "Road work ahead",
    "w57": "Traffic lights ahead",
    "w58": "Slippery road / Skid risk",
    "w59": "Crossroad / Intersection ahead",
    "w62": "Caution / General warning",

    # Indicatory & Mandatory (i, il, m)
    "i1": "Straight ahead only",
    "i2": "Turn right ahead",
    "i4": "Turn left ahead",
    "i5": "Pass on right",
    "il60": "Advisory minimum speed (60km/h)",
    "il80": "Advisory minimum speed (80km/h)",
    "il100": "Advisory minimum speed (100km/h)",
    "ip": "Parking area",
}

# Cross-Domain Semantic Alignment: Maps TT100K code to equivalent GTSRB Class ID [0..42]
TT100K_TO_GTSRB_MAP: Dict[str, int] = {
    # Speed Limits
    "pl20": 0, "pl30": 1, "pl50": 2, "pl60": 3, "pl70": 4, "pl80": 5, "pl100": 7, "pl120": 8,
    
    # Prohibitory
    "pn": 9,      # No passing
    "p2": 10,     # No passing for trucks / heavy vehicles
    "w59": 11,    # Intersection / Right-of-way
    "pg": 13,     # Yield
    "p19": 14,    # Stop
    "ps": 14,     # Stop
    "p1": 15,     # No vehicles
    "p14": 17,    # No entry
    "pa14": 17,   # No entry
    "w62": 18,    # General caution
    
    # Danger / Curves / Hazards
    "w12": 19,    # Curve left
    "w13": 20,    # Curve right
    "w16": 21,    # Double curve
    "w41": 22,    # Bumpy road
    "w58": 23,    # Slippery road
    "w22": 24,    # Road narrows on right
    "w55": 25,    # Road work
    "w57": 26,    # Traffic signals
    "w46": 26,    # Traffic lights
    "w30": 27,    # Pedestrians
    "w32": 28,    # Children crossing
    "w34": 29,    # Bicycles crossing
    "w35": 31,    # Wild animals
    "pne": 41,    # End of no passing
    
    # Mandatory
    "i2": 33,     # Turn right ahead
    "i4": 34,     # Turn left ahead
    "i1": 35,     # Ahead only
    "i5": 38,     # Keep right
    "w45": 40,    # Roundabout
}

# Reverse Mapping: GTSRB Class ID to Primary TT100K Code
GTSRB_TO_TT100K_MAP: Dict[int, str] = {
    0: "pl20", 1: "pl30", 2: "pl50", 3: "pl60", 4: "pl70", 5: "pl80",
    7: "pl100", 8: "pl120", 9: "pn", 10: "p2", 11: "w59", 13: "pg",
    14: "ps", 15: "p1", 17: "pa14", 18: "w62", 19: "w12", 20: "w13",
    21: "w16", 22: "w41", 23: "w58", 24: "w22", 25: "w55", 26: "w57",
    27: "w30", 28: "w32", 29: "w34", 31: "w35", 33: "i2", 34: "i4",
    35: "i1", 38: "i5", 40: "w45", 41: "pne"
}


def get_tt100k_name(code_or_idx) -> str:
    """Returns human-readable name for a TT100K class code or index."""
    if isinstance(code_or_idx, int):
        if 0 <= code_or_idx < len(TT100K_CLASSES):
            code = TT100K_CLASSES[code_or_idx]
        else:
            return "Unknown TT100K Sign"
    else:
        code = str(code_or_idx).lower()
    return TT100K_DESCRIPTIONS.get(code, f"TT100K Sign [{code}]")


def map_tt100k_to_gtsrb(code: str) -> Optional[int]:
    """Maps a TT100K class code to equivalent GTSRB class ID."""
    return TT100K_TO_GTSRB_MAP.get(code.lower(), None)


def map_gtsrb_to_tt100k(gtsrb_id: int) -> Optional[str]:
    """Maps a GTSRB class ID to equivalent TT100K class code."""
    return GTSRB_TO_TT100K_MAP.get(gtsrb_id, None)


def evaluate_consensus(gtsrb_id: int, tt100k_code: str) -> Tuple[bool, str]:
    """
    Evaluates whether the primary GTSRB model and secondary TT100K model agree.
    Returns: (is_consensus, consensus_summary)
    """
    mapped_gtsrb = map_tt100k_to_gtsrb(tt100k_code)
    expected_tt100k = map_gtsrb_to_tt100k(gtsrb_id)
    
    if mapped_gtsrb == gtsrb_id or expected_tt100k == tt100k_code.lower():
        return True, f"CONSENSUS VERIFIED (GTSRB Class {gtsrb_id} == TT100K '{tt100k_code}')"
    elif mapped_gtsrb is not None:
        return False, f"DOMAIN DISCORD (GTSRB: {gtsrb_id} vs TT100K: {tt100k_code} -> GTSRB {mapped_gtsrb})"
    else:
        # Category level check
        if tt100k_code.startswith("pl") and gtsrb_id in [0, 1, 2, 3, 4, 5, 7, 8]:
            return True, f"CATEGORY CONSENSUS (Speed Limit: GTSRB {gtsrb_id} ~ TT100K {tt100k_code})"
        if tt100k_code.startswith("w") and 18 <= gtsrb_id <= 31:
            return True, f"CATEGORY CONSENSUS (Danger/Warning: GTSRB {gtsrb_id} ~ TT100K {tt100k_code})"
        return False, f"CROSS-DOMAIN UNALIGNED (GTSRB: {gtsrb_id}, TT100K: {tt100k_code})"
