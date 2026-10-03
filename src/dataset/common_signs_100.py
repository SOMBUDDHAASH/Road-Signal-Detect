"""
Comprehensive Database of the 100 Most Common International Road Signs.
Standardized according to the Vienna Convention on Road Signs and Signals (1968),
MUTCD (Manual on Uniform Traffic Control Devices), and European Road Standards (StVO/TSRGD).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import os


@dataclass
class RoadSignInfo:
    id: int
    code: str
    name: str
    category: str  # Priority, Prohibitory, Mandatory, Danger, Special, Service, Direction, Supplementary
    shape: str     # Octagon, Circle, Triangle, Inverted Triangle, Diamond, Rectangle, Square
    primary_color: str
    secondary_color: str
    symbol_color: str
    has_text: bool
    text_content: Optional[str]
    has_number: bool
    number_content: Optional[str]
    description: str


# 100 Most Common International Traffic Signs
COMMON_ROAD_SIGNS_100: List[RoadSignInfo] = [
    # =========================================================================
    # 1. PRIORITY SIGNS (Vienna Convention Class B & MUTCD)
    # =========================================================================
    RoadSignInfo(1, "B-01", "Stop", "Priority", "Octagon", "Red", "White", "White", True, "STOP", False, None, "Complete stop required before proceeding at intersection"),
    RoadSignInfo(2, "B-02", "Give Way / Yield", "Priority", "Inverted Triangle", "White", "Red", "None", True, "YIELD", False, None, "Driver must give way to vehicles on intersecting road"),
    RoadSignInfo(3, "B-03", "Priority Road", "Priority", "Diamond", "Yellow", "White", "None", False, None, False, None, "Road users have right-of-way at all successive intersections"),
    RoadSignInfo(4, "B-04", "End of Priority Road", "Priority", "Diamond", "Yellow", "White", "Black", False, None, False, None, "Termination of continuous right-of-way priority status"),
    RoadSignInfo(5, "B-05", "Priority for Oncoming Traffic", "Priority", "Circle", "White", "Red", "Red/Black", False, None, False, None, "Oncoming vehicles have priority; red arrow yields to black arrow"),
    RoadSignInfo(6, "B-06", "Priority over Oncoming Traffic", "Priority", "Square", "Blue", "White", "White/Red", False, None, False, None, "Driver has priority over opposing direction; white arrow has right-of-way"),
    RoadSignInfo(7, "B-07", "All-Way Stop", "Priority", "Rectangle", "Red", "White", "White", True, "ALL WAY", False, None, "Traffic from all four intersecting approaches must come to a complete stop"),

    # =========================================================================
    # 2. PROHIBITORY & RESTRICTIVE SIGNS (Vienna Convention Class C)
    # =========================================================================
    RoadSignInfo(8, "C-01", "No Entry", "Prohibitory", "Circle", "Red", "White", "White", False, None, False, None, "Vehicular entry completely prohibited from this direction"),
    RoadSignInfo(9, "C-02", "Road Closed / No Vehicles", "Prohibitory", "Circle", "White", "Red", "None", False, None, False, None, "Road closed to all vehicles in both directions"),
    RoadSignInfo(10, "C-03", "Speed Limit 20 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "20", "Maximum vehicular speed permitted is 20 km/h"),
    RoadSignInfo(11, "C-04", "Speed Limit 30 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "30", "Maximum vehicular speed permitted is 30 km/h (often residential zone)"),
    RoadSignInfo(12, "C-05", "Speed Limit 50 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "50", "Standard urban default maximum speed limit"),
    RoadSignInfo(13, "C-06", "Speed Limit 60 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "60", "Maximum vehicular speed permitted is 60 km/h"),
    RoadSignInfo(14, "C-07", "Speed Limit 70 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "70", "Maximum vehicular speed permitted is 70 km/h"),
    RoadSignInfo(15, "C-08", "Speed Limit 80 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "80", "Maximum vehicular speed permitted is 80 km/h (standard secondary road)"),
    RoadSignInfo(16, "C-09", "Speed Limit 100 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "100", "Maximum vehicular speed permitted is 100 km/h (standard highway)"),
    RoadSignInfo(17, "C-10", "Speed Limit 120 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "120", "Maximum vehicular speed permitted is 120 km/h (expressway/motorway)"),
    RoadSignInfo(18, "C-11", "Speed Limit 130 km/h", "Prohibitory", "Circle", "White", "Red", "Black", False, None, True, "130", "Maximum speed limit on high-speed motorways"),
    RoadSignInfo(19, "C-12", "End of Speed Limit", "Prohibitory", "Circle", "White", "Black", "Black", False, None, True, "General", "Derestriction of speed restriction; return to standard baseline"),
    RoadSignInfo(20, "C-13", "End of All Restrictions", "Prohibitory", "Circle", "White", "Black", "Black", False, None, False, None, "Cancels all previous prohibitory regulations including speed and overtaking"),
    RoadSignInfo(21, "C-14", "No Overtaking / Passing", "Prohibitory", "Circle", "White", "Red", "Red/Black", False, None, False, None, "Prohibits all motor vehicles from overtaking other motor vehicles"),
    RoadSignInfo(22, "C-15", "End of No Overtaking", "Prohibitory", "Circle", "White", "Black", "Black/Gray", False, None, False, None, "Termination of overtaking prohibition for all motor vehicles"),
    RoadSignInfo(23, "C-16", "No Overtaking by Heavy Goods Vehicles", "Prohibitory", "Circle", "White", "Red", "Red/Black", False, None, False, None, "Vehicles exceeding 3.5 metric tons prohibited from passing"),
    RoadSignInfo(24, "C-17", "No Left Turn", "Prohibitory", "Circle", "White", "Red", "Black/Red", False, None, False, None, "Turning left at this intersection or junction is prohibited"),
    RoadSignInfo(25, "C-18", "No Right Turn", "Prohibitory", "Circle", "White", "Red", "Black/Red", False, None, False, None, "Turning right at this intersection or junction is prohibited"),
    RoadSignInfo(26, "C-19", "No U-Turn", "Prohibitory", "Circle", "White", "Red", "Black/Red", False, None, False, None, "Executing a 180-degree U-turn is strictly prohibited"),
    RoadSignInfo(27, "C-20", "No Parking", "Prohibitory", "Circle", "Blue", "Red", "Red", False, None, False, None, "Parking prohibited; waiting/stopping briefly to pick up passengers allowed"),
    RoadSignInfo(28, "C-21", "No Stopping / Clearway", "Prohibitory", "Circle", "Blue", "Red", "Red", False, None, False, None, "Both parking and stopping completely prohibited at all times"),
    RoadSignInfo(29, "C-22", "No Entry for Trucks / Heavy Goods", "Prohibitory", "Circle", "White", "Red", "Black", False, None, False, None, "Vehicles over designated commercial weight prohibited from entering"),
    RoadSignInfo(30, "C-23", "Maximum Weight Limit (e.g. 3.5t)", "Prohibitory", "Circle", "White", "Red", "Black", True, "t", True, "3.5", "Vehicles exceeding designated laden weight prohibited"),
    RoadSignInfo(31, "C-24", "Maximum Height Limit (e.g. 3.8m)", "Prohibitory", "Circle", "White", "Red", "Black", True, "m", True, "3.8", "Vehicles exceeding specified clearance height prohibited"),
    RoadSignInfo(32, "C-25", "Maximum Width Limit (e.g. 2.0m)", "Prohibitory", "Circle", "White", "Red", "Black", True, "m", True, "2.0", "Vehicles exceeding specified width prohibited from roadway"),
    RoadSignInfo(33, "C-26", "Maximum Axle Weight Limit", "Prohibitory", "Circle", "White", "Red", "Black", True, "t", True, "6.0", "Vehicles with axle loading exceeding limit prohibited"),
    RoadSignInfo(34, "C-27", "Minimum Distance Between Vehicles", "Prohibitory", "Circle", "White", "Red", "Black", True, "m", True, "50", "Vehicles must maintain at least the specified following distance"),
    RoadSignInfo(35, "C-28", "No Pedestrians", "Prohibitory", "Circle", "White", "Red", "Black", False, None, False, None, "Pedestrian movement prohibited on this roadway or corridor"),
    RoadSignInfo(36, "C-29", "No Bicycles / Cyclists", "Prohibitory", "Circle", "White", "Red", "Black", False, None, False, None, "Bicycle riding prohibited along this roadway"),
    RoadSignInfo(37, "C-30", "No Motorcycles", "Prohibitory", "Circle", "White", "Red", "Black", False, None, False, None, "Motorcycles and motorized two-wheelers prohibited"),
    RoadSignInfo(38, "C-31", "No Hazardous Materials Transport", "Prohibitory", "Circle", "White", "Red", "Black/Orange", False, None, False, None, "Vehicles carrying dangerous or flammable goods prohibited"),
    RoadSignInfo(39, "C-32", "No Horns / Auditory Signals", "Prohibitory", "Circle", "White", "Red", "Black", False, None, False, None, "Sounding vehicle horns or auditory warnings is prohibited (hospital/night zones)"),

    # =========================================================================
    # 3. MANDATORY SIGNS (Vienna Convention Class D)
    # =========================================================================
    RoadSignInfo(40, "D-01", "Ahead Only (Go Straight)", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must proceed straight ahead only; turning prohibited"),
    RoadSignInfo(41, "D-02", "Turn Right", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must make an immediate right turn"),
    RoadSignInfo(42, "D-03", "Turn Left", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must make an immediate left turn"),
    RoadSignInfo(43, "D-04", "Turn Right Ahead", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must turn right at the upcoming intersection"),
    RoadSignInfo(44, "D-05", "Turn Left Ahead", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must turn left at the upcoming intersection"),
    RoadSignInfo(45, "D-06", "Pass Either Side", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles may pass obstacle or traffic island on either left or right"),
    RoadSignInfo(46, "D-07", "Pass Right Side", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must pass traffic island or obstacle on right side"),
    RoadSignInfo(47, "D-08", "Pass Left Side", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Vehicles must pass traffic island or obstacle on left side"),
    RoadSignInfo(48, "D-09", "Roundabout Mandatory", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Roundabout circulation; traffic in ring has priority, yield on entry"),
    RoadSignInfo(49, "D-10", "Straight or Right Only", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Traffic must only proceed straight or turn right"),
    RoadSignInfo(50, "D-11", "Straight or Left Only", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Traffic must only proceed straight or turn left"),
    RoadSignInfo(51, "D-12", "Compulsory Cycle Track", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Designated route reserved solely for cyclists"),
    RoadSignInfo(52, "D-13", "Compulsory Footpath", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Designated route reserved solely for pedestrians"),
    RoadSignInfo(53, "D-14", "Compulsory Shared Cycle & Footpath", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Segregated or shared path reserved for pedestrians and cyclists"),
    RoadSignInfo(54, "D-15", "Minimum Speed Limit (e.g. 30)", "Mandatory", "Circle", "Blue", "White", "White", False, None, True, "30", "Vehicles must travel at least the specified minimum speed"),
    RoadSignInfo(55, "D-16", "End of Minimum Speed", "Mandatory", "Circle", "Blue", "White", "Red", False, None, True, "30", "Termination of mandatory minimum speed limit"),
    RoadSignInfo(56, "D-17", "Snow Chains Compulsory", "Mandatory", "Circle", "Blue", "White", "White", False, None, False, None, "Snow tire chains required on driving wheels to proceed"),

    # =========================================================================
    # 4. DANGER & WARNING SIGNS (Vienna Convention Class A & MUTCD)
    # =========================================================================
    RoadSignInfo(57, "A-01", "General Danger / Caution", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Hazardous section ahead not specified by other individual warning signs"),
    RoadSignInfo(58, "A-02", "Curve to Left", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Sharp dangerous roadway curve to the left ahead"),
    RoadSignInfo(59, "A-03", "Curve to Right", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Sharp dangerous roadway curve to the right ahead"),
    RoadSignInfo(60, "A-04", "Double Curve (First to Left)", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Series of sharp curves ahead, first winding to left"),
    RoadSignInfo(61, "A-05", "Double Curve (First to Right)", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Series of sharp curves ahead, first winding to right"),
    RoadSignInfo(62, "A-06", "Road Narrows on Both Sides", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Roadway width decreases from both left and right flanks ahead"),
    RoadSignInfo(63, "A-07", "Road Narrows on Right", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Right lane or roadway edge terminates or pinches"),
    RoadSignInfo(64, "A-08", "Traffic Signals Ahead", "Danger", "Triangle", "White", "Red", "Red/Yellow/Green", False, None, False, None, "Approaching intersection controlled by traffic signal lights"),
    RoadSignInfo(65, "A-09", "Pedestrian Crossing Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Marked pedestrian crosswalk ahead; yield to crossing pedestrians"),
    RoadSignInfo(66, "A-10", "Children Crossing / School Zone", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "School, kindergarten or playground zone; expect children on road"),
    RoadSignInfo(67, "A-11", "Cyclists Crossing Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Bicycle crossing or heavy cycling traffic entering roadway"),
    RoadSignInfo(68, "A-12", "Cattle / Farm Animals Crossing", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Livestock, cattle, or herd movement across roadway"),
    RoadSignInfo(69, "A-13", "Wild Animals Crossing", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Deer, boar, or wildlife collision risk in forested/rural zone"),
    RoadSignInfo(70, "A-14", "Road Works / Construction", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Active construction, road maintenance personnel, or machinery on roadway"),
    RoadSignInfo(71, "A-15", "Slippery Road Surface", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Roadway prone to loss of traction under rain, ice, or loose gravel"),
    RoadSignInfo(72, "A-16", "Uneven Road / Bumps", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Severe road depressions, bumps, or uneven pavement"),
    RoadSignInfo(73, "A-17", "Speed Bump Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Traffic calming artificial speed bump ahead; reduce speed"),
    RoadSignInfo(74, "A-18", "Steep Hill Downward", "Danger", "Triangle", "White", "Red", "Black", True, "%", True, "10", "Steep downward grade; check brakes and use lower gear"),
    RoadSignInfo(75, "A-19", "Steep Hill Upward", "Danger", "Triangle", "White", "Red", "Black", True, "%", True, "12", "Steep incline ahead; watch for slow-moving heavy commercial trucks"),
    RoadSignInfo(76, "A-20", "Falling Rocks Hazard", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Risk of rockfall or debris on mountain road pass"),
    RoadSignInfo(77, "A-21", "Two-Way Traffic Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Transition from one-way street or divided highway to opposing two-way flow"),
    RoadSignInfo(78, "A-22", "Roundabout Warning Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Circular traffic intersection ahead; be prepared to yield"),
    RoadSignInfo(79, "A-23", "Crossroad with Minor Roads", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Approaching intersection where driver has right-of-way over minor cross traffic"),
    RoadSignInfo(80, "A-24", "Railway Crossing with Barrier Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Railroad level crossing equipped with gates/barriers ahead"),
    RoadSignInfo(81, "A-25", "Railway Crossing without Barrier Ahead", "Danger", "Triangle", "White", "Red", "Black", False, None, False, None, "Unprotected railroad crossing without automatic gates; stop and look"),

    # =========================================================================
    # 5. SPECIAL REGULATION & INFORMATIONAL SIGNS (Vienna Convention Class E & F)
    # =========================================================================
    RoadSignInfo(82, "E-01", "Motorway / Highway Entrance", "Special", "Rectangle", "Blue", "White", "White", False, None, False, None, "Start of high-speed controlled-access motorway / Autobahn regulations"),
    RoadSignInfo(83, "E-02", "End of Motorway", "Special", "Rectangle", "Blue", "White", "Red/White", False, None, False, None, "Termination of motorway regulations; return to normal road rules"),
    RoadSignInfo(84, "E-03", "One-Way Street", "Special", "Rectangle", "Blue", "White", "White", True, "ONE WAY", False, None, "Vehicular traffic must only proceed in direction of indicated arrow"),
    RoadSignInfo(85, "E-04", "Pedestrian Zone", "Special", "Rectangle", "White", "Black", "Black", True, "ZONE", False, None, "Area reserved exclusively for pedestrians; motorized traffic prohibited"),
    RoadSignInfo(86, "E-05", "Living Street / Residential Home Zone", "Special", "Rectangle", "Blue", "White", "White", False, None, False, None, "Children play on street; vehicles must travel at walking pace (max 7-10 km/h)"),
    RoadSignInfo(87, "E-06", "End of Living Street", "Special", "Rectangle", "Blue", "White", "Red/White", False, None, False, None, "Termination of residential home zone speed and pedestrian rights"),
    RoadSignInfo(88, "E-07", "Bus Lane", "Special", "Rectangle", "Blue", "White", "White", True, "BUS", False, None, "Dedicated traffic lane reserved strictly for scheduled public transit buses"),
    RoadSignInfo(89, "E-08", "Taxi Rank", "Special", "Rectangle", "Blue", "White", "White", True, "TAXI", False, None, "Designated parking/waiting bay for licensed taxi cabs"),
    RoadSignInfo(90, "E-09", "Parking Area", "Service", "Square", "Blue", "White", "White", True, "P", False, None, "Designated public vehicle parking facility or zone"),
    RoadSignInfo(91, "E-10", "Hospital / Medical Center", "Service", "Square", "Blue", "White", "Red/White", True, "H", False, None, "Hospital or emergency medical care facility nearby; keep quiet"),
    RoadSignInfo(92, "E-11", "Filling Station / Petrol Gas", "Service", "Square", "Blue", "White", "White/Black", False, None, False, None, "Motor fuel refueling station located ahead"),
    RoadSignInfo(93, "E-12", "Electric Vehicle Charging Station", "Service", "Square", "Blue", "White", "Green/White", True, "EV", False, None, "Dedicated high-voltage charging point for battery electric vehicles"),
    RoadSignInfo(94, "E-13", "Dead End / No Through Road", "Service", "Square", "Blue", "White", "Red/White", False, None, False, None, "Street has no vehicular outlet; terminates in cul-de-sac"),
    RoadSignInfo(95, "E-14", "Emergency SOS Telephone", "Service", "Square", "Blue", "White", "White/Red", True, "SOS", False, None, "Roadside emergency breakdown emergency telephone booth"),
    RoadSignInfo(96, "E-15", "Tunnel Ahead", "Special", "Square", "Blue", "White", "White/Black", False, None, False, None, "Approaching road tunnel; headlights mandatory, no turning, keep distance"),
    RoadSignInfo(97, "E-16", "Emergency Escape Ramp", "Special", "Rectangle", "Blue", "White", "Red/White", False, None, False, None, "Arresting bed runaway truck ramp for brake failure on steep declines"),

    # =========================================================================
    # 6. SUPPLEMENTARY & TEMPORARY PANELS (Class H & Work Zones)
    # =========================================================================
    RoadSignInfo(98, "H-01", "Distance Panel (e.g. 500m)", "Supplementary", "Rectangle", "White", "Black", "Black", True, "m", True, "500", "Indicates distance to hazard or beginning of regulation"),
    RoadSignInfo(99, "H-02", "When Wet / Rain Hazard", "Supplementary", "Rectangle", "White", "Black", "Black", False, None, False, None, "Regulation applies exclusively when road surface is damp, rain-slick, or icy"),
    RoadSignInfo(100, "W-01", "Construction Work Zone Detour", "Supplementary", "Rectangle", "Orange", "Black", "Black", True, "DETOUR", False, None, "Temporary traffic diversion around closed road construction zone")
]


def get_road_sign_by_id(sign_id: int) -> Optional[RoadSignInfo]:
    """Look up a sign by its 1-100 ID."""
    for sign in COMMON_ROAD_SIGNS_100:
        if sign.id == sign_id:
            return sign
    return None


def get_signs_by_category(category: str) -> List[RoadSignInfo]:
    """Filter signs by functional category."""
    cat_lower = category.lower()
    return [s for s in COMMON_ROAD_SIGNS_100 if cat_lower in s.category.lower()]


def get_signs_with_numbers() -> List[RoadSignInfo]:
    """Retrieve all signs containing numerical speed, weight, or distance values."""
    return [s for s in COMMON_ROAD_SIGNS_100 if s.has_number]


def get_signs_with_text() -> List[RoadSignInfo]:
    """Retrieve all signs containing textual words or acronyms."""
    return [s for s in COMMON_ROAD_SIGNS_100 if s.has_text]


def export_to_json(target_path: str = "data/road_signs_100.json"):
    """Export the 100 signs database to standard JSON."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    data = [s.__dict__ for s in COMMON_ROAD_SIGNS_100]
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return target_path


if __name__ == "__main__":
    p = export_to_json()
    print(f"Exported {len(COMMON_ROAD_SIGNS_100)} road signs to {p}")
