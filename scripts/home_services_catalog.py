"""Domain reference data for the synthetic home-services dataset.

Everything here is hand-written domain knowledge (service catalog, market price ranges
for the Austin TX metro in 2026, service areas, platforms). The generator samples
businesses, people, estimates, reviews and posts from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Categories --------------------------------------------------------------

CATEGORIES = [
    {"category_id": "plumbing", "name": "Plumbing", "trade_noun": "plumber",
     "license_authority": "Texas State Board of Plumbing Examiners (TSBPE)",
     "license_required": True, "in_dataset": True},
    {"category_id": "electrical", "name": "Electrical", "trade_noun": "electrician",
     "license_authority": "Texas Department of Licensing and Regulation (TDLR)",
     "license_required": True, "in_dataset": True},
    {"category_id": "hvac", "name": "HVAC (Heating & Air Conditioning)", "trade_noun": "HVAC tech",
     "license_authority": "Texas Department of Licensing and Regulation (TDLR)",
     "license_required": True, "in_dataset": True},
    {"category_id": "roofing", "name": "Roofing & Gutters", "trade_noun": "roofer",
     "license_authority": None, "license_required": False, "in_dataset": True},
    {"category_id": "landscaping", "name": "Landscaping (front yard, backyard, irrigation, trees)",
     "trade_noun": "landscaper",
     "license_authority": "Texas Commission on Environmental Quality (TCEQ) for irrigation",
     "license_required": False, "in_dataset": True},
    # Planned for a later phase; kept in the taxonomy so ids stay stable.
    {"category_id": "general_contractor", "name": "General Contractor", "trade_noun": "general contractor",
     "license_authority": None, "license_required": False, "in_dataset": False},
    {"category_id": "kitchen_remodel", "name": "Kitchen Remodeling", "trade_noun": "remodeler",
     "license_authority": None, "license_required": False, "in_dataset": False},
    {"category_id": "bathroom_remodel", "name": "Bathroom Remodeling", "trade_noun": "remodeler",
     "license_authority": None, "license_required": False, "in_dataset": False},
    {"category_id": "painting", "name": "Painting", "trade_noun": "painter",
     "license_authority": None, "license_required": False, "in_dataset": False},
    {"category_id": "handyman", "name": "Handyman", "trade_noun": "handyman",
     "license_authority": None, "license_required": False, "in_dataset": False},
]

# --- Services ----------------------------------------------------------------

SUMMER = {5: 1.3, 6: 2.0, 7: 2.6, 8: 2.6, 9: 1.8, 10: 1.0}
WINTER = {11: 1.4, 12: 2.0, 1: 2.4, 2: 2.2}
SPRING = {3: 1.6, 4: 2.2, 5: 2.0, 6: 1.3}
SPRING_FALL = {3: 1.8, 4: 2.0, 5: 1.6, 9: 1.4, 10: 1.8, 11: 1.4}
HAIL = {3: 1.8, 4: 2.6, 5: 2.6, 6: 1.6}
FREEZE = {12: 1.5, 1: 1.8, 2: 2.2}


@dataclass
class Service:
    service_id: str
    category: str
    name: str
    pricing_model: str  # flat | hourly | per_unit | per_sqft | per_linear_ft | per_square | per_zone | per_visit | per_tree | per_year
    unit: str
    market_low: float
    market_high: float
    duration: str
    permit: bool
    problems: list[str]
    work: list[str]
    aliases: list[str] = field(default_factory=list)
    season: dict[int, float] | None = None
    core: bool = True  # most businesses in the category offer it
    qty: tuple[float, float] = (1, 1)  # typical job quantity in `unit`s


def _s(*args, **kwargs) -> Service:
    return Service(*args, **kwargs)


SERVICES: list[Service] = [
    # Plumbing
    _s("plb_drain_cleaning", "plumbing", "Drain cleaning", "flat", "job", 150, 375, "1-2 hours", False,
       ["the kitchen sink kept backing up", "the shower drain was barely draining", "the main line backed up into the tub"],
       ["snaked the line and cleared a grease clog", "cleared the drain and showed us the gunk that came out"],
       ["clogged drain", "snaking", "drain snake"], core=True),
    _s("plb_hydro_jetting", "plumbing", "Hydro jetting", "flat", "job", 350, 850, "2-4 hours", False,
       ["roots kept clogging our sewer line", "we had recurring backups every few months"],
       ["hydro-jetted the main line and ran a camera through it afterwards"],
       ["jetting", "hydrojet"], core=False),
    _s("plb_water_heater_tank", "plumbing", "Tank water heater replacement", "flat", "job", 1600, 3400, "3-6 hours", True,
       ["our water heater started leaking in the garage", "we had no hot water", "the 15-year-old water heater finally died"],
       ["replaced the 50-gallon gas water heater and brought the venting up to code", "hauled off the old tank and installed a new one with an expansion tank"],
       ["water heater", "hot water heater", "tank heater"], season=WINTER),
    _s("plb_tankless_install", "plumbing", "Tankless water heater installation", "flat", "job", 3800, 7200, "1 day", True,
       ["we wanted to switch to tankless", "our old tank kept running out of hot water"],
       ["installed a tankless unit, upsized the gas line and set up a recirculation pump"],
       ["tankless", "on-demand water heater"], core=False),
    _s("plb_leak_detection", "plumbing", "Leak detection", "flat", "job", 250, 650, "1-3 hours", False,
       ["our water bill doubled out of nowhere", "we heard water running behind the wall", "there was a warm spot on the floor"],
       ["found the leak with acoustic equipment without tearing up the floor", "pinpointed a pinhole leak in the wall"],
       ["leak", "water leak"]),
    _s("plb_slab_leak_repair", "plumbing", "Slab leak repair", "flat", "job", 2200, 6500, "1-2 days", True,
       ["we had a slab leak under the kitchen", "the floor was wet and warm near the hallway"],
       ["rerouted the hot water line through the attic instead of jackhammering the slab", "tunneled under the slab to fix the break"],
       ["slab leak", "foundation leak"], core=False),
    _s("plb_toilet", "plumbing", "Toilet repair or installation", "flat", "job", 225, 650, "1-2 hours", False,
       ["our toilet kept running all night", "the toilet was rocking and leaking at the base"],
       ["replaced the fill valve and flapper", "replaced the wax ring and flange and reset the toilet"],
       ["toilet", "running toilet"]),
    _s("plb_fixture_install", "plumbing", "Faucet, sink or disposal installation", "per_unit", "fixture", 175, 450, "1-2 hours", False,
       ["we bought a new kitchen faucet", "the garbage disposal stopped working"],
       ["installed the new faucet and replaced the corroded shutoff valves", "swapped in a new disposal"],
       ["faucet", "garbage disposal", "sink"], qty=(1, 3)),
    _s("plb_sewer_camera", "plumbing", "Sewer camera inspection", "flat", "job", 225, 475, "1-2 hours", False,
       ["we were buying a 1960s house and wanted the sewer line checked", "we had a sewer smell in the yard"],
       ["ran a camera down the line and sent us the video", "located a belly in the cast iron line"],
       ["camera inspection", "sewer scope"]),
    _s("plb_sewer_replacement", "plumbing", "Sewer line replacement", "per_linear_ft", "linear ft", 110, 260, "2-4 days", True,
       ["our cast iron sewer line collapsed", "roots had destroyed the clay sewer pipe"],
       ["replaced the line with PVC and handled the city permit and inspection"],
       ["sewer line", "main line replacement"], core=False, qty=(30, 90)),
    _s("plb_repipe", "plumbing", "Whole-house repipe", "flat", "job", 5500, 16000, "2-5 days", True,
       ["we kept getting pinhole leaks in our old copper", "the house still had galvanized pipes"],
       ["repiped the whole house in PEX and patched the drywall afterwards"],
       ["repipe", "PEX"], core=False, season=FREEZE),
    _s("plb_gas_line", "plumbing", "Gas line installation or repair", "flat", "job", 350, 2200, "half day", True,
       ["we smelled gas near the stove", "we needed a gas line run for an outdoor kitchen"],
       ["pressure-tested the gas system and repaired the leaking fitting", "ran a new gas line to the patio"],
       ["gas leak", "gas line"]),
    _s("plb_water_softener", "plumbing", "Water softener or filtration installation", "flat", "job", 1400, 3600, "half day", False,
       ["the hard water was ruining our fixtures", "we wanted whole-house filtration"],
       ["installed a softener loop and whole-house filter"],
       ["water softener", "filtration"], core=False),
    _s("plb_emergency", "plumbing", "Emergency plumbing call", "per_visit", "visit", 175, 350, "1-2 hours", False,
       ["a pipe burst in the attic during the freeze", "water was pouring out from under the sink at 11pm"],
       ["came out the same night, shut off the water and fixed the burst pipe"],
       ["emergency plumber", "burst pipe", "frozen pipe"], season=FREEZE),

    # Electrical
    _s("ele_troubleshooting", "electrical", "Electrical troubleshooting and repair", "hourly", "hour", 95, 165, "1-3 hours", False,
       ["half the outlets in the living room went dead", "a breaker kept tripping", "the lights were flickering"],
       ["traced the problem to a loose backstabbed outlet", "found a failing breaker and replaced it",
        "found a loose neutral at the panel that was causing the flicker"],
       ["breaker tripping", "flickering lights", "dead outlet"], qty=(1, 4)),
    _s("ele_panel_upgrade", "electrical", "Electrical panel upgrade", "flat", "job", 2400, 5200, "1 day", True,
       ["our house had an old Federal Pacific panel", "we needed a 200 amp panel for an EV and a pool"],
       ["swapped the old panel for a 200 amp panel and coordinated with Austin Energy", "upgraded the service and labeled every circuit"],
       ["panel", "breaker box", "200 amp", "service upgrade"]),
    _s("ele_ev_charger", "electrical", "EV charger installation", "flat", "job", 900, 2600, "half day", True,
       ["we bought an EV and needed a Level 2 charger", "we needed a 240V outlet in the garage"],
       ["ran a 60 amp circuit and mounted the wall charger", "installed a NEMA 14-50 in the garage"],
       ["EV charger", "Tesla charger", "level 2", "240v outlet"]),
    _s("ele_outlet_switch", "electrical", "Outlet and switch installation", "per_unit", "device", 125, 325, "1-3 hours", False,
       ["we wanted more outlets in the home office", "we needed USB outlets and dimmers"],
       ["added outlets and swapped switches for dimmers"],
       ["outlet", "switch", "dimmer"], qty=(2, 8)),
    _s("ele_gfci_code", "electrical", "GFCI/AFCI code corrections", "per_unit", "device", 150, 300, "half day", False,
       ["our home inspection flagged missing GFCI outlets", "the buyer's inspector wrote up the wiring"],
       ["fixed every item on the inspection report and gave us documentation"],
       ["GFCI", "inspection repairs", "code corrections"], qty=(3, 10)),
    _s("ele_ceiling_fan", "electrical", "Ceiling fan installation", "per_unit", "fan", 175, 475, "1-3 hours", False,
       ["we wanted ceiling fans in the bedrooms", "our patio needed a fan for the summer"],
       ["installed fans with new fan-rated boxes"],
       ["ceiling fan", "fan install"], qty=(1, 4), season=SUMMER),
    _s("ele_recessed_lighting", "electrical", "Recessed lighting installation", "per_unit", "fixture", 175, 375, "1 day", False,
       ["the living room was too dark", "we wanted can lights in the kitchen"],
       ["installed LED wafer lights with minimal drywall damage"],
       ["can lights", "recessed lights", "pot lights"], qty=(4, 12)),
    _s("ele_generator", "electrical", "Whole-home standby generator installation", "flat", "job", 9500, 19000, "1-3 days", True,
       ["after the freeze we wanted a standby generator", "we lost power for four days and never wanted that again"],
       ["installed a standby generator with an automatic transfer switch and handled permits"],
       ["generator", "standby generator", "transfer switch"], core=False, season=FREEZE),
    _s("ele_rewire", "electrical", "Whole-house rewiring", "per_sqft", "sq ft", 4.5, 11, "3-7 days", True,
       ["our 1950s house still had knob-and-tube wiring", "insurance wouldn't cover the old wiring"],
       ["rewired the whole house and patched every hole"],
       ["rewire", "knob and tube", "aluminum wiring"], core=False, qty=(1100, 2200)),
    _s("ele_surge", "electrical", "Whole-home surge protection", "flat", "job", 350, 850, "1-2 hours", False,
       ["a lightning storm fried our TV", "we wanted surge protection after the outage"],
       ["installed a whole-home surge protector at the panel"],
       ["surge protector"]),
    _s("ele_landscape_lighting", "electrical", "Landscape and outdoor lighting", "flat", "job", 1800, 6500, "1-2 days", False,
       ["we wanted the front yard lit up at night", "the patio had no lighting"],
       ["installed low-voltage path and uplighting on a timer"],
       ["outdoor lighting", "landscape lighting", "uplighting"], core=False),
    _s("ele_smart_home", "electrical", "Smart switches and lighting control", "flat", "job", 400, 2500, "1 day", False,
       ["we wanted smart switches throughout the house", "we wanted lights we could control from our phones"],
       ["installed smart switches and set up scenes in the app"],
       ["smart switches", "smart home"], core=False),

    # HVAC
    _s("hvac_ac_diagnostic", "hvac", "AC diagnostic visit", "flat", "visit", 89, 189, "1 hour", False,
       ["the AC was blowing warm air", "the AC stopped cooling in the middle of July", "the outdoor unit wasn't running"],
       ["diagnosed the problem within an hour"],
       ["AC not cooling", "service call", "diagnostic fee"], season=SUMMER),
    _s("hvac_ac_repair", "hvac", "AC repair", "flat", "job", 250, 1500, "1-4 hours", False,
       ["the AC quit on a 105 degree day", "ice was building up on the AC lines", "the condenser fan stopped spinning"],
       ["replaced the capacitor and contactor", "found and repaired a refrigerant leak at the coil", "replaced the condenser fan motor"],
       ["AC repair", "capacitor", "refrigerant leak", "freon"], season=SUMMER),
    _s("hvac_tune_up", "hvac", "AC or furnace tune-up", "flat", "visit", 89, 199, "1 hour", False,
       ["we wanted the system checked before summer", "it was time for the annual tune-up"],
       ["cleaned the coils, checked refrigerant and flushed the drain line"],
       ["tune-up", "maintenance", "AC check"], season=SPRING),
    _s("hvac_ac_replacement", "hvac", "Central AC system replacement", "flat", "system", 7500, 16500, "1-2 days", True,
       ["our 18-year-old AC finally died", "the AC needed R-22 refrigerant we couldn't get anymore", "the system couldn't keep up with the heat"],
       ["replaced the condenser, coil and furnace with a high-efficiency system", "did a load calculation and installed a properly sized 4-ton system"],
       ["new AC", "AC replacement", "new system", "condenser replacement"], season=SUMMER),
    _s("hvac_heat_pump", "hvac", "Heat pump installation", "flat", "system", 9000, 19000, "1-2 days", True,
       ["we wanted to go electric with a heat pump", "our gas furnace and AC were both at end of life"],
       ["installed a variable-speed heat pump and walked us through the rebates"],
       ["heat pump"], core=False),
    _s("hvac_furnace_repair", "hvac", "Furnace or heating repair", "flat", "job", 200, 1200, "1-3 hours", False,
       ["the heat stopped working during a cold snap", "the furnace was short cycling"],
       ["replaced the flame sensor and igniter", "fixed the inducer motor"],
       ["furnace", "no heat", "heater repair"], season=WINTER),
    _s("hvac_furnace_replacement", "hvac", "Furnace replacement", "flat", "system", 4500, 9500, "1 day", True,
       ["the furnace had a cracked heat exchanger", "our 25-year-old furnace was done"],
       ["installed a new furnace and replaced the flue"],
       ["new furnace"], core=False, season=WINTER),
    _s("hvac_ductwork", "hvac", "Ductwork replacement or repair", "flat", "job", 3500, 9500, "1-3 days", False,
       ["some rooms were 10 degrees hotter than others", "the attic ducts were crushed and leaking"],
       ["replaced the attic ductwork and balanced the airflow"],
       ["ducts", "ductwork", "hot rooms"], core=False),
    _s("hvac_duct_cleaning", "hvac", "Duct cleaning", "flat", "job", 350, 850, "half day", False,
       ["the house was dusty no matter what we did", "we moved into a house with pets"],
       ["cleaned every duct and showed us before and after photos"],
       ["duct cleaning"], core=False),
    _s("hvac_mini_split", "hvac", "Ductless mini-split installation", "per_zone", "zone", 3200, 6800, "1-2 days", True,
       ["our garage office was unbearable in summer", "the converted attic had no AC"],
       ["installed a mini-split head and hid the line set neatly"],
       ["mini split", "ductless"], core=False, qty=(1, 3)),
    _s("hvac_thermostat", "hvac", "Smart thermostat installation", "flat", "job", 175, 450, "1 hour", False,
       ["we wanted a smart thermostat", "the old thermostat stopped responding"],
       ["installed the new thermostat and ran a C-wire"],
       ["thermostat", "smart thermostat"]),
    _s("hvac_air_quality", "hvac", "Indoor air quality (filtration, UV, dehumidifier)", "flat", "job", 600, 2800, "half day", False,
       ["allergies were brutal in cedar season", "the house felt humid and musty"],
       ["installed a media filter cabinet and a whole-house dehumidifier"],
       ["air purifier", "UV light", "dehumidifier", "air quality"], core=False),
    _s("hvac_maintenance_plan", "hvac", "Annual maintenance plan", "per_year", "year", 150, 320, "2 visits per year", False,
       ["we wanted someone to check the system every spring and fall"],
       ["signed us up for twice-yearly visits with priority scheduling"],
       ["maintenance plan", "service agreement", "membership"]),

    # Roofing
    _s("roof_inspection", "roofing", "Roof inspection", "flat", "job", 0, 275, "1 hour", False,
       ["we were buying a house and wanted the roof checked", "we weren't sure if the hail did any damage"],
       ["went up on the roof and sent a photo report"],
       ["roof inspection", "free inspection"], season=HAIL),
    _s("roof_repair", "roofing", "Roof leak repair", "flat", "job", 350, 1800, "half day", False,
       ["water was dripping from the ceiling during a storm", "there was a stain spreading on the bedroom ceiling"],
       ["replaced the cracked pipe boot and resealed the flashing", "fixed the valley that was leaking"],
       ["roof leak", "leak repair"], season=HAIL),
    _s("roof_shingle_replacement", "roofing", "Asphalt shingle roof replacement", "per_square", "square (100 sq ft)", 475, 800, "1-2 days", True,
       ["hail destroyed our roof", "our roof was 22 years old and losing granules", "insurance approved a full replacement"],
       ["tore off and replaced the whole roof in one day with impact-resistant shingles", "replaced the roof and cleaned up every nail"],
       ["new roof", "reroof", "roof replacement", "shingles"], season=HAIL, qty=(20, 38)),
    _s("roof_metal", "roofing", "Metal roof installation", "per_square", "square (100 sq ft)", 950, 1700, "3-5 days", True,
       ["we wanted a metal roof that would survive hail", "we wanted a standing seam roof for the farmhouse look"],
       ["installed a standing seam metal roof with snow guards and new flashing"],
       ["metal roof", "standing seam"], core=False, qty=(20, 35)),
    _s("roof_tile", "roofing", "Tile roof repair or replacement", "per_square", "square (100 sq ft)", 1000, 1900, "3-6 days", True,
       ["several clay tiles cracked in the storm", "the underlayment under our tile roof failed"],
       ["replaced the underlayment and reset the tiles"],
       ["tile roof", "clay tile"], core=False, qty=(5, 30)),
    _s("roof_flat", "roofing", "Flat roof (TPO) installation", "per_square", "square (100 sq ft)", 650, 1150, "1-3 days", True,
       ["our modern house has a flat roof that kept ponding", "the patio's flat roof was leaking"],
       ["installed a new TPO membrane with tapered insulation for drainage"],
       ["flat roof", "TPO", "modified bitumen"], core=False, qty=(5, 20)),
    _s("roof_storm_claim", "roofing", "Storm damage inspection and insurance claim help", "flat", "job", 0, 350, "1-2 hours", False,
       ["a hailstorm went through the neighborhood", "the neighbors were all getting new roofs after the storm"],
       ["documented the hail damage and met with the insurance adjuster"],
       ["hail damage", "insurance claim", "storm damage"], season=HAIL),
    _s("roof_emergency_tarp", "roofing", "Emergency roof tarping", "flat", "job", 350, 950, "1-2 hours", False,
       ["a tree limb punched a hole in the roof", "the storm ripped shingles off and it was still raining"],
       ["tarped the roof that night"],
       ["tarp", "emergency roofer"], season=HAIL),
    _s("roof_gutter_install", "roofing", "Gutter installation", "per_linear_ft", "linear ft", 9, 24, "1 day", False,
       ["water was pouring off the roof and eroding the beds", "our house had no gutters"],
       ["installed 6-inch seamless gutters with oversized downspouts"],
       ["gutters", "seamless gutters", "downspouts"], qty=(120, 240)),
    _s("roof_gutter_cleaning", "roofing", "Gutter cleaning", "flat", "job", 150, 375, "1-2 hours", False,
       ["the gutters were full of oak leaves", "the gutters overflowed every time it rained"],
       ["cleaned the gutters and flushed the downspouts"],
       ["gutter cleaning"], season={11: 2.0, 12: 1.6, 3: 1.4, 4: 1.4}),
    _s("roof_skylight", "roofing", "Skylight installation or repair", "flat", "job", 900, 3200, "1 day", False,
       ["our skylight leaked every time it rained", "we wanted more light in the kitchen"],
       ["replaced the skylight and reflashed it properly"],
       ["skylight"], core=False),
    _s("roof_ventilation", "roofing", "Attic ventilation", "flat", "job", 450, 1800, "half day", False,
       ["the attic was 150 degrees in summer", "the upstairs was always hot"],
       ["added ridge vents and fixed the soffit intake"],
       ["ridge vent", "attic fan", "ventilation"], core=False),

    # Landscaping
    _s("land_lawn_maintenance", "landscaping", "Lawn mowing and maintenance", "per_visit", "visit", 45, 95, "1 hour", False,
       ["we needed someone to keep up with the lawn", "the HOA sent us a letter about the yard"],
       ["mowed, edged and blew every week like clockwork"],
       ["mowing", "lawn care", "yard service"], season={4: 1.6, 5: 1.8, 6: 1.6, 7: 1.4, 8: 1.4, 9: 1.4}),
    _s("land_design", "landscaping", "Landscape design", "flat", "job", 450, 3500, "2-4 weeks", False,
       ["we had a blank yard and no idea where to start", "we wanted a plan before spending money"],
       ["drew up a full plan with native plants and a phased budget"],
       ["landscape design", "landscape plan"], core=False),
    _s("land_front_yard", "landscaping", "Front yard makeover", "flat", "job", 6000, 28000, "1-3 weeks", False,
       ["our front yard was mostly dead grass", "we wanted more curb appeal before selling"],
       ["replaced the front lawn with native beds, limestone borders and a new walkway", "redid the front beds, added a dry creek bed and path lighting"],
       ["front yard", "curb appeal"], season=SPRING_FALL),
    _s("land_backyard", "landscaping", "Backyard makeover", "flat", "job", 12000, 65000, "2-6 weeks", False,
       ["we wanted to actually use our backyard", "the backyard was a mud pit every time it rained"],
       ["built a flagstone patio, a fire pit area and new planting beds", "regraded the yard, built a pergola and planted shade trees"],
       ["backyard", "outdoor living", "backyard remodel"], core=False, season=SPRING_FALL),
    _s("land_sod", "landscaping", "Sod installation", "per_sqft", "sq ft", 1.6, 3.8, "1-2 days", False,
       ["the lawn died during the drought", "we needed new grass before the house went on the market"],
       ["tilled the soil and laid Zoysia sod"],
       ["sod", "new grass", "St. Augustine", "Zoysia", "Bermuda"], season=SPRING_FALL, qty=(800, 3000)),
    _s("land_xeriscape", "landscaping", "Xeriscaping and native plants", "per_sqft", "sq ft", 9, 26, "1-2 weeks", False,
       ["we were tired of watering grass in 100 degree heat", "we wanted a drought-tolerant yard"],
       ["converted the lawn to native plants, decomposed granite and river rock"],
       ["xeriscape", "drought tolerant", "native plants"], core=False, season=SPRING_FALL, qty=(300, 1500)),
    _s("land_irrigation_install", "landscaping", "Irrigation system installation", "per_zone", "zone", 650, 1250, "1-2 days", True,
       ["we wanted a sprinkler system", "the yard needed drip irrigation for the new beds"],
       ["installed a new irrigation system with a smart controller and backflow preventer"],
       ["sprinkler system", "irrigation", "drip irrigation"], core=False, qty=(4, 9)),
    _s("land_irrigation_repair", "landscaping", "Sprinkler and irrigation repair", "hourly", "hour", 80, 135, "1-3 hours", False,
       ["a sprinkler head was geysering", "one zone stopped working"],
       ["fixed the broken heads and replaced a bad valve"],
       ["sprinkler repair", "irrigation repair"], qty=(1, 3)),
    _s("land_tree_trimming", "landscaping", "Tree trimming", "per_tree", "tree", 275, 1200, "half day", False,
       ["the live oaks were hanging over the roof", "branches were rubbing the house"],
       ["trimmed the oaks and painted the cuts to prevent oak wilt"],
       ["tree trimming", "pruning", "tree service"], season={1: 1.6, 2: 1.6, 7: 1.4, 8: 1.4, 12: 1.4}, qty=(1, 4)),
    _s("land_tree_removal", "landscaping", "Tree removal", "per_tree", "tree", 650, 3800, "1 day", False,
       ["a dead tree was leaning toward the house", "the freeze killed a big tree in the backyard"],
       ["took the tree down in sections and ground the stump"],
       ["tree removal", "stump grinding"], core=False, qty=(1, 2)),
    _s("land_hardscape", "landscaping", "Patio and paver hardscaping", "per_sqft", "sq ft", 16, 38, "1-2 weeks", False,
       ["we wanted a patio for the backyard", "the old concrete patio was cracked"],
       ["built a paver patio with a seat wall"],
       ["patio", "pavers", "flagstone", "hardscape"], core=False, qty=(200, 600)),
    _s("land_retaining_wall", "landscaping", "Retaining wall", "per_sqft", "sq ft of wall face", 45, 110, "1-2 weeks", False,
       ["the slope in the backyard kept washing out", "we needed to terrace the hillside"],
       ["built a limestone block retaining wall with drainage behind it"],
       ["retaining wall", "limestone wall"], core=False, qty=(60, 300)),
    _s("land_drainage", "landscaping", "Yard drainage (French drains)", "per_linear_ft", "linear ft", 30, 85, "2-4 days", False,
       ["water pooled against the foundation every storm", "the backyard flooded every time it rained"],
       ["installed a French drain and regraded the side yard"],
       ["French drain", "drainage", "standing water"], core=False, qty=(40, 150)),
    _s("land_cleanup", "landscaping", "Seasonal cleanup", "flat", "job", 250, 900, "half day", False,
       ["the yard was overgrown after a long summer", "we needed the leaves and dead plants cleared"],
       ["cleared the beds, cut back the perennials and hauled everything off"],
       ["yard cleanup", "leaf removal"], season={2: 1.8, 3: 1.8, 11: 1.6, 12: 1.4}),
    _s("land_mulch_beds", "landscaping", "Mulch and bed installation", "flat", "job", 350, 1800, "1 day", False,
       ["the flower beds looked tired", "we wanted fresh beds along the front of the house"],
       ["edged the beds, added new plants and laid fresh mulch"],
       ["mulch", "flower beds", "planting"]),
]

SERVICE_BY_ID = {s.service_id: s for s in SERVICES}

# How people actually refer to a service in reviews and posts (default: name, lowercased).
CASUAL_NAMES = {
    "plb_water_heater_tank": "water heater replacement", "plb_tankless_install": "tankless water heater",
    "plb_toilet": "toilet repair", "plb_fixture_install": "faucet install", "plb_gas_line": "gas line work",
    "plb_water_softener": "water softener install", "plb_emergency": "emergency plumbing call",
    "plb_repipe": "repipe", "ele_troubleshooting": "electrical repair", "ele_outlet_switch": "outlet install",
    "ele_gfci_code": "inspection repairs", "ele_generator": "standby generator", "ele_rewire": "rewiring",
    "ele_surge": "surge protector", "ele_landscape_lighting": "outdoor lighting", "ele_smart_home": "smart switch install",
    "hvac_ac_diagnostic": "AC service call", "hvac_ac_repair": "AC repair", "hvac_tune_up": "AC tune-up",
    "hvac_ac_replacement": "new AC system", "hvac_heat_pump": "heat pump", "hvac_furnace_repair": "furnace repair",
    "hvac_furnace_replacement": "new furnace", "hvac_ductwork": "ductwork replacement",
    "hvac_air_quality": "air quality upgrade", "hvac_mini_split": "mini split", "hvac_thermostat": "thermostat install",
    "hvac_maintenance_plan": "maintenance plan", "roof_repair": "roof leak repair",
    "roof_shingle_replacement": "new roof", "roof_metal": "metal roof", "roof_tile": "tile roof repair",
    "roof_flat": "flat roof", "roof_storm_claim": "hail damage claim", "roof_emergency_tarp": "emergency tarp",
    "roof_gutter_install": "new gutters", "roof_skylight": "skylight repair", "roof_ventilation": "attic ventilation",
    "land_lawn_maintenance": "lawn care", "land_front_yard": "front yard redo", "land_backyard": "backyard remodel",
    "land_sod": "new sod", "land_xeriscape": "xeriscaping", "land_irrigation_install": "sprinkler system",
    "land_irrigation_repair": "sprinkler repair", "land_hardscape": "paver patio", "land_drainage": "French drain",
    "land_cleanup": "yard cleanup", "land_mulch_beds": "flower beds",
}


def casual_name(service_id: str) -> str:
    return CASUAL_NAMES.get(service_id, SERVICE_BY_ID[service_id].name.lower())

# --- Geography (Austin TX metro) --------------------------------------------

# home_era: typical construction years, which shapes problems (cast iron, old panels, builder-grade).
SERVICE_AREAS = [
    {"city": "Austin", "county": "Travis", "weight": 4.0, "neighborhoods": [
        {"name": "Hyde Park", "zip": "78751", "home_era": (1920, 1955)},
        {"name": "Travis Heights", "zip": "78704", "home_era": (1925, 1965)},
        {"name": "Zilker", "zip": "78704", "home_era": (1940, 2015)},
        {"name": "Barton Hills", "zip": "78704", "home_era": (1960, 1985)},
        {"name": "Allandale", "zip": "78757", "home_era": (1950, 1970)},
        {"name": "Crestview", "zip": "78757", "home_era": (1950, 1965)},
        {"name": "Tarrytown", "zip": "78703", "home_era": (1935, 1975)},
        {"name": "Mueller", "zip": "78723", "home_era": (2008, 2022)},
        {"name": "Windsor Park", "zip": "78723", "home_era": (1950, 1970)},
        {"name": "East Cesar Chavez", "zip": "78702", "home_era": (1925, 2020)},
        {"name": "Circle C Ranch", "zip": "78739", "home_era": (1990, 2010)},
        {"name": "North Loop", "zip": "78751", "home_era": (1935, 1960)},
        {"name": "Avery Ranch", "zip": "78717", "home_era": (2000, 2015)},
        {"name": "Southpark Meadows", "zip": "78748", "home_era": (1995, 2015)},
    ]},
    {"city": "Round Rock", "county": "Williamson", "weight": 1.6, "neighborhoods": [
        {"name": "Brushy Creek", "zip": "78681", "home_era": (1985, 2005)},
        {"name": "Teravista", "zip": "78665", "home_era": (2004, 2020)},
        {"name": "Forest Creek", "zip": "78664", "home_era": (1990, 2005)},
    ]},
    {"city": "Cedar Park", "county": "Williamson", "weight": 1.3, "neighborhoods": [
        {"name": "Buttercup Creek", "zip": "78613", "home_era": (1985, 2000)},
        {"name": "Twin Creeks", "zip": "78613", "home_era": (2000, 2015)},
    ]},
    {"city": "Pflugerville", "county": "Travis", "weight": 1.2, "neighborhoods": [
        {"name": "Falcon Pointe", "zip": "78660", "home_era": (2005, 2018)},
        {"name": "Blackhawk", "zip": "78660", "home_era": (2000, 2012)},
    ]},
    {"city": "Georgetown", "county": "Williamson", "weight": 1.2, "neighborhoods": [
        {"name": "Sun City", "zip": "78633", "home_era": (1996, 2020)},
        {"name": "Wolf Ranch", "zip": "78628", "home_era": (2010, 2022)},
        {"name": "Old Town", "zip": "78626", "home_era": (1900, 1960)},
    ]},
    {"city": "Leander", "county": "Williamson", "weight": 0.9, "neighborhoods": [
        {"name": "Crystal Falls", "zip": "78641", "home_era": (2000, 2018)},
        {"name": "Travisso", "zip": "78641", "home_era": (2015, 2024)},
    ]},
    {"city": "Kyle", "county": "Hays", "weight": 0.8, "neighborhoods": [
        {"name": "Plum Creek", "zip": "78640", "home_era": (2002, 2020)},
    ]},
    {"city": "Buda", "county": "Hays", "weight": 0.6, "neighborhoods": [
        {"name": "Sunfield", "zip": "78610", "home_era": (2008, 2022)},
    ]},
    {"city": "Lakeway", "county": "Travis", "weight": 0.6, "neighborhoods": [
        {"name": "Rough Hollow", "zip": "78734", "home_era": (2005, 2022)},
    ]},
    {"city": "Dripping Springs", "county": "Hays", "weight": 0.5, "neighborhoods": [
        {"name": "Belterra", "zip": "78737", "home_era": (2003, 2020)},
    ]},
]

# Cities a business based in X usually also serves.
NEARBY = {
    "Austin": ["Round Rock", "Pflugerville", "Cedar Park", "Lakeway", "Buda", "Kyle", "Dripping Springs"],
    "Round Rock": ["Austin", "Pflugerville", "Georgetown", "Cedar Park", "Leander"],
    "Cedar Park": ["Leander", "Austin", "Round Rock", "Georgetown"],
    "Pflugerville": ["Austin", "Round Rock", "Georgetown"],
    "Georgetown": ["Round Rock", "Leander", "Cedar Park", "Pflugerville"],
    "Leander": ["Cedar Park", "Georgetown", "Round Rock", "Austin"],
    "Kyle": ["Buda", "Austin", "Dripping Springs"],
    "Buda": ["Kyle", "Austin", "Dripping Springs"],
    "Lakeway": ["Austin", "Dripping Springs", "Cedar Park"],
    "Dripping Springs": ["Austin", "Lakeway", "Buda", "Kyle"],
}

# --- Platforms ---------------------------------------------------------------

REVIEW_PLATFORMS = [
    {"platform": "google", "name": "Google Business Profile", "rating_scale": 5, "has_title": False,
     "has_cost_field": False, "has_sub_ratings": False, "verified_hire": False, "owner_responses": True},
    {"platform": "yelp", "name": "Yelp", "rating_scale": 5, "has_title": False,
     "has_cost_field": False, "has_sub_ratings": False, "verified_hire": False, "owner_responses": True},
    {"platform": "angi", "name": "Angi", "rating_scale": 5, "has_title": True, "has_cost_field": True,
     "has_sub_ratings": True, "verified_hire": True, "owner_responses": True,
     "sub_rating_keys": ["quality", "value", "punctuality", "professionalism", "responsiveness"]},
    {"platform": "thumbtack", "name": "Thumbtack", "rating_scale": 5, "has_title": False,
     "has_cost_field": True, "has_sub_ratings": False, "verified_hire": True, "owner_responses": True},
    {"platform": "bbb", "name": "Better Business Bureau", "rating_scale": 5, "has_title": False,
     "has_cost_field": False, "has_sub_ratings": False, "verified_hire": False, "owner_responses": True,
     "notes": "Includes formal complaints with a resolution status as well as customer reviews."},
    {"platform": "houzz", "name": "Houzz", "rating_scale": 5, "has_title": True, "has_cost_field": True,
     "has_sub_ratings": False, "verified_hire": False, "owner_responses": True,
     "notes": "Skews toward design-heavy projects: landscaping, roofing, outdoor lighting."},
]

SOCIAL_PLATFORMS = [
    {"platform": "nextdoor", "name": "Nextdoor", "content": "Neighborhood recommendation threads and warnings.",
     "has_threads": True, "has_video": False},
    {"platform": "facebook", "name": "Facebook Groups", "content": "Local community and homeowner groups; recommendations, price checks, complaints.",
     "has_threads": True, "has_video": False},
    {"platform": "reddit", "name": "Reddit", "content": "City subreddits; recommendation threads and PSAs.",
     "has_threads": True, "has_video": False},
    {"platform": "tiktok", "name": "TikTok", "content": "Short videos: business showcases, homeowner storytimes, price reveals.",
     "has_threads": False, "has_video": True},
    {"platform": "instagram", "name": "Instagram", "content": "Before/after project photos from businesses and customers.",
     "has_threads": False, "has_video": True},
]

FACEBOOK_GROUPS = {
    "Austin": ["Austin Homeowners Network", "South Austin Neighbors", "Central Austin Old House Owners"],
    "Round Rock": ["Round Rock Neighbors"], "Cedar Park": ["Cedar Park & Leander Families"],
    "Leander": ["Cedar Park & Leander Families"], "Pflugerville": ["Pflugerville Community Board"],
    "Georgetown": ["Georgetown TX Community", "Sun City Texas Residents"], "Kyle": ["Kyle & Buda Neighbors"],
    "Buda": ["Kyle & Buda Neighbors"], "Lakeway": ["Lakeway Area Residents"],
    "Dripping Springs": ["Dripping Springs Community"],
}

SUBREDDITS = {"Austin": "r/Austin", "Round Rock": "r/RoundRock", "Georgetown": "r/GeorgetownTX",
              "Cedar Park": "r/Austin", "Pflugerville": "r/Austin", "Leander": "r/Austin", "Kyle": "r/Austin",
              "Buda": "r/Austin", "Lakeway": "r/Austin", "Dripping Springs": "r/Austin"}

# --- Business attributes -----------------------------------------------------

BRANDS = {
    "plumbing": ["Rheem", "Bradford White", "A.O. Smith", "Rinnai", "Navien", "Kohler", "Moen", "Delta"],
    "electrical": ["Square D", "Eaton", "Siemens", "Generac", "Kohler", "Tesla Wall Connector", "ChargePoint", "Lutron"],
    "hvac": ["Trane", "Carrier", "Lennox", "Goodman", "Rheem", "Daikin", "Mitsubishi", "American Standard"],
    "roofing": ["GAF", "Owens Corning", "CertainTeed", "Tamko", "Malarkey"],
    "landscaping": ["Rain Bird", "Hunter", "Rachio", "Belgard", "Pavestone"],
}

LICENSES = {
    "plumbing": [("Master Plumber", "TSBPE", "M"), ("Responsible Master Plumber", "TSBPE", "RMP")],
    "electrical": [("Electrical Contractor", "TDLR", "TECL"), ("Master Electrician", "TDLR", "ME")],
    "hvac": [("Air Conditioning and Refrigeration Contractor, Class A", "TDLR", "TACLA")],
    "roofing": [],
    "landscaping": [("Licensed Irrigator", "TCEQ", "LI")],
}

EXPERT_LICENSES = {
    "plumbing": {"senior": ("Master Plumber", "TSBPE", "M"), "mid": ("Journeyman Plumber", "TSBPE", "J"),
                 "junior": ("Plumber's Apprentice", "TSBPE", "AP")},
    "electrical": {"senior": ("Master Electrician", "TDLR", "ME"), "mid": ("Journeyman Electrician", "TDLR", "JE"),
                   "junior": ("Apprentice Electrician", "TDLR", "AE")},
    "hvac": {"senior": ("ACR Contractor, Class A", "TDLR", "TACLA"), "mid": ("Registered ACR Technician", "TDLR", "ACR"),
             "junior": ("Registered ACR Technician", "TDLR", "ACR")},
    "roofing": {},
    "landscaping": {"senior": ("Licensed Irrigator", "TCEQ", "LI"), "mid": ("Irrigation Technician", "TCEQ", "IT")},
}

CERTIFICATIONS = {
    "plumbing": ["Backflow Prevention Assembly Tester", "Medical Gas Installer", "Water Treatment Specialist",
                 "Tankless manufacturer-certified installer"],
    "electrical": ["Generator manufacturer-certified dealer", "EV charger certified installer", "OSHA 30"],
    "hvac": ["NATE Certified", "EPA Section 608 Universal", "Manufacturer Factory Authorized Dealer",
             "ACCA member"],
    "roofing": ["Shingle manufacturer certified installer", "HAAG Certified Roof Inspector",
                "OSHA 30", "NRCA member", "Metal roofing manufacturer certified"],
    "landscaping": ["ISA Certified Arborist", "Texas Certified Landscape Professional",
                    "Hardscape installer certification (ICPI)", "Native Plant Society member"],
}

EXPERT_ROLES = {
    "plumbing": {"senior": ["Master Plumber", "Lead Plumber"], "mid": ["Journeyman Plumber", "Service Plumber"],
                 "junior": ["Plumber's Apprentice"], "office": ["Service Manager", "Estimator"]},
    "electrical": {"senior": ["Master Electrician", "Lead Electrician"], "mid": ["Journeyman Electrician"],
                   "junior": ["Apprentice Electrician"], "office": ["Project Estimator"]},
    "hvac": {"senior": ["Lead HVAC Technician", "Install Manager"], "mid": ["HVAC Service Technician"],
             "junior": ["HVAC Apprentice"], "office": ["Comfort Advisor"]},
    "roofing": {"senior": ["Project Manager", "Crew Lead"], "mid": ["Roofing Crew Lead", "Repair Technician"],
                "junior": ["Roofer"], "office": ["Insurance Claims Specialist", "Estimator"]},
    "landscaping": {"senior": ["Landscape Designer", "Certified Arborist", "Project Manager"],
                    "mid": ["Crew Lead", "Irrigation Technician"], "junior": ["Landscape Technician"],
                    "office": ["Design Consultant"]},
}

# --- Name pools ---------------------------------------------------------------

FIRST_NAMES = [
    "Carlos", "Maria", "James", "Aisha", "Wei", "Priya", "Michael", "Sofia", "David", "Fatima", "Daniel", "Grace",
    "Jose", "Linh", "Robert", "Keisha", "Ahmed", "Hannah", "Luis", "Mei", "Brandon", "Olivia", "Tyrell", "Elena",
    "Kevin", "Ana", "Marcus", "Yuki", "Juan", "Rachel", "Omar", "Nicole", "Trevor", "Imani", "Diego", "Laura",
    "Samuel", "Ngozi", "Ethan", "Camila", "Hector", "Jasmine", "Raj", "Emily", "Andre", "Kim", "Victor", "Megan",
    "Tomas", "Destiny", "Cody", "Gabriela", "Jamal", "Sarah", "Hiro", "Valeria", "Patrick", "Zainab", "Ruben",
    "Allison", "Bao", "Kayla", "Eduardo", "Tara", "Dustin", "Lucia", "Arjun", "Brianna", "Wyatt", "Rosa", "Noah",
    "Chioma", "Garrett", "Isabel", "Kenji", "Monica", "Ricardo", "Leah", "Felipe", "Shonda", "Colton", "Mariana",
    "Anthony", "Esperanza", "Blake", "Thu", "Javier", "Courtney", "Malik", "Ingrid", "Austin", "Paloma", "Reggie",
]
NICKNAMES = {"Michael": "Mike", "Robert": "Bob", "James": "Jim", "Daniel": "Danny", "David": "Dave",
             "Samuel": "Sam", "Anthony": "Tony", "Eduardo": "Eddie", "Patrick": "Pat", "Javier": "Javi",
             "Ricardo": "Rick", "Gabriela": "Gaby", "Valeria": "Val", "Nicole": "Nikki", "Thomas": "Tom",
             "Tomas": "Tommy", "Reggie": "Reg", "Victor": "Vic", "Hector": "Tito", "Allison": "Ali"}
LAST_NAMES = [
    "Garcia", "Nguyen", "Okafor", "Patel", "Martinez", "Johnson", "Kowalski", "Hernandez", "Chen", "Williams",
    "Rodriguez", "Brooks", "Delgado", "Kim", "Thompson", "Alvarado", "Mensah", "Reyes", "Sullivan", "Tran",
    "Castillo", "Washington", "Huang", "Morales", "Fischer", "Ramirez", "Abernathy", "Gonzalez", "Ibrahim",
    "Becker", "Torres", "Lindqvist", "Flores", "Haddad", "Vasquez", "Ortega", "Pham", "Carter", "Navarro",
    "Okonkwo", "Medina", "Holloway", "Santos", "Yamamoto", "Dunn", "Salazar", "Bishop", "Guerrero", "Price",
    "Mendoza", "Kaur", "Fuentes", "Wright", "Aguilar", "Bell", "Cruz", "Osei", "Hayes", "Rivas", "Sandoval",
    "Lowe", "Esparza", "Whitfield", "Contreras", "Ng", "Padilla", "McAllister", "Zamora", "Begay", "Ochoa",
]

PLACE_WORDS = ["Hill Country", "Bluebonnet", "Live Oak", "Cedar Ridge", "Limestone", "Lone Star", "Balcones",
               "Brushy Creek", "Onion Creek", "Pedernales", "Barton Creek", "Capitol City", "Mopac", "Lady Bird",
               "Riverside", "Pecan Grove", "Texas Star", "Longhorn", "Mesquite", "Caliche", "Prairie Wind",
               "Shoal Creek", "Bull Creek", "Bee Cave", "Red Oak", "Cypress Creek", "Spicewood", "Agave",
               "Sunset Valley", "Granite Peak", "Blackland", "Lantana", "Big Sky", "Armadillo", "Monarch"]

NAME_PATTERNS = {
    "plumbing": ["{last} Plumbing", "{last} & Sons Plumbing", "{place} Plumbing Co.", "{place} Pipe & Drain",
                 "{last} Plumbing Services", "{place} Rooter", "{first}'s Plumbing", "{last} Family Plumbing"],
    "electrical": ["{last} Electric", "{place} Electrical Contractors", "{last} Electric Co.", "{place} Electric",
                   "{first}'s Electrical Services", "{place} Power & Light", "{last} & Daughters Electric"],
    "hvac": ["{place} Air", "{last} Heating & Air", "{place} Comfort Systems", "{last} AC & Heating",
             "{place} Air Conditioning", "Cool {place} HVAC", "{first}'s Heating & Cooling"],
    "roofing": ["{last} Roofing", "{place} Roofing & Exteriors", "{place} Roof Pros", "{last} Roofing Co.",
                "{place} Roofing", "{last} Brothers Roofing", "{place} Storm & Roof"],
    "landscaping": ["{last} Landscaping", "{place} Gardens & Hardscapes", "{place} Lawn & Landscape",
                    "Native {place} Landscapes", "{last} Outdoor Living", "{place} Tree & Landscape",
                    "{first}'s Lawn Care"],
}

PAYMENT_METHODS = ["credit card", "debit card", "check", "cash", "ACH transfer", "Zelle", "Venmo", "financing"]
LANGUAGES = ["English", "Spanish", "Vietnamese", "Mandarin", "Hindi", "Korean", "Arabic"]
