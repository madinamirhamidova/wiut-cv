"""Registry: class id -> rule module. Classes without a rule are simply never predicted."""
from . import (accident, congestion, failure_to_yield, illegal_u_turn, jaywalking, red_light,
               solid_line_crossing, stop_line, stopped_vehicle, wrong_way)

RULES = {m.LABEL: m for m in (
    accident, congestion, failure_to_yield, illegal_u_turn, jaywalking, red_light,
    solid_line_crossing, stop_line, stopped_vehicle, wrong_way,
)}
# Not implemented yet (never predicted): near_miss, illegal_turn, road_obstacle, fire_smoke
