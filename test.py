from demoparser2 import DemoParser
import json

DEM_PATH = "replays/match730_003792110568626717009_0497164850_408.dem.info"
OUT_PATH = "demo_dump.json"

parser = DemoParser(DEM_PATH)

header = parser.parse_header()
events = parser.list_game_events()

events_data = parser.parse_events(events)  # <-- already a list

data = {
    "header": header,
    "events_supported": events,
    "events": events_data,
}

with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=2, default=str)

print(f"Saved {len(events_data)} events to {OUT_PATH}")
