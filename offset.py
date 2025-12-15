

from demoparser2 import DemoParser

parser = DemoParser("match730_003791801599416861069_0133264804_389.dem")

round_starts = parser.parse_events(["round_freeze_end"])

print(type(round_starts))
print(round_starts[:5])

