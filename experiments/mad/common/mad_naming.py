import re

ACTIVITY_NAMES = {1:"Walk",2:"Stand Up",3:"Sit Down",4:"Up Stairs",5:"Down Stairs",
                  6:"Pick an Object",7:"Step Over",8:"Semi Turn Around",9:"Fall"}
SENSOR_CODES   = {1:"eb_cam",4:"mocap",5:"radar"}
REVERSED_PAIRS = [(2,3),(4,5)]  # (Stand Up<->Sit Down), (Up<->Down Stairs)

# format reel : A{Y1(2)}{Y2(1)}P{XXX}R{Z}S{U}D{V}
PAT = re.compile(r"A(?P<act>\d{2})(?P<sub>\d)P(?P<part>\d{3})R(?P<rep>\d)S(?P<sensor>\d)D(?P<ant>\d)")

def parse(name):
    m = PAT.search(str(name))
    if not m:
        return None
    d = m.groupdict()
    return {"activity": int(d["act"]), "sub": int(d["sub"]),
            "participant": int(d["part"]), "rep": int(d["rep"]),
            "sensor": int(d["sensor"]), "antenna": int(d["ant"]),
            "activity_name": ACTIVITY_NAMES.get(int(d["act"]), "?")}
