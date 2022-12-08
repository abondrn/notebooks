import sys

def matching_pins(pins1, pins2, num_pins, offset, prune_match):
    count = 0
    total = len(pins1)
    for p in pins1:
        if ((p+offset-1) % num_pins)+1 in pins2:
            count += 1
        total -= 1
        if count+total <= prune_match:
            return prune_match
    return count

num_pins = int(sys.stdin.readline())
plug_pins = [int(x) for x in sys.stdin.readline().split()]
socket_pins = [num_pins-int(x)+1 for x in sys.stdin.readline().split()]

if len(plug_pins) > len(socket_pins):
    pins1 = socket_pins
    pins2 = set(plug_pins)
else:
    pins1 = plug_pins
    pins2 = set(socket_pins)
    
max_match = min(len(plug_pins), len(plug_pins))
best_match = 0
for o in range(num_pins):
    best_match = max(best_match, matching_pins(plug_pins, socket_pins, num_pins, o, best_match))
    if best_match == max_match:
        break
print(best_match)