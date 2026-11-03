"""
2x2 examples of what agent B stores, and the same numbers re-keyed so agent A
can look them up.

Grid, row then column:

    (0,0) floor    (0,1) floor
    (1,0) floor    (1,1) green

Policy: from (0,0) step east, from (0,1) step south, from (1,0) step east.
Reaching green ends the episode. Green is counted once.
Discount gamma = 0.9, so one step away counts as 0.9 and two steps away as 0.81.

What changed from the drafted examples
--------------------------------------
1. Raw log: one episode per start cell. (0,0) has two rows because that
   walk takes two steps. (0,1), (1,0), and (1,1) have one row each.
   The step out of (0,1) appears twice because two episodes walk it:
   the one that starts at (0,0), and the one that starts at (0,1).
2. Raw log: dropped "facing". The matrices below have one entry per cell,
   and the turn actions that "facing" + "step forward" would need were not
   in the log. The move is the direction itself.
3. Successor representation: numbers kept. Added the grid layout, gamma, and
   the "count green once, then stop" rule so 0.81 has a definition.
4. Successor features: numbers kept. They are each successor-representation
   row with the three floor cells added together.
5. Coordinate keys do not transfer. A never builds the key (0,0). The
   scene-keyed block at the bottom is the version A can read. (0,1) and
   (1,0) collapse to one key because both are floor with green ahead.
"""

GAMMA = 0.9

# --- keyed by B's coordinates. Correct worksheet. A cannot look these up. ---

# One episode per start cell. (0,0) takes two steps. The others take one.
# Starting on green is already done: stay there and collect the payment.
raw_by_cell = [
    [
        {"at": (0, 0), "move": "east", "landed": (0, 1), "payment": 0},
        {"at": (0, 1), "move": "south", "landed": (1, 1), "payment": 1},
    ],
    [
        {"at": (0, 1), "move": "south", "landed": (1, 1), "payment": 1},
    ],
    [
        {"at": (1, 0), "move": "east", "landed": (1, 1), "payment": 1},
    ],
    [
        {"at": (1, 1), "move": "stay", "landed": (1, 1), "payment": 1},
    ],
]

# weights[start] is [[(0,0), (0,1)],
#                    [(1,0), (1,1)]]
successor_representation = {
    (0, 0): [[1.0, 0.9],
             [0.0, 0.81]],
    (0, 1): [[0.0, 1.0],
             [0.0, 0.9]],
    (1, 0): [[0.0, 0.0],
             [1.0, 0.9]],
    (1, 1): [[0.0, 0.0],
             [0.0, 1.0]],
}

successor_features_by_cell = {
    (0, 0): {"empty floor": 1.9, "green cell": 0.81},
    (0, 1): {"empty floor": 1.0, "green cell": 0.9},
    (1, 0): {"empty floor": 1.0, "green cell": 0.9},
    (1, 1): {"empty floor": 0.0, "green cell": 1.0},
}

# Counted from raw_by_cell. One move per cell. Stop on green.
option_by_cell = {
    "start_when": {(0, 0), (0, 1), (1, 0)},
    "policy": {
        (0, 0): "east",
        (0, 1): "south",
        (1, 0): "east",
        (1, 1): "stay",
    },
    "stop_when": {(1, 1)},
}

# --- keyed by the scene. This is the object that transfers. ---
#
# key = (tile under me, tile ahead)
# (0,1) facing south and (1,0) facing east are both ("floor", "green").

raw_by_scene = [
    [
        {"under": "floor", "ahead": "floor", "move": "forward", "payment": 0},
        {"under": "floor", "ahead": "green", "move": "forward", "payment": 1},
    ],
    [
        {"under": "floor", "ahead": "green", "move": "forward", "payment": 1},
    ],
    [
        {"under": "floor", "ahead": "green", "move": "forward", "payment": 1},
    ],
    [
        {"under": "green", "ahead": "green", "move": "stay", "payment": 1},
    ],
]

successor_features_by_scene = {
    ("floor", "floor"): {"empty floor": 1.9, "green cell": 0.81},
    ("floor", "green"): {"empty floor": 1.0, "green cell": 0.9},
    ("green", "green"): {"empty floor": 0.0, "green cell": 1.0},
}

# Same option, keyed by (under, ahead). (0,1) and (1,0) share one move.
option_by_scene = {
    "start_when": {("floor", "floor"), ("floor", "green")},
    "policy": {
        ("floor", "floor"): "forward",
        ("floor", "green"): "forward",
        ("green", "green"): "stay",
    },
    "stop_when": {("green", "green")},
}

reward_weights = {"empty floor": 0, "green cell": 1}


def value(features):
    return sum(features[name] * reward_weights[name] for name in features)


if __name__ == "__main__":
    for scene, features in successor_features_by_scene.items():
        print(scene, features, "value", round(value(features), 2))
