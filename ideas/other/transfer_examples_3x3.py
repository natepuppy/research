"""
3x3 examples of successor representation and successor features — same story as
transfer_examples.py, but big enough that a coarse scene key is not enough.

Grid, row then column (all floor except green at bottom-right):

    (0,0) floor    (0,1) floor    (0,2) floor
    (1,0) floor    (1,1) floor    (1,2) floor
    (2,0) floor    (2,1) floor    (2,2) green

Policy (move toward green: south while row < 2, else east while col < 2):
    (0,0) south → (1,0) south → (2,0) east → (2,1) east → (2,2)
    and similarly from every other start.

Reaching green ends the episode. Green is counted once, then stop.
Discount gamma = 0.9.

Successor representation
------------------------
For each start cell, a 3x3 matrix. Entry [r][c] = discounted expected
visits to cell (r,c) if you follow the policy above.

Read one row of the big SR matrix (not shown here) as a flattened 3x3 grid.

Successor features
------------------
Each SR row, aggregated by tile type:
  "empty floor" = sum of entries over the eight floor cells
  "green cell"  = entry for (2,2)

What transfers
--------------
B can store successor_features_by_cell — A cannot use it (different coordinates).

View key = the 3x3 window centered on the agent, row-major, nine tile names.
Off the map counts as "wall". On this grid every cell gets a different view,
so successor_features_by_view has nine keys (no collisions).

Coarse key (tile under me, tile ahead) is too weak here: six cells share
("floor", "floor") but three different feature vectors. That is why the real
experiment uses a 7x7 view, not two tiles.

Layout of view tuples (agent always at center index 4):

    0 1 2
    3 4 5
    6 7 8
"""

GAMMA = 0.9

# --- keyed by B's coordinates ---

# One greedy episode per start cell (shortest path under the policy above).
raw_by_cell = [
    [  # from (0, 0)
        {"at": (0, 0), "move": "south", "landed": (1, 0), "payment": 0},
        {"at": (1, 0), "move": "south", "landed": (2, 0), "payment": 0},
        {"at": (2, 0), "move": "east", "landed": (2, 1), "payment": 0},
        {"at": (2, 1), "move": "east", "landed": (2, 2), "payment": 1},
    ],
    [  # from (0, 1)
        {"at": (0, 1), "move": "south", "landed": (1, 1), "payment": 0},
        {"at": (1, 1), "move": "south", "landed": (2, 1), "payment": 0},
        {"at": (2, 1), "move": "east", "landed": (2, 2), "payment": 1},
    ],
    [  # from (0, 2)
        {"at": (0, 2), "move": "south", "landed": (1, 2), "payment": 0},
        {"at": (1, 2), "move": "south", "landed": (2, 2), "payment": 1},
    ],
    [  # from (1, 0)
        {"at": (1, 0), "move": "south", "landed": (2, 0), "payment": 0},
        {"at": (2, 0), "move": "east", "landed": (2, 1), "payment": 0},
        {"at": (2, 1), "move": "east", "landed": (2, 2), "payment": 1},
    ],
    [  # from (1, 1)
        {"at": (1, 1), "move": "south", "landed": (2, 1), "payment": 0},
        {"at": (2, 1), "move": "east", "landed": (2, 2), "payment": 1},
    ],
    [  # from (1, 2)
        {"at": (1, 2), "move": "south", "landed": (2, 2), "payment": 1},
    ],
    [  # from (2, 0)
        {"at": (2, 0), "move": "east", "landed": (2, 1), "payment": 0},
        {"at": (2, 1), "move": "east", "landed": (2, 2), "payment": 1},
    ],
    [  # from (2, 1)
        {"at": (2, 1), "move": "east", "landed": (2, 2), "payment": 1},
    ],
    [  # from (2, 2)
        {"at": (2, 2), "move": "stay", "landed": (2, 2), "payment": 1},
    ],
]

# weights[start][row][col] = discounted visits to (row, col)
successor_representation = {
    (0, 0): [[1.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.81, 0.729, 0.6561]],
    (0, 1): [[0.0, 1.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.81, 0.729]],
    (0, 2): [[0.0, 0.0, 1.0], [0.0, 0.0, 0.9], [0.0, 0.0, 0.81]],
    (1, 0): [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.81, 0.729]],
    (1, 1): [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.81]],
    (1, 2): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.9]],
    (2, 0): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.9, 0.81]],
    (2, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.9]],
    (2, 2): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
}

successor_features_by_cell = {
    (0, 0): {"empty floor": 3.439, "green cell": 0.6561},
    (0, 1): {"empty floor": 2.71, "green cell": 0.729},
    (0, 2): {"empty floor": 1.9, "green cell": 0.81},
    (1, 0): {"empty floor": 2.71, "green cell": 0.729},
    (1, 1): {"empty floor": 1.9, "green cell": 0.81},
    (1, 2): {"empty floor": 1.0, "green cell": 0.9},
    (2, 0): {"empty floor": 1.9, "green cell": 0.81},
    (2, 1): {"empty floor": 1.0, "green cell": 0.9},
    (2, 2): {"empty floor": 0.0, "green cell": 1.0},
}

# --- keyed by 3x3 view (what A can read after a layout change) ---

successor_features_by_view = {
    ("wall", "wall", "wall", "wall", "floor", "floor", "wall", "floor", "floor"): {
        "empty floor": 3.439,
        "green cell": 0.6561,
    },
    ("wall", "wall", "wall", "floor", "floor", "floor", "floor", "floor", "floor"): {
        "empty floor": 2.71,
        "green cell": 0.729,
    },
    ("wall", "wall", "wall", "floor", "floor", "wall", "floor", "floor", "wall"): {
        "empty floor": 1.9,
        "green cell": 0.81,
    },
    ("wall", "floor", "floor", "wall", "floor", "floor", "wall", "floor", "floor"): {
        "empty floor": 2.71,
        "green cell": 0.729,
    },
    ("floor", "floor", "floor", "floor", "floor", "floor", "floor", "floor", "green"): {
        "empty floor": 1.9,
        "green cell": 0.81,
    },
    ("floor", "floor", "wall", "floor", "floor", "wall", "floor", "green", "wall"): {
        "empty floor": 1.0,
        "green cell": 0.9,
    },
    ("wall", "floor", "floor", "wall", "floor", "floor", "wall", "wall", "wall"): {
        "empty floor": 1.9,
        "green cell": 0.81,
    },
    ("floor", "floor", "floor", "floor", "floor", "green", "wall", "wall", "wall"): {
        "empty floor": 1.0,
        "green cell": 0.9,
    },
    ("floor", "floor", "wall", "floor", "green", "wall", "wall", "wall", "wall"): {
        "empty floor": 0.0,
        "green cell": 1.0,
    },
}

# Coarse scene key from the 2x2 file — shows why two tiles are not enough here.
# ("floor", "floor") would need THREE different vectors (0,0) vs (0,1)/(1,0) vs (0,2)/(1,1)/(2,0).
successor_features_by_scene_coarse = {
    ("floor", "green"): {"empty floor": 1.0, "green cell": 0.9},
    ("green", "green"): {"empty floor": 0.0, "green cell": 1.0},
}

reward_weights = {"empty floor": 0, "green cell": 1}


def value(features):
    return sum(features[name] * reward_weights[name] for name in features)


if __name__ == "__main__":
    print("Successor features by cell (B's coordinates):\n")
    for cell, features in successor_features_by_cell.items():
        print(f"  {cell}  {features}  value={value(features):.3f}")

    print("\nDistinct views:", len(successor_features_by_view))
    print("\nCoarse ('floor','green') matches cells (1,2) and (2,1):")
    print("  ", successor_features_by_scene_coarse[("floor", "green")])
