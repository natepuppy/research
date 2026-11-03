"""
Example return value of agents.successor_representation on a 3x3 grid.

Same layout and policy as transfer_examples_3x3.py:
  floor everywhere except green (goal) at (2, 2)
  policy: south while row < 2, else east while col < 2
  gamma = 0.9; goal and lava end the episode (goal row of P is all zeros)

Read successor[i, j] as discounted expected time on cells[j] starting from cells[i].
Use index_of_cell[(r, c)] to get i or j from coordinates.
"""

GAMMA = 0.9

# --- return value: (cells, index_of_cell, successor) ---

cells = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
]

index_of_cell = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 3,
    (1, 1): 4,
    (1, 2): 5,
    (2, 0): 6,
    (2, 1): 7,
    (2, 2): 8,
}

# Each row is a start cell (cells[i]). Each column is a destination cell (cells[j]).
# If you start at cells[i], how much discounted time you expect to spend on cells[j].
# Built from (I - GAMMA * P)^-1; P filled by the same loop as agents.py.
successor = [
    [1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.81, 0.729, 0.6561],
    [0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.81, 0.729],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.81],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.81, 0.729],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.81],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.81],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

# # Optional: one-step matrix P (same shape) before the inverse.
# transition_probabilities = [
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]


def row_as_grid(flat_row):
    """Reshape one SR row into a 3x3 grid aligned with (row, col)."""
    return [flat_row[r * 3 : (r + 1) * 3] for r in range(3)]


def lookup(start, dest):
    i, j = index_of_cell[start], index_of_cell[dest]
    return successor[i][j]


if __name__ == "__main__":
    print("cells (index -> coordinate):")
    for i, cell in enumerate(cells):
        print(f"  {i} -> {cell}")
    print(f"\nsuccessor.shape = ({len(successor)}, {len(successor[0])})")
    print("\nExample: start (0, 0) as a 3x3 grid of discounted visits:")
    i = index_of_cell[(0, 0)]
    for r, row in enumerate(row_as_grid(successor[i])):
        print(f"  row {r}: " + "  ".join(f"{v:.4f}" for v in row))
    print(f"\n  lookup((0,0), (2,2)) = green mass = {lookup((0, 0), (2, 2)):.4f}")
    print(f"  lookup((2,2), (2,2)) = on goal already = {lookup((2, 2), (2, 2)):.4f}")
