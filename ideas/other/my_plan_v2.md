## The question
Can one agent's experience help another agent handle something it has never seen?

## Setup
Two agents, two separate four-room maps. They never visit the same cells.

- **Demonstrator (B):** trains on a map with lava. Learns to reach the goal and avoid lava.
- **Learner (A):** trains on a map with no lava. Learns to reach the goal.

Each agent's state is what it sees (its 7x7 view), not just its position.

## Test
Add lava to the Learner's map. Run the Learner from every start cell (not goal or lava) under four conditions:

1. Nothing from the Demonstrator (baseline)
2. The Demonstrator's raw moves
3. The Demonstrator's successor features
4. The Demonstrator's options

For successor features, the Learner is told lava's penalty. It still has no experience near lava.

Score: percentage of runs that reach the goal without stepping in lava.

Lava is new to the Learner in all four conditions, so any difference between them comes from the Demonstrator's experience, not from lava being new.

## Graph
- **Y-axis:** percentage of runs that reach the goal without stepping in lava.
- **X-axis:** how different the Learner's lava situations are from the Demonstrator's. Measured as the average number of tiles that differ between the Learner's view and the closest view the Demonstrator stored. 0 is an exact match.

Each lava configuration on the Learner's map is one point on the X-axis. Start with configurations that look like the Demonstrator's, then move to ones that don't. Four lines, one per condition.

Expected result: raw moves help when situations match and hurt when they don't. Options hold up longest. The paper is where the lines cross.

## How the Demonstrator's experience is stored
By what the Demonstrator saw, not where it was. Example with a 2x2 view (the real view is 7x7):

<table>
<tr><td>open</td><td>open</td></tr>
<tr><td>open</td><td>lava</td></tr>
</table>

Every visit is stored as a pair: this view, and what happened (the raw move taken, the successor features, or the option). All three methods share the same key. Only the stored value differs.

## How the Learner uses it
The Learner rarely sees the exact view the Demonstrator stored. Say it sees:

<table>
<tr><td>wall</td><td>open</td></tr>
<tr><td>open</td><td>lava</td></tr>
</table>

So we train one small neural net per method on the Demonstrator's stored pairs. Input: a view. Output: that method's answer (the next move, the future counts, or the option). The net learns which tiles matter, so it works on views the Demonstrator never saw. When visits to the same view disagree, the net averages them.

## Order of work
1. One agent in Four Rooms. Learn the successor representation for a few goal positions.
2. Add lava (cells where the agent fails and gets a negative reward). Check that the successor representation changes.
3. Train the Demonstrator with lava. Build the store.
4. Train the Learner without lava.
5. Train the three nets on the store.
6. Add lava to the Learner's map. Run the four conditions. Make the graph.
7. Keys and doors: Demonstrator trains with them, Learner without. Can the Learner pick up a key and open a door? (The Learner must already have the pick-up and open actions.)

## Later
- Demonstrator sees differently than the Learner (smaller window, merged tile types).
- Bigger maps.
- Different goals for the two agents.
- A model of the Demonstrator (what it wants, what moves it has) to weight its experience. Only matters once goals or moves differ.
- Extract options from the Demonstrator's log automatically.

## Questions for Killian
1. Is it fair to tell the Learner lava's penalty for successor features?
Laptop only, MiniGrid, no GPU.
