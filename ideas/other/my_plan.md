## The Vision
We are measuring if we can transfer experience from one agent (the Demonstrator) to another agent (the Learner) to see if the Learner can avoid obstacles (Lava) it has never encountered before. It will attempt to learn based on the experience from the Demonstrator who encountered lava in a completely different part of the Grid.

## The Plan
We have two Agents. One is a Demonstrator Agent and one is a Learner Agent. We also have two four-room grid environments. The grid consists of walls, lava, and a goal.

The Demonstrator Agent and the Learner Agent train in different grid environments. The only difference is that the Demonstrator Agent's map has lava in it, and the Learner's does not. Therefore, the Demonstrator learns how to navigate the lava, but the Learner has never encountered lava.

Each agent then learns how to navigate their own map. After each agents learns to navigate it's own map, then we introduce lava to the Learner for the first time. This is what the Learners new map looks like:

Then we run the Learner from each starting position (every non-goal and non-lava cell) and with many different lava configuration. Do that four separate times. Each time it is trying to get to the goal under these circumstances:
1. The Learner tries to get to the goal without any information about the lava
2. The Learner tries to get to the goal with the Demonstrator’s raw moves
3. The Learner tries to get to the goal with the Demonstrator’s successor features
4. The Learner tries to get to the goal with the Demonstrator’s options

For successor features, the Learner is told lava's penalty but has no experience near lava.

The exact way we will transfer this knowledge is in a separate section below. 

Then we record the percentage of the time the Learner reached the goal without stepping into lava in all four scenarios. Scenario one is the baseline. Then we graph the results. Something like this:
Y-axis is the percentage of the time the goal was reached without stepping into lava
X-axis is how often the block of tiles in front of the Learner is not one the Demonstrator already stored.

Each lava configuration gives one point on the X-axis. The starting positions are averaged to get that point's height. 

output_graph.png

## How are we going to transfer this knowledge from Demonstrator to Learner?
Since the exact coordinates that the Demonstrator encounters lava are different from the coordinates that the Learner encounters lava, we cannot simply transfer knowledge by coordinates, we must instead encode that knowledge into a key that doesn’t rely on exact coordinates. This is what I was attempting to demonstrate with my grid example code above. 

The way we will transfer knowledge from one to the other will be by creating a hash based on the full view of the Demonstrator. The Demonstrator agent will be able to see a 7x7 grid in front of them, but for the sake of this example, let’s do a 2x2.

Say the Demonstrator is at (13,13) and it sees the following:

<table>
<tr><td>open</td><td>open</td></tr>
<tr><td>open</td><td>lava</td></tr>
</table>

This view is then encoded into a hash value like this:

((open, open),(open, lava)) —> mapped to the successor feature, or the option, or the list of raw moves that the agent took

The experience from the Demonstrator is only used if the outcomes actually agree.
- For raw moves, the visits had to lead to the same next result.
- For successor features, they had to give the same future counts.
- For options, they had to make the same move.

If a view isn't stored, the Learner uses its own policy.

Then when the Learner is at (2,13) and it sees this:

<table>
<tr><td>open</td><td>open</td></tr>
<tr><td>open</td><td>lava</td></tr>
</table>

Then it can look up that view in the Demonstrator's experience and use it to decide what to do.






