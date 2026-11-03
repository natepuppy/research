I said:

# [borrowing-another-agents-experience] I'd like to take Paper 1

## Which stub

[Borrowing Another Agent's Experience](https://github.com/byu-adept-lab/research-stubs/blob/main/fundamental-rl/borrowing-another-agents-experience.md)

I want to work on this if that is possible. I'd stick to Paper 1 (the gridworld comparison) and leave the hospital / eICU part until I actually have promising data from part 1.

## My questions

1. The stub is incubating because we haven't decided whether to do Paper 1 alone. If I commit to Paper 1 only, is that enough to claim it, or is something else still unfinished?
2. Here's how I'm reading the setup, please tell me if I have it backwards. A is the one we train and score. B already has a log from part of the maze A hasn't seen much. We pass information from B to A, not the other way around. Same goal at first. They walk the same grid, but B's "I am here" labels are blurry (a 4×4 maze where B only sees 2×2 blobs, so four of A's squares look the same to B). We judge A against the optimal value, not against B.
3. I think the first experiment is just: does dumping B's raw events into A help or hurt as B gets blurrier? If they're labeled the same, it should help. If B is very blurry, it should do worse than giving A nothing. If it never hurts, I don't think there's a project. Is that the right first plot?

## What I would try first

My plan is to build a small gridworld using Gymnasium or maybe I will just build it from scratch. Then I will create two agents A and B. B will have a blur, and I will change that blur as my independent variable. Then I will transfer knowledge from B to A using the three methods mentioned and a baseline (Raw transitions, successor features, options and no knowledge transfer). Then I will test how closely A can get to the perfect strategy. The final output of this experiment will be a line graph. The x-axis will be the amount of blur in B, the y-axis will be how close A was to the perfect strategy, and the four lines will be the 3 knowledge transfer methods and a baseline: Raw transitions, successor features, options and no knowledge transfer. Then we can see which method still helps as B gets blurrier. I will first do the baseline and raw transfer methods, then continue to the other two methods.

This can all be done in a simple environment and no GPU needed. 

## What worries me about it

I haven't implemented successor features or options before. I'd get the raw-pooling plot working first so I know the maze and the blur knob work before I take on the harder methods.






Then he said:


Hey @natepuppy, apologies for my delay.

I'd love to have you work on this project and help shape it out further.

Re: 1, I think that the main mechanism that hasn't been fully thought out yet is how we transfer "Agent B"'s experience to our primary agent. We can transfer raw transitions, successor representations, etc. I think finding the unification of these ideas/concepts is going to be the biggest lift.

Re: 2, I'm thinking that we have a very large grid world where there's likely never going to be overlap in the Agent's experiences. Say there are local features in Agent B's experience (say a pit, cliff, lava, etc.) that they learn to avoid but Agent A hasn't seen anything like that yet. Then when Agent A encounters something along these lines, can it avoid it because it's learned from Agent B's experience?

Re: 3, I think that you've got a solid idea here. I think that the biggest thing is figuring out how to best categorize and store the experience. I'm not sure that raw state information is going to get the job done because it might not adequately describe the experiences of the other agent. This is why successor representations or something else that encodes behaviors + outcomes/rewards is what I think may be most useful.

Re: starting with successor features and/or options. I'd start with a single agent in a "4 room" environment (for example: https://minigrid.farama.org/environments/minigrid/FourRoomsEnv/) where you can set the goal in different places in one of the four rooms. Then you can start learning the successor representation for each point in the grid as a function of where the goal state is.

After you get that configured, I'd recommend you placing "danger zones" (where the agent fails and gets negative reward) in different places and see how that changes the successor representations.



Then I said:


Here is my updated plan. Let me know if it seems good to you.

## Plan

Below is a simplified example of how I envision we will do this. It seems long because I write out everything, but there isn't too much to it. In this example, there is one goal, in the bottom-right corner. Then we train B from all 25 starting positions in the 5x5 grid. There will be one episode from every start cell. The agent can see a 2x2 block.

The top half of the examples below are all keyed by coordinates (and therefore specific to B). But the bottom half is all keyed by the values of the 2x2 block that the agent can see. This is what we will actually give to A because it is generalized and no longer specific to B's exact coordinates. 

In summary, visits that agree on the next payment become a raw step. Visits that agree on the future counts become successor features. Visits that agree only on the move become an option. A window that still disagrees is not given to A.

The plan is still the one you suggested. One agent in Four Rooms, successor representation at each cell for a few goal locations, then danger zones, and only then a second agent. For that second step I want to use two grids drawn the same way, rather than one large grid. B trains on a grid that has lava. A trains on a different draw that does not. A is then tested on a new draw that has lava somewhere else. Same tiles, same actions, same window. The layouts differ, so a coordinate from B's grid is never a cell on A's grid. The score is whether A steps on the new lava.

Laptop only, MiniGrid, no GPU. Does the attached example match the store you had in mind?

## My questions

1. Is this correct? A window only gets one successor-feature row when every visit that sees it has the same counts ex: successor_features_by_scene. However, if the counts disagree between runs, we will not use the successor features ex: unresolved_windows.

2. Do I ever want to combine the types of transfers to see if I can get even better results? Or keep them seperate?



Then I pasted the 5x5 transfer examples:
/Users/nathanclark/personal_dev/sales_conversation_ai/transfer_examples_5x5.py





Then he said:

Can you help me understand your vision/plan a little better? There's a bit of non-standard language (not a knock on having Claude, or some other model help formalize things, this is always a good first step) in there that obfuscates some of the intent.

My main concern with the 5x5 grid is that it's not outside the realm of possibility of any single agent happening to encounter the same states as the "demonstrator" (what you seem to have written out as agent B).

The nice part of the four room environment is that you can isolate one (or more) areas from one agent where other agents do get the chance to visit those areas.





Then I said:

After reviewing our conversation, I see where my earlier note was unclear. Here is a more concrete plan to give feedback on.

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



Then he said:

Hey @natepuppy, I should have been more clear about LLM outputs and usage. I am glad that you're eager to get started on this project and using an LLM to help dive into things is a great idea.

I am however unable to devote much time to parse through a copy and pasted output from an LLM. To be honest, I'd strongly prefer a tl;dr summary in your own words and with your own insights. Ultimately these research problems are designed for my students and collaborators to deeply engage with and develop their own intuition and expertise. I do care about completion but it's not the end all - be all for me.

Without your time spent developing a deeper understanding of the problem and formulating your own plan, I can't commit much time myself to support you.




Then I said:

I give you my word I did not use an LLM to write my latest response. I did take the comment about not filtering my responses through Claude to heart. I had it generate the images I used and look for grammatical errors is all.

My last response was just me trying to fully write out my entire thought process in a formal and extensive research plan. I have been researching a bunch, and also trying to implement the feedback in the comments you gave me, so the plan has changed, and I thought this was more aligned with what you envisioned.

I REALLY don't want you to feel like I am a time suck. That is not fun for anyone, especially you. If I am taking up too much time, let me know, and promise I will stop bothering you.

If you are willing to give me one more chance, would you want me to make the plan more brief and concise? Is this plan even on the right track at all?

Please let me know what else I am doing incorrectly. I want to be an asset, not an annoyance. If it is better, I could start out assisting one of you PHD students and get more mentorship from them before I end up taking on my own project.



Then He said:

I think for these issues, summaries are a better way to dig down to core questions and issues.

I think that the plan / design for the four room environment looks good. I think that my primary concern still remains that there will be overlap in the observations between Agents A and B, but where there may be contextual differences that arise (e.g. the lava). So there might be a confounding issue where we unecessarily complicate agent A's experience by introducing a new mechanic. But this also would be a nice way to probe whether or not agent B's experience with lava elsewhere can be transferred. Here, it is likely that you'll need some additional information encoded in the state (other than just position).

My ideal setting is that we have completely disjoint experience streams between the agents (totally different states observed) but where Agent A is able to distill/incorporate Agent B's experience when it encounters something it has never seen before.



Then I said:

To Clarify: 

1. My current plan assumes that A can use B's experience ONLY if it sees the exact same view (cell configuration) that B saw. But you want A to be able to still use B's experience even if it encounters a view B has never seen before. Is that correct? If so, then I would say, instead of relying on exact views to transfer over, what if we did one of the following:
- Trained a very small neural net on B's experience for each of the 3 transfer methods? 
- Find a way to map A's view to the closest view that B saw. (Like what view of B has the most overlapping cells with the view of A)
That way, we can "distill/incorporate" the information and make it so that info can be used even when the exact view is different.

2. My plan also allows A and B to explore the same cells. What if I walled off the four room grid to two isolated sections so A can ONLY travel in rooms 1 and 2 and B can ONLY travel in rooms 3 and 4? Then A and B never visit the same cells.




Then he said:

Yeah, re:1, I do want A to be able to incorporate the data from B no matter how different it is. There are likely representations from B's experience that can help A accomplish things in a more efficient manner. This could be as simple as A learning in a grid world environment that has zero keys/doors but B is trained in such an environment. Then, after incorporating B's experience, can A pick up keys and open doors?

Another alternative is A operating in a clean grid world without any obstacles but B has to deal with lava and learns to avoid it. What then happens if a pit of lava appears in A's environment after it gets to incorporate B's experience. We can test with/without the experience of B being made available to A.

The protocol you put together for extending/training B's experience for the three transfer methods seems like a good start.

Re 2: This could work. I'm not sure how it'll work but I think it's worth examining.






Then I submitted a PR to claim the project because he asked me to







Then He said:

I think that the first major effort for this project, that we can work through as we move through this PR is getting the major hypotheses and initial experiments nailed down a bit more.

Part of the reason that this paper was left "incubating" is that I hadn't fully considered what the best mechanism for transferring an agent's experience to another would entail. Having that initial mechanism figured out gives us a set of first experiments to try. For example, can we try a successor representation first thing? This would give some structure on the papers to read, concepts to study + implement, etc.

Additionally, I think it may make sense to split out the plans/hypotheses for papers 1, 2, ... into separate "stubs" so there's a way to isolate the experimental design and prevent scope creep.

Can you work to refine the underlying research hypotheses, directions and core focus areas to focus on at the outset of this project?

Then he said:

I think that it would be helpful to keep the previous plan in place, perhaps in a difference section so that it is able to be referenced as we continue to flesh out and get this project fully underway.





