---
title: "Borrowing Another Agent's Experience"
area: "Fundamental RL"
theme: "Multi-agent & Shared Experience"
collaborators: ["John Martin", "Will Dabney"]
active_contributors: ["@natepuppy"]
lead_interest: "High"
status: "Active"
summary: "Raw transitions, successor features, or options - which survives when two agents encode the world differently? Paper 1: The fundamental comparison."
---

# 👥 Borrowing Another Agent's Experience

> [!NOTE]
> No single agent can experience everything worth learning from. Others are out there having experiences you never will — but their data arrives encoded in their terms, chosen for their goals. What unit of experience actually survives the trip?
> **Area:** Fundamental RL · **Collaborators:** John Martin, Will Dabney
> **Why this fits ADEPT.** Learning from data collected by someone whose representation and goals differ from yours is the deployment-versus-benchmark gap in miniature.

> [!TIP]
> **This stub is the fundamental comparison.** Which unit of experience survives representation mismatch? It runs in purpose-built environments with a dial on mismatch severity and state-space size. Domain-neutral, no data access required, and the contribution is a crossover curve.
> **Clinical application:** [Borrowing Another Agent's Experience: Clinical Application](borrowing-another-agents-experience-paper-2.md). The clinical application waits until this experiemnt has an answer.

## Problem

Any agent is small relative to the world it operates in. It cannot visit every state, cannot represent every value, and will never see most of what happens. That is the **Big World Hypothesis** (Javed & Sutton), and it is a permanent condition rather than a limitation better data collection fixes.

The obvious response is to learn from other agents, who are exploring parts of the world you never will. But their experience does not arrive in a usable form:

- They encode the world differently — the same underlying quantity measured, discretized, or featurized another way.
- They wanted something different, so their choices reflect goals that are not yours.
- Their available actions may differ, so what worked for them may not even be open to you.

Naive pooling therefore teaches the wrong lessons, and the failure is not subtle. The question is what you can share instead.

## Why now

The tools have quietly matured, and — more usefully — there are now **three candidate answers that can be compared against each other** rather than one to advocate for.

**Successor features** (co-authored by Will Dabney, a collaborator here) decompose a value function into "what tends to happen" and "what I want." The **options framework** offers a behavioural unit instead of a representational one. And the two now compose: the **Option Keyboard** builds options in successor-feature space, while **Successor Options** discovers options *from* successor representations. A comparison that would have been apples-to-oranges a few years ago is now a controlled experiment.

## Core hypothesis

One agent can use another agent's experience to avoid obstacles and reach a goal, even when the two never had the same states, never saw the exact same obstacles, and had different goal locations. We can transfer those experiences by what the other agent saw rather than where it was.

The open question underneath is **what unit of experience actually transfers.** Three candidates, not equally plausible:

- **Raw transitions.** The obvious thing to share, and the thing that fails: their states are measured differently and their actions were chosen for different reasons.
- **Successor features.** Share "what tends to happen" and keep your own "what I want." Survives a difference in goals, but still assumes a shared feature space.
- **Options** — temporally extended behaviours with an initiation condition, a policy, and a termination condition. Defined by *what the agent does* rather than by how it encodes the world, so an option can transfer even when the state representations do not line up, provided its initiation and termination conditions can be stated in terms both sides recognize.

**The sharpest version of the claim:** the right unit of transfer is **behavioural rather than representational**, and options are the formalism for that. If true, the problem stops being "align two state spaces" and becomes "identify which of their behaviours are portable."

**Where we start.** The proposed way to hand experience from B to A is to store it by B's view, train a small net so it covers views B never saw, and let A use the output of that net. The first thing we try is the successor representation. If that transfer method helps A reach the goal even as their views differ more and more, then we have success. We then try the other two methods.

> [!WARNING]
> **Everything here exists separately.** Successor features for transfer (2017), machine theory of mind (2018), importance-sampling corrections (V-trace), third-person imitation learning (2017), cross-embodiment data pooling (Open X-Embodiment), and a 2025 extension of successor features to multi-agent transfer.
> **What is not done** is the comparison itself. Nobody has put the three candidate units against each other on a controlled mismatch axis and reported where they cross. That framing is what turns a crowded ingredient list into a specific question, and it is why this stub is worth doing before anything is built.

## Candidate first experiments

> [!IMPORTANT]
> **Compute note.** This paper runs in gridworlds where the ground truth is computable exactly. The environment family *is* the contribution's instrument, so building it carefully is the first real task rather than a preliminary. A laptop suffices.

**Steps for the fundamental comparison experiment:**

**First experiments.** Two agents, a demonstrator (B) and a learner (A). Each one has its own Four Rooms map, so they never see the same cells. The question is: does B's experience help A avoid lava?

1. **One agent, no lava:** One agent (B) in Four Rooms. Learn the successor representation for a multiple different goal positions. Save this by exact cell location.
2. **Add lava:** Add lava and retrain B. Refine the Successor Representation with this new context, and check that the successor representation changes.
3. **Train A:** Train a separate agent (A) in separate four rooms map without lava.
4. **Train the neural net:** Train a small Neural net on B's experience. The input will be B's 7x7 view and the goal, not the exact cell it was on, the output will be the successor feature.
5. Add lava to A's map. And allow A to access B's experience by inputting it's own view into the neural net, and having the neural net respond with the successor feature of the view that was closest to B's. Use this to see if A can avoid the lava before it ever sees it.
6. **Final Graph:** The final output will be as follows: y-axis is the percentage of the time A avoids lava, x-axis is how much A and B's views overlapped.
7. Repeat steps 1-6 with the other forms of knowledge transfer. Nothing, raw moves, and options. Add all of these tot eh line graph. And see what methods continues to help as A's and B's views get more different.
8. **Expand the test:** Add keys and doors instead of just lava. Make the grids bigger. Change the view size from 7x7. Force A and B to have different goals.


1. **One agent, no lava:** One agent (B) in Four Rooms. Learn the successor representation for multiple different goal positions. Save this by exact cell location.
2. **Add lava:** Add lava and retrain B. Check that the successor representation changes, and that a lava cell looks different from a wall cell. Both block the path, but a wall bounces you back and lava ends the episode. If they look the same, A will never be able to tell danger from an obstacle.
3. **Train A:** Train a separate agent (A) in Four Rooms without lava, on its own map.
4. **Train the neural net:** Change what the successor representation counts so it can leave B's map. Instead of counting visits to each cell on the map, count visits to each of the 49 cells in B's 7x7 view, plus a count for each tile type (lava, wall, goal). Then train a small neural net on B's experience. The input is B's 7x7 view and where the goal is relative to B, not the exact cell it was on. The output is this view-based successor representation. This lets A use B's experience on views B never saw.
5. **Let A use B's experience:** Add lava to A's map. Tell A lava's penalty. For each move, A puts the view it would land on into the neural net, gets back the successor representation, and multiplies the lava count by the penalty to score the move. A only does this when a gate says to, and otherwise follows its own policy. Two gates: A has never seen this view before, or B's closest stored view is within k tiles of A's view. Sweep k. Also run "always use B" as a check. Use this to see if A can avoid the lava before it ever sees it.
6. **Final graph:** Y-axis is the percentage of the time A reaches the goal without stepping in lava. X-axis is how different A's view is from B's closest stored view, counted as the number of the 49 tiles that differ.
7. **Repeat with the other methods:** Repeat steps 4-6 with nothing, raw moves, and options. Raw moves and options are stored by view the same way, with the neural net's output swapped for the next move or the option. Add all of these to the line graph and see which method continues to help as A's and B's views get more different.
8. **Expand the test:** Add keys and doors instead of just lava. Make the grids bigger. Change the view size from 7x7. Force A and B to have different goals...

**Previous plan.** Kept here so we can look back at it while we work out the steps above.

1. **Build the environment family, with two dials.** Two agents in a shared world, and knobs on (a) **how badly their state representations disagree** and (b) **state-space size**. The second matters because as the space grows, each agent's coverage thins and the transfer problem sharpens — so results become curves rather than points. Establish first that naive sharing *hurts*; if it does not, there is no problem to solve.
2. **Add successor features.** Decompose the value function and share only the "what tends to happen" part. The cleanest version of the core idea; try it before anything fancier.
3. **Add an agent model.** Learn a small representation of the other agent — what it seems to want, what it can do — and use it to weight their experience. Compare against sharing everything and sharing nothing.
4. **The head-to-head, swept along the mismatch dial.** Same two agents, three things shared: raw transitions, successor features, options. **The hypothesis is that options degrade most gracefully**, and the deliverable is the crossover point where each unit overtakes the next. That curve is the paper.
5. **Extract options rather than assume them.** Use Successor-Options-style discovery to pull candidate behaviours out of agent B's logged trajectories, then test whether those transfer. This matters because in any real setting you get a database, not a documented protocol.

## Necessary background

**Read these first**

- Javed & Sutton, *The Big World Hypothesis and its Ramifications for AI*, RLC 2024 — [OpenReview](https://openreview.net/forum?id=Sv7DazuCn8). Short, and the premise for the whole stub.
- Barreto et al., *Successor Features for Transfer in RL*, NeurIPS 2017 — [arXiv:1606.05312](https://arxiv.org/abs/1606.05312). One candidate transfer unit, and how the successor representation gets combined with a reward in step 5. Co-authored by Will Dabney.
- Sutton, Precup & Singh, *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL*, Artificial Intelligence 1999. The options framework — the other candidate, and the one the central claim backs.
- Machado, Barreto, Precup & Bowling, *Temporal Abstraction in RL with the Successor Representation*, JMLR 2023 — [arXiv:2110.05740](https://arxiv.org/abs/2110.05740). Shows how options can be found from the successor representation. This is the bridge between our first method and our third, and the route for step 6.

**Skim later**

- Barreto et al., *The Option Keyboard: Combining Skills in RL*, NeurIPS 2019 — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf). Builds and combines options in successor-feature space, so a borrowed behaviour can be re-weighted toward your own priorities rather than adopted wholesale.
- Ramesh, Tomar & Ravindran, *Successor Options*, IJCAI 2019 — [arXiv:1905.05731](https://arxiv.org/abs/1905.05731). Discovers options from successor representations — the route to extracting behaviours from logs.
- Rabinowitz et al., *Machine Theory of Mind*, ICML 2018 — [arXiv:1802.07740](https://arxiv.org/abs/1802.07740). Learning a model of another agent by watching it.
- Stadie, Abbeel & Sutskever, *Third-Person Imitation Learning*, ICLR 2017 — [arXiv:1703.01703](https://arxiv.org/abs/1703.01703). Learning from a demonstration filmed from the wrong angle.
- Espeholt et al., *IMPALA*, ICML 2018 — [arXiv:1802.01561](https://arxiv.org/abs/1802.01561). The standard correction for learning from another policy's data.
- *Successor Features for Transfer in Alternating Markov Games*, 2025 — [arXiv:2507.22278](https://arxiv.org/abs/2507.22278). The closest recent extension; read it to see what is left.

**Open questions**

- Which unit transfers best under mismatch, and is there a clean crossover where one overtakes another? This is the whole project in one sentence.
- Is it fair to tell A the lava's penalty for the successor-feature condition, when the other conditions don't get that hint?
- An option's initiation and termination conditions are themselves stated in terms of state. Does that quietly reintroduce the alignment problem options were meant to sidestep, or are those conditions coarse enough to survive it?
- Should the model of the other agent capture their reward function, their policy, or their state representation? Very different costs.
- Is there a bound on how much cross-agent transfer helps before mismatched capabilities make it actively harmful?
- If you discover behaviours from logs rather than being handed a protocol, are you recovering actual practice or an artifact of how that site records things?
- Under the Big World Hypothesis an agent must keep adapting forever. Does leaning on others' experience help, or make you slower to notice your own world has changed?



---

*Part of the [Research Stubs](../README.md) catalog.*