# Activity 07 — Your First RL Agent

> **🛠️ Stack for this lesson** — Python / Jupyter notebook · runs in Google Colab.
> 📥 Template: [/learn/ai25/template/activity-07-your-first-rl-agent](/learn/ai25/template/activity-07-your-first-rl-agent)

Reinforcement learning starts with the simplest possible loop: state in, action out, reward back. The notebook ships with `CartPole-v1` configured, a random-action baseline that runs 100 episodes, and visualisations of episode reward over time. Your job is to make the agent *do* something — pick a heuristic, then prove it beats random.

**Time:** ~75 minutes · **Concept:** Concept 07 — Introduction to Reinforcement Learning

---

## What You'll Build

| # | TODO | Cell |
|---|------|------|
| 1 | Implement an epsilon-greedy policy that mixes random action with a pole-angle heuristic | TODO 1 |
| 2 | Add an action-frequency bar chart and report the left/right balance | TODO 2 |
| 3 | Race three policies on the same env: pure random, epsilon-greedy (ε=0.3), pure heuristic | TODO 3 |
| 4 | Sweep epsilon ∈ [0.0, 0.1, …, 1.0] over 50 episodes each; plot reward vs ε | TODO 4 |

The agent class signature is provided. The body is yours to write.

## Run It

1. Open `activity-07-your-first-rl-agent.ipynb` in **[Google Colab](https://colab.research.google.com/)**.
2. `Runtime → Run all`.

First run takes ~45 seconds. CPU is fine.

## Verify

Your activity is done when:

- [ ] TODO 1: epsilon-greedy agent achieves **average reward > 30** (random baseline ~22). Epsilon is a constructor argument.
- [ ] TODO 2: bar chart of left/right counts rendered; balance ratio (`min/max`) printed.
- [ ] TODO 3: a comparison table or chart shows mean reward and success rate for all three policies; pure heuristic should win.
- [ ] TODO 4: line plot of reward vs epsilon for at least 11 epsilon values; the chosen optimum is highlighted.

## Investigation

> **📓 Where this lives now:** open your lesson note for this activity.
> Click **Take Notes** at the top of the lesson page if you don't already have a note —
> it creates one titled `<Lesson title> — <COURSE>` with a scaffold pre-filled from
> the lesson headings. Add an H2 section titled **`Investigation`** to that note and put
> your work there. The note persists across devices and is queryable by the AI Tutor.
>
> 💡 **Tip:** The Tutor will not write findings for you, and that is deliberate. With a draft in place, open the AI Tutor while viewing your note and ask
> *"Read my Investigation notes about the agent behaviour and ask me what evidence I am missing"*. Revise the note with your own answers.

**Prompts to answer in your note:**

The CartPole reward function is a classic — `+1 per timestep` until the pole falls. But the agent's behaviour can hide things.

- After your TODO 1 run, plot the action distribution. Is the agent biased toward one action? If your heuristic uses `observation[2]` (pole angle), what does the sign convention tell you about the bias?
- After TODO 4, what's the optimal ε? Is it 0.0 (pure heuristic)? If yes, what does that say about the heuristic's quality? If not, *why* does some randomness help?
- Read the env source (`gym/envs/classic_control/cartpole.py`) and find the termination condition. Is your heuristic actually responding to it?

Aim for 3–4 paragraphs of findings in your Investigation section.

## Stretch

Pick one:
- Implement epsilon decay: start at 0.9, anneal to 0.05 over 100 episodes. Does it beat your fixed-ε best?
- Use both pole angle *and* cart velocity in the heuristic. Does the optimal ε change?
- Test the same agent on `MountainCar-v0` and `Acrobot-v1`. Does the heuristic transfer?

## 🪞 Reflect on Your Work

Answer in 2-3 sentences each, in your Notes doc. Your tutor reads these as part of grading.

1. **What did you learn that you didn't know before?** Name the most surprising thing — a bug you hit, a syntax quirk, a way the simulator and real device differ.
2. **How did you collaborate with AI?** If you used Claude / ChatGPT / Cursor / Copilot, what part of the work did *you* contribute — the prompt, the verification, the design decision, the bug-fix? If you didn't use AI, what was the hardest thing to figure out alone?
3. **How do you know your code works?** Describe one specific thing you did to confirm — a test you ran, a screenshot you took, a behavior you watched on the device.

> AI is a great collaborator. Owning your thinking, verifying the output, and explaining your design choices is what *learning* looks like in this course.

## Submit

When the Verify checklist is green, head to **[/learn/ai25/certification](/learn/ai25/certification)** and submit your notebook link. Your Investigation section will be graded from your Knowledge Notebook.

<!-- claude-template-fix: readme-v3 -->

<!-- claude-template-fix: notes-migration-v1 -->
