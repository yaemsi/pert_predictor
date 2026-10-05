# Perturbation Prediction with a Small Language Model

Use real L1000 data and a small language model to build the most convincing perturbation-prediction system you can. The consequential modeling choices are yours.

- **Role:** Founding Engineer work sample
- **Timebox:** 7 calendar days; designed for about 1 focused day of work
- **Hardware:** CPU in this workspace; GPU through Slurm
- **Tools:** use whatever local tools help you, including AI coding tools
- **Primary signal:** decision quality, experimental taste, engineering, communication, and ownership

This file is the only assignment. Supporting notes live next to it. `data/` is the raw LINCS L1000 Phase II release.

## Context

At CellType we train language models on biological data — especially gene expression — to predict how cells respond to perturbations such as drugs or genetic interventions. This exercise is close to the day-to-day work of the role.

The assignment is open-ended. We care more about how you frame the problem, make tradeoffs, and communicate results than about hitting a specific leaderboard number.

## Task

Using the supplied LINCS L1000 Phase II data and Qwen/Qwen2.5-0.5B as your starting point, build an approach for perturbation prediction with an LLM or LLM-style sequence model.

In plain terms:

> Given information about a cell context and a perturbation, predict the transcriptional response.

There is no single correct architecture. A clear, well-reasoned small experiment beats an opaque large one.

## What is fixed

1. **Use the supplied LINCS L1000 Phase II files as the source of truth.** Do not use other expression data, hosted prediction services, or public target lookups for evaluation rows.
2. **Use Qwen/Qwen2.5-0.5B meaningfully.** The model should be a real part of the system, not a decorative mention. Fetch it from Hugging Face and record the revision you used. Suggested revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.
3. **Work in this environment.** CPU work happens here. GPU work goes through Slurm.
4. **Return a coherent submission.** We need enough code, evidence, and writeup to understand and rerun your main result.

Everything else is open. There is no required output schema and no prepared train/val split.

## What is open

You decide, and we expect different people to decide differently:

- what the model actually predicts, and at what granularity;
- how a condition and a response are represented and serialized;
- the training objective, or whether you train the base weights at all;
- how you adapt or use the model, and how you spend your compute;
- decoding, stopping, and how you handle malformed or missing output;
- which baselines and controls are worth running;
- what you measure, on what, and how you decide it means something;
- what you do when the result is disappointing.

There is no single expected answer to any of these. A simple, well-justified approach with honest evidence beats an elaborate one you cannot defend.

## What you receive

```text
assignment.md          this file; the only assignment
DOMAIN_PRIMER.md       what L1000 values mean
DATASHEET.md           provenance, files, and limits
COMPUTE_POLICY.md      GPU / Slurm notes
data/                  raw GEO GSE70138 LINCS L1000 Phase II
```

`data/README.md` lists the matrices and metadata. Level 3, Level 5, and GEO tables are all there.

## Compute

```bash
gpu submit -- <command>
gpu status
gpu logs <job-id>
gpu cancel <job-id>
gpu shell <job-id>
```

`sbatch` also works. Each job gets one-eighth of the node: 1 A100, 12 CPUs, and 158 GiB RAM, for up to 12 hours. You may have one running job and up to three submitted jobs. Logs land in `~/slurm-logs`. Scratch under `/scratch/slurm` is ephemeral.

## Suggested questions

- What does “prediction success” mean here?
- Is correlation the right metric, or is it misleading?
- Would a simple baseline embarrass your approach?
- Is the model using biology, chemistry, or just metadata shortcuts?
- What failure mode did you find that changed your mind?

## What to submit

When you are ready, run:

```bash
submit
```

That packages your workspace (code, writeup, notes — not the read-only `data/`) and sends it to us. You will get a SHA-256 receipt.

Your submission should contain:

1. **Short writeup** — problem framing, metric choice, approach, results, failure modes, and what you would do next.
2. **Runnable code** — clear README with environment and reproduction steps.
3. **Artifacts** — key metrics on your held-out split, plus at least one strong baseline.
4. **Decision record** — three to seven decisions that shaped the system, alternatives considered, why you chose what you chose, and what result would tell you the choice was wrong.
5. **AI disclosure** — where AI helped, how you verified it, and what you rejected.

Optional but valued: data pitfalls, an ablation or negative result, and plots, tables, logs, or diagnostics that support your claims.

## Working style

Leave enough trace that another engineer can follow how you got to the final result.

A good submission usually includes some lightweight process evidence, such as:

- frequent commits in git;
- dated notes in `DECISIONS.md`;
- notebooks, logs, plots, or intermediate outputs;
- a short “what I tried and what changed” section in the writeup.

If you use git, commit as you normally would when exploring a problem.

## How we evaluate

We will look for problem framing, experimental taste, engineering quality, communication, and ownership.

Correctness of the system you built, and the honesty of the evidence behind it, matter far more than predictive lift. A well-built system that fails to beat a simple baseline, with a clear account of why, does well. A system with attractive numbers that we cannot verify, or that turns out to be measuring something other than what it claims, does not.

If you run out of time, stop and submit the most trustworthy state you have.

## Using AI

Use it. We do. You own every line you submit: if you cannot explain a decision, justify it against the alternative you did not take, and modify the code live, it will not help you.

In the follow-up conversation we will pick a few of your decisions and ask why you chose that path over the others, and we will ask you to change something in your code while we watch.
