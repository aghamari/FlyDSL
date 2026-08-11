---
name: git-worktree-workflow
description: Use isolated git worktrees in this multi-repo GPU-kernel workspace to run parallel experiments without colliding source trees. Use when asked to "work in a worktree", run several kernel-optimization attempts in parallel, isolate a risky change, or fan out subagents that each edit the same kernel file. Covers the EnterWorktree/ExitWorktree tools, when isolation is actually worth its cost, and the fan-out-then-fold pattern for parallel lever experiments.
argument-hint: [a branch/experiment name, or nothing]
---

# Git Worktree Workflow (multi-repo kernel workspace)

How to use git worktrees in this environment so parallel experiments don't clobber
each other's source. The workspace holds several repos (`FlyDSL`, `aiter`,
`rocm-libraries`, `PyISA`, `mlse-tools-internal`, `claude-knowledge-base`); kernel
optimization often means trying several mutually-exclusive edits to the **same**
kernel file, which is exactly what worktrees isolate.

## When to use a worktree (and when NOT to)
**Use it when:**
- The user explicitly says "worktree" (only create one when asked, or when project
  instructions direct it — see the EnterWorktree tool's own guidance).
- You're fanning out **parallel** attempts that each edit the same source file (e.g.
  4 different BLOCK_M tile choices, or XCD-remap vs wide-store vs B-stationary). Two
  agents editing the same file in one tree contaminate each other's measurements.
- A risky change you want to throw away cleanly without touching the main checkout.

**Do NOT use it when:**
- The work is sequential (one lever at a time on one tree) — just use a branch.
- The change is trivial / reversible in place.
- You only need a different branch — use `git checkout`/`switch`, not a worktree.

Worktrees cost ~200-500ms setup + disk per tree. The shared GPU is fine
(`do_bench` serializes on the device); it's the **source trees** that must be
separate. Don't pay the cost for sequential work.

## The tools (preferred over raw git in this harness)
- **EnterWorktree** — creates a worktree under `.claude/worktrees/` on a new branch
  and switches the session into it. `name` for a new one; `path` to enter an existing
  registered worktree. Base ref is governed by the `worktree.baseRef` setting
  (`fresh` = branch from origin/<default>; `head` = branch from current HEAD).
  ONLY call when the user asked for a worktree.
- **ExitWorktree** — leave it. `action: "keep"` preserves the dir+branch on disk;
  `action: "remove"` deletes both (refuses if there are uncommitted changes /
  unmerged commits unless `discard_changes: true`). No-op if no worktree session is
  active.
- Raw git also works: `git worktree add .claude/worktrees/<name> -b <branch>` and
  `git worktree list` / `git worktree remove <path>`. `git worktree list` is the
  source of truth for what's registered.

## Fan-out then fold (the parallel-experiment pattern)
The proven pattern for parallel kernel levers (mirrors the jdbba-autoresearch skill):
1. Pick up to **4 mutually-exclusive attempts** — different levers, or different grid
   points of one lever that need source edits. Mutual exclusivity is mandatory: no two
   attempts edit the same file in the same tree.
2. Spawn **one subagent per attempt** with `isolation: "worktree"` (Agent tool, all in
   ONE message so they run concurrently). Each gets its own worktree so concurrent
   edits don't collide.
3. Each subagent runs a full mini-loop on its attempt: implement -> correctness-gate ->
   measure -> confirm structural change in ISA/PMC -> return a **structured verdict**
   (`{lever, files, per-shape numbers, keep/discard, reason}`), discards included so the
   loop never re-tries a dead lever.
4. **Fold:** keep only attempts that passed correctness AND beat baseline; partition by
   compatibility (levers touching disjoint mechanisms compose; two that both grow LDS
   can blow the 64KB ceiling); apply compatible winners together on a fresh tree and
   **RE-MEASURE** (combined speedup != product of individuals).
5. Commit the folded result, then fan out the next batch against the new baseline.

`isolation: "worktree"` on the Agent tool auto-removes the worktree if the agent made
no changes; otherwise it returns the path + branch.

## Environment gotchas
- **Subagents are sandbox-blocked from running python/GPU here** (they return "BASH
  BLOCKED"). So the fan-out-with-isolation pattern is for *editing* in parallel; the
  GPU **verification must run from the main session**. Plan accordingly: subagents
  prepare/edit isolated trees, main session benches them serially.
- FlyDSL has a **module-global SmemAllocator that finalizes once per process** -> one
  kernel shape per process. This is orthogonal to worktrees but bites the same parallel
  workflows: bench/correctness harnesses fork a subprocess per shape regardless of tree.
- The default repo here is `FlyDSL` at `/workspaces/amir/FlyDSL`; worktrees land under
  its `.claude/worktrees/`.

## Cleanup discipline
- Worktrees with no changes: auto-cleaned by the Agent `isolation` path; otherwise
  `ExitWorktree action:"remove"` (it refuses on uncommitted work — commit or pass
  `discard_changes:true` deliberately).
- Stale worktrees linger in `git worktree list`; prune with `git worktree remove` or
  `git worktree prune`.
- Never `rm -rf` a worktree dir without `git worktree remove` — it leaves a dangling
  registration.

## One-sentence takeaway
> Reach for a worktree only for genuinely parallel edits to the same source (fan-out
> levers), use EnterWorktree/Agent `isolation:"worktree"` to create them, remember
> subagents can't run the GPU here so fold/measure on the main session, and clean up
> with ExitWorktree/`git worktree remove`.
