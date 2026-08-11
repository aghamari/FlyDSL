---
name: claude-statusline-setup
description: Install or modify the Claude Code status line — the bottom bar showing model+effort, a colored context-usage bar, cwd, git branch (as a clickable remote link) + staged/modified counts, cost, and elapsed time. Use when asked to set up, fix, restore, or customize the status line, wire statusLine into settings.json, or change what the bar displays. Ships a ready-to-use script.
argument-hint: [nothing, or a field to add/change]
---

# Claude Code Status Line Setup

Install and customize the Claude Code status line. A working script ships next to
this skill at **`statusline-command.sh`** (in this skill's directory) — copy it in
and point `settings.json` at it.

## What the bar renders
Two lines:
1. `[Model (effort thinking)]  📁 <cwd-basename> | 🌿 <branch> +<staged>~<modified>`
   - **Model + effort** in cyan. Effort comes from the live input
     (`.effort.level`) or falls back to `settings.json`'s `.effortLevel`; appends
     `thinking` when `.thinking.enabled`.
   - **branch** is wrapped in an OSC-8 hyperlink to `<remote>/tree/<branch>` (the
     remote URL is derived from the branch's upstream, else `origin`, else `public`,
     and SSH `git@github.com:` is rewritten to `https://github.com/`). Click-through
     in terminals that support OSC-8.
   - **git status**: `+N` staged (green) / `~N` modified (yellow). Omitted outside a
     git repo (then only model + cwd show).
2. `<context bar> <pct>% | $<cost> | ⏱️ <m>m <s>s`
   - 10-segment **context-usage bar** (`█`/`░`), colored green <70% / yellow 70-90% /
     red >=90%, from `.context_window.used_percentage`.
   - **cost** `$X.XX` from `.cost.total_cost_usd`; **elapsed** from
     `.cost.total_duration_ms`.

It reads the status JSON Claude Code pipes on stdin via `jq` (so **`jq` must be on
PATH**).

## Install
```bash
# 1. Put the script where every session can reach it:
mkdir -p ~/.claude
cp "$CLAUDE_SKILL_DIR/statusline-command.sh" ~/.claude/statusline-command.sh   # or copy from this skill dir
chmod +x ~/.claude/statusline-command.sh

# 2. Point settings.json at it (use the update-config skill, or edit directly):
```
Add to `~/.claude/settings.json`:
```json
{
  "statusLine": { "type": "command", "command": "~/.claude/statusline-command.sh" }
}
```
Use an **absolute** path if `~` isn't expanded in your setup
(`/home/<user>/.claude/statusline-command.sh`). Restart / new session to see it.

## Verify
```bash
echo '{"model":{"display_name":"Opus 4"},"workspace":{"current_dir":"/tmp"},
"context_window":{"used_percentage":42},"cost":{"total_cost_usd":1.23,"total_duration_ms":65000}}' \
  | ~/.claude/statusline-command.sh
```
Should print the two formatted lines. If you get raw JSON or errors, `jq` is missing
or the path/permissions are wrong.

## Customize (common edits)
- **Add a field** (e.g. session id, output style): read it with another `jq -r '.<key>'`
  from the same stdin `input` and append to one of the two `echo -e` lines.
- **Change the bar width**: it's hard-coded to 10 segments (`PCT/10`); change the
  divisor and the `FILLED/EMPTY` math together.
- **Recolor thresholds**: edit the `if [ "$PCT" -ge 90 ]` / `-ge 70` block.
- **Drop the hyperlink** (some terminals show escape junk): replace the OSC-8 `BRANCH=$(printf ...)`
  block with plain `BRANCH="$BRANCH_NAME"`.

## Portability note
This is a machine-level file outside any repo, so it doesn't travel with `git clone`.
To carry it to a new machine, copy `~/.claude/statusline-command.sh` (the KB's
SETUP.md §4 documents this alongside CLAUDE.md + user skills) and re-point
`settings.json`. A vendored copy lives in the knowledge-base repo at
`.claude/statusline-command.sh` and in this skill's directory.

## Related
- `update-config` skill — the sanctioned way to edit `settings.json` (permissions,
  env, hooks, statusLine).
- KB `SETUP.md` §4 (Carry your global config) — backup/restore of CLAUDE.md + skills +
  this script across machines.

## One-sentence takeaway
> Copy the shipped `statusline-command.sh` to `~/.claude/`, make it executable, set
> `statusLine.command` in settings.json to its path (jq required), and edit the two
> `echo -e` lines to change what the bar shows.
