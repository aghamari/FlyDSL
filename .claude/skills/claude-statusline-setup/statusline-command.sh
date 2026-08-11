#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
EFFORT=$(echo "$input" | jq -r '.effort.level // empty')
if [ -z "$EFFORT" ]; then
    CFG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
    EFFORT=$(jq -r '.effortLevel // empty' "$CFG_DIR/settings.json" 2>/dev/null)
fi
THINKING=$(echo "$input" | jq -r '.thinking.enabled // empty')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')
COST=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
DURATION_MS=$(echo "$input" | jq -r '.cost.total_duration_ms // 0')

CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'

# Pick bar color based on context usage
if [ "$PCT" -ge 90 ]; then BAR_COLOR="$RED"
elif [ "$PCT" -ge 70 ]; then BAR_COLOR="$YELLOW"
else BAR_COLOR="$GREEN"; fi

FILLED=$((PCT / 10)); EMPTY=$((10 - FILLED))
printf -v FILL "%${FILLED}s"; printf -v PAD "%${EMPTY}s"
BAR="${FILL// /█}${PAD// /░}"

MINS=$((DURATION_MS / 60000)); SECS=$(((DURATION_MS % 60000) / 1000))

BRANCH_NAME=$(git branch --show-current 2>/dev/null)

BRANCH=""
git rev-parse --git-dir > /dev/null 2>&1 && BRANCH="$BRANCH_NAME"

# Derive remote name from the current branch's upstream (falls back to origin, then public)
REMOTE_NAME=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null | cut -d/ -f1)
[ -z "$REMOTE_NAME" ] && git remote get-url origin >/dev/null 2>&1 && REMOTE_NAME="origin"
[ -z "$REMOTE_NAME" ] && git remote get-url public >/dev/null 2>&1 && REMOTE_NAME="public"

# Convert git SSH URL to HTTPS
REMOTE=$(git remote get-url "$REMOTE_NAME" 2>/dev/null | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')

GREEN='\033[32m'
YELLOW='\033[33m'
RESET='\033[0m'


if [ -n "$REMOTE" ]; then
    REPO_NAME=$(basename "$REMOTE")
    # OSC 8 format: \e]8;;URL\a then TEXT then \e]8;;\a
    # printf %b interprets escape sequences reliably across shells
    BRANCH=$(printf '%b' "\e]8;;${REMOTE}/tree/${BRANCH}\a${BRANCH}\e]8;;\a\n")
fi

GIT_STATUS=""
if git rev-parse --git-dir > /dev/null 2>&1; then
    STAGED=$(git diff --cached --numstat 2>/dev/null | wc -l | tr -d ' ')
    MODIFIED=$(git diff --numstat 2>/dev/null | wc -l | tr -d ' ')

    [ "$STAGED" -gt 0 ] && GIT_STATUS="${GREEN}+${STAGED}${RESET}"
    [ "$MODIFIED" -gt 0 ] && GIT_STATUS="${GIT_STATUS}${YELLOW}~${MODIFIED}${RESET}"
fi

SUFFIX=""
[ -n "$EFFORT" ] && SUFFIX="$EFFORT"
[ "$THINKING" = "true" ] && SUFFIX="${SUFFIX:+$SUFFIX }thinking"
[ -n "$SUFFIX" ] && SUFFIX=" ($SUFFIX)"
MODEL_LABEL="${MODEL}${SUFFIX}"

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${CYAN}[$MODEL_LABEL]${RESET} 📁 ${DIR##*/} | 🌿 $BRANCH $GIT_STATUS"
else
    echo -e "${CYAN}[$MODEL_LABEL]${RESET} 📁 ${DIR##*/}"
fi

# echo -e "${CYAN}[$MODEL]${RESET} 📁 ${DIR##*/}$BRANCH"
COST_FMT=$(printf '$%.2f' "$COST")
echo -e "${BAR_COLOR}${BAR}${RESET} ${PCT}% | ${YELLOW}${COST_FMT}${RESET} | ⏱️ ${MINS}m ${SECS}s"