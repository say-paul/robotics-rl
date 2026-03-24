#!/usr/bin/env bash
# =============================================================
# LeRobot Training Launcher for G1 Whole-Body Controller
# =============================================================
#
# Two-stage pipeline:
#   Stage 1 — SARM: Learn a task-progress reward model from video demos.
#   Stage 2 — SmolVLA: Fine-tune the vision-language-action policy using
#             the generated dataset (optionally with SARM-based RA-BC).
#
# Usage:
#   # Stage 1 only (SARM reward model, ~30 min)
#   bash training/train_lerobot.sh sarm
#
#   # Stage 2 only (SmolVLA fine-tune, hours-scale)
#   bash training/train_lerobot.sh smolvla
#
#   # Both stages sequentially (overnight run)
#   bash training/train_lerobot.sh all
#
# Prerequisites:
#   cd /home/lerobot && pip install -e ".[sarm,smolvla]"
# =============================================================

set -euo pipefail

DATASET_ROOT="/home/robotics-rl/datasets/g1_wb_stairs"
DATASET_REPO="g1_wb_stairs"
OUTPUT_BASE="/home/robotics-rl/outputs"
LEROBOT_DIR="/home/lerobot"

SARM_STEPS="${SARM_STEPS:-5000}"
SARM_BATCH="${SARM_BATCH:-32}"
SMOLVLA_STEPS="${SMOLVLA_STEPS:-20000}"
SMOLVLA_BATCH="${SMOLVLA_BATCH:-16}"

train_sarm() {
    echo "=== Stage 1: SARM Reward Model ==="
    echo "  dataset: ${DATASET_ROOT}"
    echo "  steps: ${SARM_STEPS},  batch: ${SARM_BATCH}"
    echo ""

    cd "${LEROBOT_DIR}"
    lerobot-train \
        --dataset.repo_id="${DATASET_REPO}" \
        --dataset.root="${DATASET_ROOT}" \
        --policy.type=sarm \
        --policy.annotation_mode=single_stage \
        --policy.image_key=observation.images.global_view \
        --policy.state_key=observation.state \
        --policy.push_to_hub=false \
        --batch_size="${SARM_BATCH}" \
        --steps="${SARM_STEPS}" \
        --output_dir="${OUTPUT_BASE}/sarm_g1" \
        --job_name=sarm_g1_stairs \
        --log_freq=100 \
        --save_freq=1000

    echo ""
    echo "SARM training complete. Checkpoint: ${OUTPUT_BASE}/sarm_g1"
}

train_smolvla() {
    echo "=== Stage 2: SmolVLA Fine-tune ==="
    echo "  dataset: ${DATASET_ROOT}"
    echo "  steps: ${SMOLVLA_STEPS},  batch: ${SMOLVLA_BATCH}"
    echo ""

    cd "${LEROBOT_DIR}"
    lerobot-train \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id="${DATASET_REPO}" \
        --dataset.root="${DATASET_ROOT}" \
        --policy.push_to_hub=false \
        --policy.max_action_dim=32 \
        --policy.max_state_dim=32 \
        --batch_size="${SMOLVLA_BATCH}" \
        --steps="${SMOLVLA_STEPS}" \
        --output_dir="${OUTPUT_BASE}/smolvla_g1" \
        --job_name=smolvla_g1_stairs \
        --log_freq=100 \
        --save_freq=2000

    echo ""
    echo "SmolVLA training complete. Checkpoint: ${OUTPUT_BASE}/smolvla_g1"
}

case "${1:-all}" in
    sarm)
        train_sarm
        ;;
    smolvla)
        train_smolvla
        ;;
    all)
        train_sarm
        echo ""
        echo "=========================================="
        echo ""
        train_smolvla
        ;;
    *)
        echo "Usage: $0 {sarm|smolvla|all}"
        exit 1
        ;;
esac

echo ""
echo "All requested training stages complete."
