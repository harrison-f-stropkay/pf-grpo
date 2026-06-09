#!/usr/bin/env bash
set -euo pipefail

workspace_name="${1:-}"

if [[ -z "$workspace_name" ]]; then
  echo "Usage: $0 <workspace-name>" >&2
  exit 1
fi

runai workspace submit "$workspace_name" \
  --project strophf1 \
  --image docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1 \
  --gpu-devices-request 1 \
  --cpu-core-request 8 \
  --cpu-memory-request 64G \
  --existing-pvc claimname=pf-grpo-project-abz8u,path=/home/apluser \
  --node-pools dgx-h100-80gb \
  --node-pools abyss-hgx-h100-80gb \
  --node-pools aos-a40-48gb \
  --node-pools itsd-general \
  --node-pools default \
  --external-url container=8888 \
  --run-as-user \
  --preemptible \
  --environment HOME=/home/apluser \
  --environment USER=apluser
