#!/usr/bin/env bash
# Push code to the cluster, submit a training job, check status, and pull results back.
#
# Requires passwordless (key-based) SSH to run non-interactively; otherwise it will
# just prompt for your password/2FA each time, same as running ssh/scp by hand.
#
# Usage:
#   ./deploy.sh setup                            # one-time: create venv + install requirements.txt
#   ./deploy.sh push                             # copy the train_*.py scripts / train.job / requirements.txt over
#   ./deploy.sh submit <script> <architecture>   # push, then submit train.job, e.g. submit train_25d.py unetplusplus
#   ./deploy.sh status                           # show your jobs in the queue
#   ./deploy.sh pull <job_id>                    # copy back results/<job_id>* and its logs

set -euo pipefail

REMOTE_HOST="voievuds@rocket.hpc.ut.ee"
REMOTE_DIR="dimas_challenge"

usage() {
    sed -n '2,/^$/p' "$0"
    exit 1
}

cmd="${1:-}"
[ -z "$cmd" ] && usage

case "$cmd" in
  setup)
    echo "Copying requirements.txt and creating a venv on $REMOTE_HOST..."
    scp requirements.txt "$REMOTE_HOST:$REMOTE_DIR/"
    ssh "$REMOTE_HOST" "
        cd $REMOTE_DIR &&
        module load python &&
        python -m venv .venv &&
        source .venv/bin/activate &&
        pip install -r requirements.txt
    "
    echo
    echo "Venv created. torch is NOT installed by this step on purpose -- install it"
    echo "yourself, matched to the falcon nodes' Tesla V100 GPUs (compute capability 7.0,"
    echo "which newer cu13x builds no longer ship kernels for -- use cu126 instead):"
    echo "  ssh $REMOTE_HOST"
    echo "  cd $REMOTE_DIR && source .venv/bin/activate && pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126"
    ;;

  push)
    echo "Copying $(echo train_*.py), train.job, requirements.txt to $REMOTE_HOST:$REMOTE_DIR..."
    scp train_*.py train.job requirements.txt "$REMOTE_HOST:$REMOTE_DIR/"
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/logs $REMOTE_DIR/results"
    ;;

  submit)
    script="${2:-}"
    architecture="${3:-}"
    if [ -z "$script" ] || [ -z "$architecture" ]; then
        echo "Usage: $0 submit <script> <architecture>   e.g. $0 submit train_25d.py unetplusplus"
        exit 1
    fi
    [ -f "$script" ] || { echo "$script not found here"; exit 1; }

    "$0" push
    # the job name becomes part of the log file names and of the results folder
    name="${architecture}_${script%.py}"
    echo "Submitting $name ..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && sbatch --job-name=$name --export=ALL,SCRIPT=$script,ARCHITECTURE=$architecture train.job"
    ;;

  status)
    ssh "$REMOTE_HOST" 'squeue -u $USER'
    ;;

  pull)
    jobid="${2:-}"
    if [ -z "$jobid" ]; then
        echo "Usage: $0 pull <job_id>"
        exit 1
    fi
    mkdir -p results logs
    echo "Pulling results/${jobid}* ..."
    scp -r "$REMOTE_HOST:$REMOTE_DIR/results/${jobid}"* results/
    echo "Pulling logs for job $jobid ..."
    scp "$REMOTE_HOST:$REMOTE_DIR/logs/"*"_${jobid}"[_.]* logs/ 2>/dev/null \
        || echo "(no matching log files found yet)"
    ;;

  *)
    usage
    ;;
esac
