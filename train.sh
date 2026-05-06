source setup_env.sh

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

CONFIG=$1      
EXP_ID=$(date "+%Y%m%d-%H%M%S")  
EXP_NAME="${CONFIG}"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}_${EXP_ID}.log"

uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

uv run scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite  2>&1 | tee -a "$LOG_FILE"
