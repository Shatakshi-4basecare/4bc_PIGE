#!/bin/bash
set -e

# Activate Environment
eval "$(conda shell.bash hook)"
conda activate pige_atlas

# Resource Profiler function
monitor_resources() {
    local log_file=$1
    local pid=$2
    echo "Timestamp, CPU_%, RAM_MB, GPU_%, VRAM_MB" > "$log_file"
    
    while kill -0 "$pid" 2>/dev/null; do
        local cpu_ram=$(ps -p "$pid" -o %cpu,rss | tail -n 1)
        local cpu=$(echo $cpu_ram | awk '{print $1}')
        local ram_kb=$(echo $cpu_ram | awk '{print $2}')
        local ram_mb=$((ram_kb / 1024))
        
        local gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
        local gpu=$(echo $gpu_info | cut -d',' -f1)
        local vram=$(echo $gpu_info | cut -d',' -f2)
        
        echo "$(date +%H:%M:%S), ${cpu}%, ${ram_mb}MB, ${gpu}%, ${vram}MB" >> "$log_file"
        sleep 5
    done
}

# Python Execution and Timing wrapper
run_and_profile() {
    local step_name=$1
    local script_file=$2
    local drug_list=$3
    local log_dir=$4
    
    echo "  -> Starting $step_name..."
    
    python "$script_file" "$drug_list" > "${log_dir}/${step_name}.log" 2>&1 &
    local py_pid=$!
    
    monitor_resources "${log_dir}/${step_name}_resources.csv" "$py_pid" &
    { time wait "$py_pid"; } 2> "${log_dir}/${step_name}_time.txt"
}

# ==========================================
# 1. READ TEXT FILE AND COMBINE DRUG NAMES
# ==========================================
if [ ! -f "drugs_list.txt" ]; then
    echo "ERROR: drugs_list.txt not found!"
    exit 1
fi

DRUGS_STRING=""
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Append drug name and a comma
    DRUGS_STRING="${DRUGS_STRING}${line},"
done < "drugs_list.txt"

# Remove the trailing comma at the end of the string
DRUGS_STRING=${DRUGS_STRING%,}

if [ -z "$DRUGS_STRING" ]; then
    echo "ERROR: No valid drugs found in drugs_list.txt!"
    exit 1
fi

# ==========================================
# 2. RUN PIPELINE ONCE FOR THE ENTIRE BATCH
# ==========================================
echo "======================================================"
echo "▶ STARTING MULTI-DRUG PIPELINE"
echo "Drugs detected: $DRUGS_STRING"
echo "======================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="pipeline_logs/MultiDrugBatch_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

run_and_profile "step1_train" "run_step1_train.py" "$DRUGS_STRING" "$LOG_DIR"
run_and_profile "step2_knockout" "run_step2_knockout.py" "$DRUGS_STRING" "$LOG_DIR"
run_and_profile "step3_graphs" "run_step3_graphs.py" "$DRUGS_STRING" "$LOG_DIR"

echo "======================================================"
echo "PIPELINE COMPLETE! Summaries generated."
echo "======================================================"