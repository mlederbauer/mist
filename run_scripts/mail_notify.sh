# Source this after the #SBATCH header in any submit_*.sh script to email
# yourself the tail of .out/.err on job exit, instead of Slurm's generic
# END/FAIL notice. Drop the script's own --mail-type/--mail-user lines when
# using this (they'd otherwise send a second, useless email).
notify() {
    local status=$1
    local out="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
    local err="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    {
        echo "To: ${MAIL_NOTIFY_USER:-magled@mit.edu}"
        echo "Subject: Slurm job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) $status"
        echo
        echo "=== tail -n 200 $err ==="
        tail -n 200 "$err" 2>/dev/null
        echo
        echo "=== tail -n 200 $out ==="
        tail -n 200 "$out" 2>/dev/null
    } | sendmail -t
}
trap 'notify "exit code $?"' EXIT
