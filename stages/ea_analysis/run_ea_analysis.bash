RUN_OUTPUT_DIR=/home/lukas/projects/sa_ea/stages/run_ea/stages/run_ea/run_outputs/stk_ea_runs
ANALYSIS_SCRIPT=/home/lukas/projects/sa_ea/stages/ea_analysis/sa_score.py
SA_MODEL_PATH=/home/lukas/temp/ea_analysis/sa_model.pkl
STK_PATH=/home/lukas/projects/stk/src

for RUN in {1..6}
do
    mkdir $RUN
    POPULATION_PATH="$RUN_OUTPUT_DIR"/"$RUN"/scratch/generation_0.json
    PYTHONPATH="$STK_PATH" python "$ANALYSIS_SCRIPT" "$POPULATION_PATH" "$SA_MODEL_PATH" "$RUN" &> "$RUN".log
done
