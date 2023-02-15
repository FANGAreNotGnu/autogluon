dataset_names=("chest10" "nfl_logo" "kitchen" "dota")
presets=("best_quality" "high_quality" "medium_quality")

for d in ${dataset_names[*]}
do
  for p in ${presets[*]}
  do
        python3 finetune_presets.py \
            -d $d \
            -p $p \
            >& finetune_presets_logs/02101700_${d}_${p}.txt
  done
done
