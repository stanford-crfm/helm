for x in {a..r}
do
    echo $x
    python3 compute_metrics_from_ngrams.py --ngrams-path "pile_ngrams/output_stats_pi..gram_xa${x}_ngrams" --scenario-path "filtered_scenario_data_new" --out-path "new_filtered_pile_metrics_${x}" --filter-path "scenario_spec_i.._ids_1000.jsonl"
done

