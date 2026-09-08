# INVALID: these are pixel-arm numbers wearing a model-scaling label

Produced 6-7 Sep 2026 by `experiments/measure_femnist_c1_inputs_scaling.py` before
`run_cross_distribution_compositions.run_one` applied `attack.manipulate_update`.
Without that call `committed_scaling` and `committed_pixel` are the same experiment
(the two attacks share `poison_dataset`; the update-level scaling is the only difference),
so every value here is bit-identical to the frozen **pixel** cell:

    trimmed_mean  0.9881918374116122  ==  0.9881918374116121  (pixel, payoff_results.json)
    rfa           0.9977806926289444  ==  0.9977806926289444  (pixel)
    coord_median  0.6091787658118112  ==  0.6091787658118112  (pixel)

The true model-scaling values differ: trimmed_mean 0.9999, rfa 0.9990, coord_median 0.6567.

Kept, not deleted, because the bit-identity is the evidence for the diagnosis.
No paper number was ever taken from this directory.
