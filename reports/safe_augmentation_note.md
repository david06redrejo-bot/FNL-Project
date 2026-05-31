# Safe Clinical Text Augmentation Decision

We did not use random deletion, unverified synonym replacement, negation changes,
or back-translation in the final experiments. Clinical literals can be short and
semantically dense, so a small wording change can alter the correct ICD category.

The only tested strategies are data-handling strategies that do not invent new
clinical meaning: non-conflicting duplicate handling and class-balanced sampling.
Back-translation and synonym augmentation remain future work unless reviewed by
domain experts or controlled with verified medical terminology resources.

A custom class-balanced batch sampler was considered but left as future work.
For this phase, `WeightedRandomSampler` was the lower-risk way to test whether
more exposure to rare categories helps without adding extra batch construction
complexity.
