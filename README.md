# Explain-Health reproducibility package

This package reanalyzes the public Explain-Health repository at commit **430e1e7d5c87189241f1b8fd0a061eb17b42628e**. Download and unpack the [pinned repository ZIP](https://codeload.github.com/gebriye1986/Explain-Health/zip/430e1e7d5c87189241f1b8fd0a061eb17b42628e). The script verifies the SHA-256 hashes of the source CSV and original JSON model. The package does not redistribute those inputs.

Use Python 3.10.11 and install the pinned dependencies in an isolated Python environment. From the extracted evidence folder, run the following, substituting the path to the downloaded repository folder.

~~~text
python -m pip install -r requirements.txt
python audit.py "path/to/Explain-Health-430e1e7d5c87189241f1b8fd0a061eb17b42628e" --output reproduced
~~~

- **results.json** contains the observed counts, file hashes, exact-overlap replication, nearest-training-distance strata, class-union structure, fold-specific clean results, model metrics, perturbation collisions, model data sizes, and deletion checks. Accuracy, macro F1, and expected calibration error are reported as fractions.
- **folds.csv** maps each first-occurrence deduplicated profile to its test fold and nearest training distance. Profile IDs and folds are one-based. The source CSV order determines profile IDs.
- Six classifiers share five stratified folds and a fixed 131-column symptom schema. The five randomized scenarios use identical draws across models. A sixth scenario makes one deterministic rival-class swap per profile using the reference label and the training-fold symptom frequencies. This scenario tests sensitivity to class-informed edits. It cannot estimate what users report. Accuracy excluding exact training-vector collisions is reported separately.
- The fidelity audit compares a training-fold class union with signed changes in the original forest score after deleting each active feature. It verifies the complete cached class-score vector against full forest rescoring and counts the extra traversals in both procedures. The original forest scoring pass is shared. The Python cache excludes overhead from the reported traversal counts. The checked-in Android inference path does not populate the per-class explanation fields passed to the result view.
- Numeric payload estimates assume 32-bit parameters and indices, with one packed bit per symptom for the Jaccard comparator. Sizes exclude vocabulary strings, metadata, interpreter or object overhead, and hardware effects. The original JSON file size is measured separately.
- The dataset supplies structured associations without verified clinical provenance or external outcomes. The package contains no patient evaluation, participant responses, mobile benchmarks, or ethics approvals.
