# Explain-Health

Explain-Health is an Android research prototype for classifying structured symptom profiles with local model inference. The repository includes the Java inference code, interface components, a serialized random forest, and a reproducible audit of the symptom benchmark.

The benchmark consists of structured symptom–label associations with no confirmed patient records or independently verified outcomes. The prototype and the audit provide no evidence for clinical screening, diagnosis, triage, or treatment decisions.

## Benchmark audit

The audit uses [the original repository snapshot](https://github.com/gebriye1986/Explain-Health/tree/430e1e7d5c87189241f1b8fd0a061eb17b42628e). The 4,920 rows contain 41 labels, 131 binary symptom features, and 304 distinct profiles. All 984 rows in the original stratified test set exactly match profiles present in training. The reproduced 100.0% accuracy for that split therefore measures agreement on profiles already seen during training.

After deduplication, 41 profiles equal the union of their class's symptoms, and the other 263 each omit one symptom from that union. Of the 304 held-out profiles across five folds, 280 are one symptom change from a training profile and 24 are two changes away. These folds compare models on distinct profiles from the same source.

| Classifier | Clean accuracy | Random symptom replacement | Rival-class symptom replacement |
| --- | ---: | ---: | ---: |
| Class prior | 3.3% | 3.3% | 3.3% |
| Class-union Jaccard | 100.0% | 99.8% | 90.8% |
| Bernoulli naive Bayes | 97.7% | 91.4% | 73.4% |
| Multinomial logistic regression | 100.0% | 99.2% | 78.9% |
| Depth-limited decision tree | 44.1% | 37.7% | 22.4% |
| Random forest | 99.7% | 91.2% | 68.8% |

Both replacement tests remove one symptom and add another. The rival-class test selects features using the reference label and training-fold symptom frequencies. The test measures model sensitivity under controlled input changes and does not estimate how people report symptoms.

The audit also reruns the forest after each of 2,325 single-symptom deletions. Among 2,324 symptoms marked as class matches, 102 deletions leave the original top-label score unchanged within tolerance and 34 increase it. A Python procedure that reuses unchanged tree paths reproduces full forest rescoring within numerical tolerance. It reduces additional tree traversals from 116,250 to 15,820. These counts exclude cache overhead and provide no measurement of Android latency.

## Reproduce the audit

See the [reproducibility instructions](reproducibility/README.md), [full results](reproducibility/results.json), and [fold assignments](reproducibility/folds.csv). The audit verifies the SHA-256 hashes of the original CSV and model before running. It writes results to a separate output directory.

## Android prototype and bundled model

| File | Purpose |
| --- | --- |
| [RandomForestModelManager.java](RandomForestModelManager.java) | Loads the bundled forest, encodes selected symptoms, and accumulates tree predictions locally. |
| [DatasetManager.java](DatasetManager.java) | Loads the symptom and precaution assets used by the interface. |
| [PredictionActivity.java](PredictionActivity.java) and [ResultActivity.java](ResultActivity.java) | Collect symptom selections and display model output. |
| [ProfileActivity.java](ProfileActivity.java), [HistoryManager.java](HistoryManager.java), and [AndroidManifest.xml](AndroidManifest.xml) | Implement profile and history behavior and declare app settings. |
| [DiseaseAndSymptoms.csv](DiseaseAndSymptoms.csv) | Structured symptom–label benchmark. |
| [rf_model.json](rf_model.json) | Historical 50-tree artifact. The audited file is 354,040 bytes. Its accuracy field is historical metadata. |
| [train_model.py](train_model.py) | Historical trainer and exporter. It uses the original row-level split. Running it from the repository root can overwrite the bundled JSON files. |

This snapshot contains Android source files, layouts, and assets. A separate Android project with its build configuration and resource structure is needed to build and test the app. The checked-in Java inference path runs classification locally. Report sharing and Android backup are separate features. The app's risk badges and percentages are prototype displays without clinical calibration. The new deletion analysis runs in Python.

The repository does not include handset measurements or the source records needed to reproduce a user study. The audit reports model-data sizes and tree traversal counts.

## Citation and reuse

When referencing this work, record the repository URL and the exact commit used. The historical repository snapshot declares no software or dataset license and does not establish upstream provenance or reuse terms for the symptom table. Confirm permissions with the repository maintainers and relevant data owners before redistributing the materials.
