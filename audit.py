"""Reproduce the repository audit and the held-out symptom-profile experiments."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from scipy.spatial.distance import cdist
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


COMMIT = "430e1e7d5c87189241f1b8fd0a061eb17b42628e"
DATA_SHA256 = "ebbd391c4ba4d64f57a00eb3d0a55f0ca9b920b0c5de6b9af4234e72519c9618"
ASSET_SHA256 = "ac1181d9d34f71c51643e82dbbc6f28d86f6b21eb7faf8ea797ecd51795577f8"
SEED = 42
DRAWS = 20


def load_profiles(path):
    def normalize(cell):
        return cell.strip().replace("_", " ").lower()

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        raw = list(reader)
    if any(len(row) != len(header) for row in raw):
        raise ValueError("A CSV row has a different field count from the header.")
    rows = []
    for row in raw:
        symptoms = tuple(sorted({normalize(cell) for cell in row[1:]} - {"", "nan", "null"}))
        if row[0].strip() and symptoms:
            rows.append((row[0].strip(), symptoms))
    return raw, rows


def encode(rows, vocabulary, labels):
    index = {symptom: i for i, symptom in enumerate(vocabulary)}
    label_index = {label: i for i, label in enumerate(labels)}
    x = np.zeros((len(rows), len(vocabulary)), dtype=np.int8)
    for i, (_, symptoms) in enumerate(rows):
        x[i, [index[symptom] for symptom in symptoms]] = 1
    return x, np.array([label_index[label] for label, _ in rows])


def forest():
    return RandomForestClassifier(n_estimators=50, max_depth=15, random_state=SEED, n_jobs=1)


def scores(y, probabilities, class_count):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y
    # Ten equal-width bins for top-label expected calibration error.
    bins = np.minimum((10 * confidence).astype(int), 9)
    ece = 0.0
    for bin_id in range(10):
        member = bins == bin_id
        if member.any():
            ece += member.mean() * abs(correct[member].mean() - confidence[member].mean())
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=np.arange(class_count),
                                    average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=np.arange(class_count))),
        "top_label_ece_10": float(ece),
    }


def count_components(distances, radius):
    unseen = set(range(len(distances)))
    components = []
    while unseen:
        pending = [unseen.pop()]
        connected = []
        while pending:
            current = pending.pop()
            connected.append(current)
            neighbors = unseen.intersection(np.flatnonzero(distances[current] <= radius))
            unseen.difference_update(neighbors)
            pending.extend(neighbors)
        components.append(connected)
    return components


def perturb(x, observed, remove, add, seed):
    rng = np.random.default_rng(seed)
    perturbed = np.repeat(x[None, :, :], DRAWS, axis=0)
    for draw in perturbed:
        for row, original in zip(draw, x):
            active = np.flatnonzero(original)
            inactive = np.flatnonzero(observed & (original == 0))
            if remove:
                row[rng.choice(active, min(remove, len(active)), replace=False)] = 0
            if add:
                row[rng.choice(inactive, add, replace=False)] = 1
    return perturbed.reshape(-1, x.shape[1])


class ClassUnionJaccard:
    def fit(self, x, y):
        self.profiles = np.array([x[y == label].any(axis=0) for label in np.unique(y)])
        return self

    def similarity(self, x):
        intersection = x @ self.profiles.T.astype(np.int16)
        union = x.sum(axis=1, keepdims=True) + self.profiles.sum(axis=1) - intersection
        return intersection / union

    def predict_proba(self, x):
        similarity = self.similarity(x)
        total = similarity.sum(axis=1, keepdims=True)
        return np.divide(similarity, total, out=np.full_like(similarity, 1 / similarity.shape[1]),
                         where=total > 0)


def structured_perturbation(x, y, x_train, y_train, prototypes):
    """Replace a class-associated symptom with one more associated with a rival."""
    changed = x.copy()
    similarity = prototypes.similarity(x)
    eligible = (x == 0).astype(np.int16) @ prototypes.profiles.T.astype(np.int16) > 0
    similarity[~eligible] = -1
    similarity[np.arange(len(x)), y] = -1
    rivals = similarity.argmax(axis=1)
    frequencies = np.array([x_train[y_train == label].mean(axis=0)
                            for label in range(len(prototypes.profiles))])
    for index, (row, label, rival) in enumerate(zip(x, y, rivals)):
        difference = frequencies[label] - frequencies[rival]
        active = np.flatnonzero(row)
        inactive = np.flatnonzero((row == 0) & (frequencies[rival] > 0))
        if not inactive.size:
            raise ValueError("The selected rival has no eligible inactive symptom.")
        changed[index, active[difference[active].argmax()]] = 0
        changed[index, inactive[difference[inactive].argmin()]] = 1
    return changed


def walk_tree(tree, row, removed=-1):
    node = 0
    path = set()
    while tree.children_left[node] >= 0:
        feature = int(tree.feature[node])
        path.add(feature)
        value = 0 if feature == removed else row[feature]
        node = tree.children_left[node] if value <= tree.threshold[node] else tree.children_right[node]
    return path, tree.value[node, 0]


def cached_deletions(model, row, active):
    """Reuse the original leaf for a tree whose original path omits the deleted feature."""
    trees = [estimator.tree_ for estimator in model.estimators_]
    original = [walk_tree(tree, row) for tree in trees]
    total = np.sum([leaf for _, leaf in original], axis=0)
    changed = np.tile(total, (len(active), 1))
    traversals = 0
    for tree, (path, leaf) in zip(trees, original):
        for index, feature in enumerate(active):
            if feature in path:
                _, alternative = walk_tree(tree, row, removed=feature)
                changed[index] += alternative - leaf
                traversals += 1
    return changed / len(trees), traversals


def deletion_checks(model, x, probabilities, profiles):
    results = []
    for row, probability in zip(x, probabilities):
        predicted = probability.argmax()
        active = np.flatnonzero(row)
        alternatives = np.repeat(row[None, :], len(active), axis=0)
        alternatives[np.arange(len(active)), active] = 0
        changed = model.predict_proba(alternatives)
        cached, traversals = cached_deletions(model, row, active)
        parity_error = float(np.max(np.abs(changed - cached)))
        if not np.allclose(changed, cached, rtol=0, atol=1e-12):
            raise ValueError("Cached deletion differs from full re-evaluation.")
        delta = probability[predicted] - changed[:, predicted]
        in_profile = profiles[predicted, active]
        results.append({
            "matches": delta[in_profile].tolist(),
            "nonmatches": delta[~in_profile].tolist(),
            "best_deletion": float(delta.max()),
            "uniform_deletion": float(delta.mean()),
            "argmax_flip_fraction": float((changed.argmax(axis=1) != predicted).mean()),
            "additional_full_tree_traversals": len(active) * len(model.estimators_),
            "additional_cached_tree_traversals": traversals,
            "cached_score_max_absolute_error": parity_error,
        })
    return results


def summarize_fidelity(rows):
    paired = [row for row in rows if row["matches"] and row["nonmatches"]]
    result = {
        "profiles": len(rows),
        "profiles_with_both_groups": len(paired),
        "mean_maximum_score_drop": float(np.mean([row["best_deletion"] for row in rows])),
        "mean_uniform_score_drop": float(np.mean([row["uniform_deletion"] for row in rows])),
        "mean_argmax_flip_fraction": float(np.mean([row["argmax_flip_fraction"] for row in rows])),
        "additional_full_tree_traversals": sum(row["additional_full_tree_traversals"] for row in rows),
        "additional_cached_tree_traversals": sum(row["additional_cached_tree_traversals"] for row in rows),
        "cached_score_max_absolute_error": max(row["cached_score_max_absolute_error"] for row in rows),
    }
    if paired:
        result["paired_mean_matched_drop"] = float(np.mean([np.mean(row["matches"]) for row in paired]))
        result["paired_mean_unmatched_drop"] = float(np.mean([np.mean(row["nonmatches"]) for row in paired]))
    for key in ("matches", "nonmatches"):
        values = np.array([delta for row in rows for delta in row[key]])
        result[key] = {
            "deletions": len(values),
            "positive": int((values > 1e-12).sum()),
            "zero": int((np.abs(values) <= 1e-12).sum()),
            "negative": int((values < -1e-12).sum()),
        }
    return result


def numeric_payload_bytes(model, feature_count, class_count):
    # Score coefficients and probabilities use float32, tree indices use int32.
    # A tree node has a feature plus two child indices. Leaves store class scores.
    if isinstance(model, DummyClassifier):
        return 4 * class_count
    if isinstance(model, ClassUnionJaccard):
        return int(np.packbits(model.profiles, axis=1).nbytes)
    if isinstance(model, (BernoulliNB, LogisticRegression)):
        return 4 * class_count * (feature_count + 1)
    trees = model.estimators_ if isinstance(model, RandomForestClassifier) else [model]
    return sum(12 * int(tree.tree_.node_count) + 4 * class_count * int(tree.tree_.n_leaves)
               for tree in trees)


def class_union_structure(x, y, labels):
    profiles = []
    for index, label in enumerate(labels):
        class_rows = x[y == index]
        union = class_rows.any(axis=0)
        missing = union.sum() - class_rows.sum(axis=1)
        profiles.append({
            "label": label,
            "profiles": len(class_rows),
            "union_size": int(union.sum()),
            "distance_to_union": {int(distance): int(count) for distance, count
                                  in zip(*np.unique(missing, return_counts=True))},
        })
    return profiles


def run(repo):
    data = repo / "DiseaseAndSymptoms.csv"
    model_file = repo / "rf_model.json"
    for file, expected in ((data, DATA_SHA256), (model_file, ASSET_SHA256)):
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{file.name} does not match the byte hash for commit {COMMIT}.")
    raw, rows = load_profiles(data)
    group_labels = defaultdict(set)
    for label, symptoms in rows:
        group_labels[symptoms].add(label)
    conflicts = {symptoms for symptoms, labels in group_labels.items() if len(labels) > 1}
    unique = list(dict.fromkeys((label, symptoms) for label, symptoms in rows if symptoms not in conflicts))
    vocabulary = sorted({symptom for _, symptoms in rows for symptom in symptoms})
    labels = sorted({label for label, _ in rows})
    x_raw, y_raw = encode(rows, vocabulary, labels)
    x, y = encode(unique, vocabulary, labels)
    class_count = len(labels)
    row_train, row_test = train_test_split(np.arange(len(y_raw)), test_size=0.2,
                                           random_state=SEED, stratify=y_raw)
    train_profiles = {rows[i][1] for i in row_train}
    row_forest = forest().fit(x_raw[row_train], y_raw[row_train])
    row_scores = scores(y_raw[row_test], row_forest.predict_proba(x_raw[row_test]), class_count)
    distances = np.rint(cdist(x, x, metric="hamming") * x.shape[1]).astype(int)
    components = {}
    for radius in (1, 2):
        groups = count_components(distances, radius)
        group_counts = Counter(int(c) for group in groups for c in np.unique(y[group]))
        components[radius] = {
            "component_count": len(groups),
            "largest": max(map(len, groups)),
            "mixed_label_components": sum(len(np.unique(y[group])) > 1 for group in groups),
            "classes_confined_to_one_component": sum(count == 1 for count in group_counts.values()),
        }

    model_factories = {
        "prior": DummyClassifier,
        "jaccard": ClassUnionJaccard,
        "bernoulli_nb": BernoulliNB,
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
        "tree": lambda: DecisionTreeClassifier(max_depth=15, random_state=SEED),
        "forest": forest,
    }
    scenarios = {
        "clean": (0, 0), "remove_1": (1, 0), "remove_2": (2, 0),
        "add_1": (0, 1), "add_2": (0, 2), "remove_1_add_1": (1, 1),
    }
    results = {name: defaultdict(list) for name in model_factories}
    payload_sizes = defaultdict(list)
    collision_counts = defaultdict(Counter)
    novel_predictions = {name: defaultdict(Counter) for name in model_factories}
    fold_ids = np.empty(len(x), dtype=int)
    nearest = np.empty(len(x), dtype=int)
    fidelity = []
    clean_predictions = {}
    for fold, (train, test) in enumerate(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=SEED).split(x, y)):
        fold_ids[test] = fold
        nearest[test] = distances[np.ix_(test, train)].min(axis=1)
        assert (nearest[test] > 0).all()
        observed = x[train].any(axis=0)
        fold_inputs = {
            name: perturb(x[test], observed, remove, add,
                          np.random.SeedSequence([SEED, fold, index]))
            for index, (name, (remove, add)) in enumerate(scenarios.items()) if name != "clean"
        }
        fold_inputs["clean"] = x[test]
        prototypes = ClassUnionJaccard().fit(x[train], y[train])
        fold_inputs["rival_replace"] = structured_perturbation(x[test], y[test], x[train],
                                                               y[train], prototypes)
        profiles = prototypes.profiles
        training_keys = {row.tobytes(): label for row, label in zip(x[train], y[train])}
        labels_by_scenario = {}
        novel_by_scenario = {}
        for scenario, inputs in fold_inputs.items():
            truth = np.tile(y[test], len(inputs) // len(test))
            seen_labels = np.array([training_keys.get(row.tobytes(), -1) for row in inputs])
            novel = seen_labels == -1
            labels_by_scenario[scenario] = truth
            novel_by_scenario[scenario] = novel
            collision_counts[scenario].update({
                "queries": len(inputs),
                "equal_to_training_vector": int((~novel).sum()),
                "equal_to_training_vector_with_other_label": int(((~novel) & (seen_labels != truth)).sum()),
            })
        for name, factory in model_factories.items():
            model = factory().fit(x[train], y[train])
            payload_sizes[name].append(numeric_payload_bytes(model, x.shape[1], class_count))
            for scenario, inputs in fold_inputs.items():
                truth = labels_by_scenario[scenario]
                novel = novel_by_scenario[scenario]
                probabilities = model.predict_proba(inputs)
                results[name][scenario].append((truth, probabilities))
                novel_predictions[name][scenario].update({
                    "total": int(novel.sum()),
                    "correct": int((probabilities[novel].argmax(axis=1) == truth[novel]).sum()),
                })
                if scenario == "clean":
                    if name == "forest":
                        fidelity.extend(deletion_checks(model, inputs, probabilities, profiles))
                    clean_predictions.setdefault(name, np.empty((len(x), class_count)))[test] = probabilities

    by_distance = {}
    for threshold in (1, 2, 3):
        mask = nearest >= threshold
        by_distance[threshold] = {
            "profiles": int(mask.sum()),
            "represented_classes": int(len(np.unique(y[mask]))),
            "accuracy": {name: float((prediction[mask].argmax(axis=1) == y[mask]).mean())
                         for name, prediction in clean_predictions.items()} if mask.any() else {},
        }
    model_results = {
        model: {scenario: scores(np.concatenate([block[0] for block in blocks]),
                                  np.concatenate([block[1] for block in blocks]), class_count)
                for scenario, blocks in scenarios_out.items()}
        for model, scenarios_out in results.items()
    }
    asset = json.loads(model_file.read_text(encoding="utf-8"))
    class_structure = class_union_structure(x, y, labels)
    union_distances = Counter()
    for profile in class_structure:
        union_distances.update(profile["distance_to_union"])
    report = {
        "source_commit": COMMIT,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "csv_sha256": DATA_SHA256,
        "asset_sha256": ASSET_SHA256,
        "asset_bytes": model_file.stat().st_size,
        "asset_declared_accuracy": asset["accuracy"],
        "asset_trees": len(asset["trees"]),
        "raw_rows": len(raw),
        "exact_csv_duplicate_rows_after_first": len(raw) - len({tuple(row) for row in raw}),
        "eligible_rows": len(rows),
        "symptoms": len(vocabulary),
        "classes": class_count,
        "training_vocabulary_matches_export": vocabulary == asset["symptoms"],
        "different_vocabulary_positions_zero_based": [
            {"index": i, "training": training, "exported": exported}
            for i, (training, exported) in enumerate(zip(vocabulary, asset["symptoms"]))
            if training != exported
        ],
        "unique_normalized_label_profile_pairs": len(set(rows)),
        "unique_normalized_symptom_profiles": len(group_labels),
        "conflicting_normalized_profiles": len(conflicts),
        "rows_excluded_for_label_conflicts": sum(symptoms in conflicts for _, symptoms in rows),
        "deduplicated_class_counts": dict(sorted(Counter(label for label, _ in unique).items())),
        "class_union_distances": dict(sorted(union_distances.items())),
        "class_union_structure": class_structure,
        "legacy_split": {
            "train_rows": len(row_train),
            "test_rows": len(row_test),
            "overlapping_test_rows": sum(rows[i][1] in train_profiles for i in row_test),
            "distinct_test_profiles": len({rows[i][1] for i in row_test}),
            "overlapping_distinct_test_profiles": len({rows[i][1] for i in row_test} & train_profiles),
            **row_scores,
        },
        "components_by_hamming_radius": components,
        "near_neighbor_sensitivity": by_distance,
        "robustness_draws_per_profile": DRAWS,
        "perturbation_collisions": collision_counts,
        "models": model_results,
        "clean_accuracy_by_test_fold": {
            name: [float((prediction[fold_ids == fold].argmax(axis=1) == y[fold_ids == fold]).mean())
                   for fold in range(5)] for name, prediction in clean_predictions.items()
        },
        "estimated_numeric_payload_bytes_by_test_fold": dict(payload_sizes),
        "accuracy_excluding_training_vector_collisions": {
            model: {scenario: counts["correct"] / counts["total"]
                    for scenario, counts in scenario_rows.items() if counts["total"]}
            for model, scenario_rows in novel_predictions.items()
        },
        "fidelity": summarize_fidelity(fidelity),
    }
    fold_rows = [(i + 1, label, fold_ids[i] + 1, nearest[i])
                 for i, (label, _) in enumerate(unique)]
    return report, fold_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="unpacked pinned Explain-Health repository")
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    report, fold_rows = run(args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["deduplicated_profile_index", "label", "test_fold", "nearest_training_hamming"])
        writer.writerows(fold_rows)


if __name__ == "__main__":
    main()
