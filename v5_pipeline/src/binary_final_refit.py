from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier

from .binary_candidates import OBJECTIVE, BinaryProcessCandidate
from .features import RawFeatureEncoder


def refit_binary_process_candidate(
    template: BinaryProcessCandidate,
    partitions: Any,
    *,
    source_dataset_id: str,
    history_sha256: str,
) -> BinaryProcessCandidate:
    """Refit the binary process head through September and calibrate in October."""

    training = tuple(partitions.train)
    calibration = tuple(partitions.probability_calibration)
    training_ids = tuple(row.row_id for row in training)
    calibration_ids = tuple(row.row_id for row in calibration)
    if (
        not training
        or not calibration
        or len(training_ids) != len(set(training_ids))
        or len(calibration_ids) != len(set(calibration_ids))
        or set(training_ids) != set(partitions.train_labels)
        or set(calibration_ids) != set(partitions.probability_calibration_labels)
    ):
        raise ValueError("binary final training/calibration identities must match exactly")
    if set(training_ids) & set(calibration_ids):
        raise ValueError("binary final training and calibration identities overlap")
    if any(row.date > "2025-09-30" for row in training):
        raise ValueError("binary final training labels must end by 2025-09-30")
    if {row.month for row in calibration} != {"2025-10"}:
        raise ValueError("binary final probability calibration must use October 2025 only")
    if max(row.date for row in training) >= min(row.date for row in calibration):
        raise ValueError("binary final probability calibration must follow training")

    train_target = np.asarray(
        [int(partitions.train_labels[row.row_id] == "PROCESS") for row in training],
        dtype=np.int8,
    )
    calibration_target = np.asarray(
        [
            int(
                partitions.probability_calibration_labels[row.row_id]
                == "PROCESS"
            )
            for row in calibration
        ],
        dtype=np.int8,
    )
    if set(train_target.tolist()) != {0, 1}:
        raise ValueError("binary final training requires process and unsafe labels")
    if set(calibration_target.tolist()) != {0, 1}:
        raise ValueError("binary final calibration requires process and unsafe labels")

    model = BinaryProcessCandidate(
        customer_aware=template.customer_aware,
        random_state=template.random_state,
        max_iter=template.max_iter,
        score_batch_size=template.score_batch_size,
        max_score_rows=template.max_score_rows,
        validation_config=template.validation_config,
    )
    encoder = RawFeatureEncoder(
        model.feature_spec,
        validation_config=model.validation_config,
    )
    training_matrix = encoder.fit_transform(training)
    head = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        alpha=1e-5,
        max_iter=model.max_iter,
        tol=1e-4,
        random_state=model.random_state,
    )
    head.fit(training_matrix, train_target)
    calibration_matrix = encoder.transform(calibration)
    raw_scores = np.asarray(
        head.decision_function(calibration_matrix), dtype=np.float64
    )
    calibrator = LogisticRegression(
        max_iter=1_000,
        random_state=model.random_state + 100,
    )
    calibrator.fit(raw_scores.reshape(-1, 1), calibration_target)

    training_cutoff = max(row.date for row in training)
    calibration_cutoff = max(row.date for row in calibration)
    calibration_row_sha256 = _sha256_json(
        [
            {
                "rowId": row.row_id,
                "date": row.date,
                "actual": partitions.probability_calibration_labels[row.row_id],
            }
            for row in sorted(calibration, key=lambda row: row.row_id)
        ]
    )
    model_sha256 = hashlib.sha256(
        pickle.dumps(
            {
                "objective": OBJECTIVE,
                "sourceDatasetId": source_dataset_id,
                "historySha256": history_sha256,
                "customerAware": model.customer_aware,
                "encoderManifest": encoder.encoder_manifest,
                "head": head,
                "calibrator": calibrator,
                "trainingIds": tuple(sorted(training_ids)),
                "calibrationIds": tuple(sorted(calibration_ids)),
                "trainingCutoff": training_cutoff,
                "calibrationCutoff": calibration_cutoff,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).hexdigest()
    model._encoder = encoder
    model._head = head
    model._calibrator = calibrator
    model._training_ids = frozenset(training_ids)
    model._calibration_ids = frozenset(calibration_ids)
    model._training_cutoff = training_cutoff
    model._calibration_cutoff = calibration_cutoff
    model._calibration_row_sha256 = calibration_row_sha256
    model._source_dataset_id = source_dataset_id
    model._history_sha256 = history_sha256
    model._model_sha256 = model_sha256
    model._feature_schema_sha256 = None
    return model


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
