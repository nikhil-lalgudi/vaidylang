"""Capture and compare next-token logits from identical DeepSeek prefixes."""

from contextlib import contextmanager
import hashlib
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from harness import digest


def summarize_logits(logits, vocab_size, prefix_sha256):
    logits = np.asarray(logits, dtype="<f4")
    if logits.shape != (vocab_size,) or not np.isfinite(logits).all():
        raise ValueError("Unexpected shape or nonfinite logits")
    winners = np.argsort(-logits, kind="stable")[:5]
    return logits, {
        "prefix_sha256": prefix_sha256,
        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "top_logits": [[int(i), float(logits[i])] for i in winners],
    }


@contextmanager
def capture_logit_archive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            yield archive
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_member(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest() + ".npy"


def write_logit_capture(archive, case_id, vectors):
    with archive.open(_capture_member(case_id), "w") as file:
        np.save(file, np.asarray(vectors, dtype="<f4"), allow_pickle=False)


def read_logit_capture(archive, reference, vocab_size):
    with archive.open(_capture_member(reference["id"])) as file:
        vectors = np.load(file, allow_pickle=False)
    if vectors.dtype.str != "<f4" or vectors.shape != (len(reference["logit_steps"]), vocab_size):
        raise ValueError("Reference logit capture has an incompatible dtype or shape")
    for vector, step in zip(vectors, reference["logit_steps"]):
        if hashlib.sha256(vector.tobytes()).hexdigest() != step["logits_sha256"]:
            raise ValueError("Reference logit capture does not match the report digest")
    return vectors


def _validate_replay(compiled, reference, stop_ids, vocab_size):
    prompt = compiled.record()
    for field in ("id", "prompt_ids", "prompt_sha256", "token_types", "images", "new_tokens"):
        if prompt.get(field) != reference.get(field):
            raise ValueError(f"Reference {field} differs from the compiled case")
    tokens, steps = reference["output_ids"], reference["logit_steps"]
    if (not 0 < len(tokens) == len(steps) <= compiled.new_tokens
            or any(type(token) is not int or not 0 <= token < vocab_size for token in tokens)
            or any(token in stop_ids for token in tokens[:-1])
            or (len(tokens) < compiled.new_tokens and tokens[-1] not in stop_ids)):
        raise ValueError("Reference continuation or EOS is inconsistent with the token budget")
    prefix_hash = hashlib.sha256(np.asarray(compiled.ids, dtype="<i8").tobytes())
    for token, step in zip(tokens, steps):
        if step["prefix_sha256"] != prefix_hash.hexdigest():
            raise ValueError("Reference prefix digest does not match its continuation IDs")
        if step["top_logits"][0][0] != token:
            raise ValueError("Reference continuation does not follow the greedy logits")
        prefix_hash.update(np.asarray([token], dtype="<i8").tobytes())


def _compare_step(candidate, candidate_step, reference_step, reference_id, vector, token_label, atol, rtol):
    candidate_id = candidate_step["top_logits"][0][0]
    reference_top = dict(reference_step["top_logits"])
    competing_id = candidate_id
    if candidate_id == reference_id:
        competing_id = next((token for token in reference_top if token != reference_id), None)
    step = {
        "candidate_token_id": candidate_id, "reference_token_id": reference_id,
        "candidate_token": token_label(candidate_id), "reference_token": token_label(reference_id),
        "competing_token_id": competing_id, "reference_top_logits": reference_step["top_logits"],
        "reference_logits_sha256": reference_step["logits_sha256"],
        "logit_digest_match": candidate_step["logits_sha256"] == reference_step["logits_sha256"],
    }
    if competing_id is not None:
        reference_score = (reference_top.get(competing_id) if vector is None
                           else float(vector[competing_id]))
        reference_winner = reference_top[reference_id]
        candidate_winner, candidate_score = float(candidate[reference_id]), float(candidate[competing_id])
        step.update(
            competing_token=token_label(competing_id),
            reference_scores=[reference_winner, reference_score],
            candidate_scores=[candidate_winner, candidate_score],
            reference_margin=None if reference_score is None else reference_winner - reference_score,
            candidate_margin=candidate_winner - candidate_score,
        )
    if vector is not None:
        error = np.abs(candidate.astype(np.float64) - vector)
        index = int(error.argmax())
        step.update(
            max_abs_error=float(error[index]), max_error_token_id=index,
            max_error_scores=[float(vector[index]), float(candidate[index])],
            outside_tolerance=int(np.count_nonzero(error > atol + rtol * np.abs(vector.astype(np.float64)))),
        )
    return step


def replay_case(engine, compiled, stop_ids, reference, *, vectors=None, archive=None, atol=0, rtol=0):
    """Score the candidate on the prompt followed by the reference's greedy choices."""
    vocab = engine.config.vocab_size
    _validate_replay(compiled, reference, stop_ids, vocab)
    tokens, reference_steps = reference["output_ids"], reference["logit_steps"]
    capacity = len(compiled.ids) + len(tokens) - 1
    if compiled.image_prompt:
        logits, state = engine.prefill(compiled.ids, capacity, images=compiled.image_prompt.images)
    else:
        logits, state = engine.prefill(compiled.ids, capacity)
    steps, predictions, captured = [], [], []
    token_label = compiled.tokenizer.backend.id_to_token
    for index, (reference_id, reference_step) in enumerate(zip(tokens, reference_steps)):
        values, record = summarize_logits(logits, vocab, reference_step["prefix_sha256"])
        step = _compare_step(values, record, reference_step, reference_id,
                             None if vectors is None else vectors[index], token_label, atol, rtol)
        steps.append(dict(record, step=index, prefix_length=len(compiled.ids) + index, **step))
        predictions.append(step["candidate_token_id"])
        if archive is not None:
            captured.append(values.copy())
        if index + 1 < len(tokens):
            logits = engine.decode(reference_id, state)
    if archive is not None:
        write_logit_capture(archive, compiled.id, captured)
    return {
        **compiled.record(), "reference_output_ids": tokens,
        "reference_output_sha256": digest(tokens),
        "candidate_ids_on_reference_prefix": predictions, "logit_steps": steps,
        "first_logit_difference": next((s["step"] for s in steps if not s["logit_digest_match"]), None),
        "first_token_difference": next((i for i, (a, b) in enumerate(zip(tokens, predictions)) if a != b), None),
        "first_outside_tolerance": next((s["step"] for s in steps if s.get("outside_tolerance")), None),
    }
