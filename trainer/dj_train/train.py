from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from .config import POLICY_OUTPUT, TOTAL_BARS
from .dataset import TransitionDataset, read_manifest, write_manifest_rows
from .models import (
    DifferentiableFeatureProjector,
    DifferentiableMixer,
    TransitionCritic,
    TransitionLocalizer,
    TransitionPolicy,
    export_policy,
)


def training_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"mps", "auto"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested not in {"auto", "cpu"}:
        raise RuntimeError(f"Requested training device {requested!r} is not available")
    return torch.device("cpu")


def collate_common_fields(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate heterogeneous domains without requiring domain-only arrays."""
    common = set(items[0]).intersection(*(set(item) for item in items[1:]))
    return default_collate([{key: item[key] for key in common} for item in items])


def _loader(dataset: TransitionDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    if len(dataset) == 0:
        raise ValueError("No matching transition samples exist in the dataset")
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_common_fields,
    )


def _tensors(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["mel"].to(device=device, dtype=torch.float32).contiguous(),
        batch["mir"].to(device=device, dtype=torch.float32).contiguous(),
        batch["bar_mask"].to(device=device, dtype=torch.float32).contiguous(),
    )


def _checkpoint_path(root: Path, name: str) -> Path:
    path = root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _report_epoch(stage: str, epoch: int, epochs: int, loss: float) -> None:
    print(
        f"Training {stage}: epoch {epoch}/{epochs}, loss={loss:.6f}",
        file=sys.stderr,
        flush=True,
    )


def apply_localizer(root: Path, model: TransitionLocalizer, device: torch.device) -> int:
    """Convert weak bags into persisted start/mid/end/confidence estimates."""
    rows = read_manifest(root)
    updated = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            if row["domain"] != "professional":
                continue
            path = root / row["sample"]
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
            mel = torch.from_numpy(arrays["mel"].astype(np.float32)).unsqueeze(0).to(device)
            mir = torch.from_numpy(arrays["mir"].astype(np.float32)).unsqueeze(0).to(device)
            probability = torch.sigmoid(model(mel, mir))[0].detach().cpu().numpy()
            weak = arrays["weak_positive_mask"].astype(np.float32)
            probability = probability * weak
            center = int(np.argmax(probability))
            peak = float(probability[center])
            threshold = max(0.20, peak * 0.55)
            start = center
            while start > 0 and center - start < 16 and probability[start - 1] >= threshold:
                start -= 1
            end = center
            while end + 1 < TOTAL_BARS and end - center < 16 and probability[end + 1] >= threshold:
                end += 1
            if end - start < 3:
                start, end = max(0, center - 2), min(TOTAL_BARS - 1, center + 2)
            arrays["transition_probability"] = probability.astype(np.float16)
            arrays["localized_transition"] = np.asarray([start, center, end, peak], dtype=np.float32)
            temporary = path.with_name(path.name + ".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(path)
            row["localizedStartBar"] = start
            row["localizedMidBar"] = center
            row["localizedEndBar"] = end
            row["localizerConfidence"] = round(peak, 5)
            updated += 1
    write_manifest_rows(root, rows)
    return updated


def train_localizer(
    root: Path,
    *,
    epochs: int = 20,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = training_device(device_name)
    dataset = TransitionDataset(root, {"professional", "negative"}, augment=True)
    domains = {row["domain"] for row in dataset.rows}
    if domains != {"professional", "negative"}:
        raise ValueError("Localizer training requires professional weak positives and far-away negatives")
    model = TransitionLocalizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    final_loss = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _loader(dataset, batch_size):
            mel, mir, bar_mask = _tensors(batch, device)
            logits = model(mel, mir).masked_fill(bar_mask <= 0, -1e4)
            bag_mask = batch["weak_positive_mask"].to(device=device, dtype=torch.float32)
            positive = torch.tensor(
                [domain == "professional" for domain in batch["domain"]],
                device=device, dtype=torch.float32,
            )
            # Multiple-instance learning: at least one bar inside a weak ±45s
            # bag should be a transition; negatives must stay quiet everywhere.
            candidate_mask = torch.where(positive[:, None] > 0, bag_mask, bar_mask)
            bag_logits = logits.masked_fill(candidate_mask <= 0, -1e4).amax(dim=1)
            mil_loss = F.binary_cross_entropy_with_logits(bag_logits, positive)
            smoothness = torch.sigmoid(logits).diff(dim=1).abs().mean()
            sparsity = torch.sigmoid(logits).mean()
            loss = mil_loss + 0.08 * smoothness + 0.025 * sparsity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        _report_epoch("localizer", epoch, epochs, final_loss)
    checkpoint = _checkpoint_path(root, "localizer.pt")
    torch.save({"model": model.state_dict(), "epochs": epochs, "loss": final_loss}, checkpoint)
    localized = apply_localizer(root, model, device)
    return {
        "stage": "localizer", "device": str(device), "epochs": epochs, "loss": final_loss,
        "localizedTransitions": localized, "checkpoint": str(checkpoint),
    }


def _critic_auxiliary_targets(mir: torch.Tensor) -> torch.Tensor:
    transition = mir[:, 16:48]
    energy_jump = transition[:, :, 0].diff(dim=1).abs().mean(dim=1)
    spectral_jump = transition[:, :, 6:9].diff(dim=1).abs().mean(dim=(1, 2))
    bass_spike = transition[:, :, 2].amax(dim=1) - transition[:, :, 2].median(dim=1).values
    flux = transition[:, :, 5]
    outside_flux = torch.cat((mir[:, :16, 5], mir[:, 48:, 5]), dim=1).mean(dim=1)
    inside_flux = flux.mean(dim=1)
    boundary_novelty = 0.5 * (mir[:, 16, 5] + mir[:, 47, 5])
    return torch.stack(
        (
            (1.0 - boundary_novelty).clamp(0, 1),
            (1.0 - energy_jump * 3.0).clamp(0, 1),
            (1.0 - bass_spike * 2.0).clamp(0, 1),
            (1.0 - spectral_jump * 4.0).clamp(0, 1),
            torch.ones_like(energy_jump),
            torch.sigmoid((inside_flux - outside_flux) * 5.0),
        ),
        dim=1,
    )


def train_critic(
    root: Path,
    *,
    epochs: int = 24,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = training_device(device_name)
    dataset = TransitionDataset(root, {"professional", "synthetic"}, augment=True)
    domains = {row["domain"] for row in dataset.rows}
    if domains != {"professional", "synthetic"}:
        raise ValueError("Critic training requires both professional and synthetic transitions")
    model = TransitionCritic().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=2e-3)
    final_loss = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _loader(dataset, batch_size):
            mel, mir, bar_mask = _tensors(batch, device)
            output = model(mel, mir, bar_mask)
            professional = torch.tensor(
                [domain == "professional" for domain in batch["domain"]],
                device=device, dtype=torch.float32,
            )
            realism_loss = F.binary_cross_entropy_with_logits(output["realism"], professional)
            auxiliary_targets = _critic_auxiliary_targets(mir)
            auxiliary_loss = F.binary_cross_entropy_with_logits(output["auxiliary"], auxiliary_targets)
            location_target = batch["transition_target"].to(device=device, dtype=torch.float32)
            location_loss = F.binary_cross_entropy_with_logits(output["location"], location_target)
            loss = realism_loss + 0.40 * auxiliary_loss + 0.20 * location_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        _report_epoch("critic", epoch, epochs, final_loss)
    checkpoint = _checkpoint_path(root, "critic.pt")
    torch.save({"model": model.state_dict(), "epochs": epochs, "loss": final_loss}, checkpoint)
    return {"stage": "critic", "device": str(device), "epochs": epochs, "loss": final_loss, "checkpoint": str(checkpoint)}


def _load_critic(root: Path, device: torch.device) -> TransitionCritic:
    checkpoint_path = _checkpoint_path(root, "critic.pt")
    if not checkpoint_path.exists():
        raise ValueError("Train the critic before optimizing the transition policy")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    critic = TransitionCritic().to(device)
    critic.load_state_dict(checkpoint["model"])
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return critic


def train_policy(
    root: Path,
    *,
    epochs: int = 30,
    batch_size: int = 2,
    device_name: str = "auto",
    policy_output: Path = POLICY_OUTPUT,
) -> dict[str, Any]:
    device = training_device(device_name)
    dataset = TransitionDataset(root, {"synthetic"}, augment=False)
    if not dataset.rows:
        raise ValueError("Generate synthetic deck pairs before optimizing the policy")
    critic = _load_critic(root, device)
    policy = TransitionPolicy().to(device)
    mixer = DifferentiableMixer().to(device)
    projector = DifferentiableFeatureProjector().to(device)
    all_features = torch.stack(
        [torch.from_numpy(dataset[index]["policy_features"]) for index in range(len(dataset))]
    ).to(device)
    policy.set_feature_statistics(all_features)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=5e-4)
    final_loss = math.inf
    for epoch in range(1, epochs + 1):
        policy.train()
        losses: list[float] = []
        for batch in _loader(dataset, batch_size):
            state = batch["policy_features"].to(device=device, dtype=torch.float32)
            outgoing = batch["outgoing_audio"].to(device=device, dtype=torch.float32)
            incoming = batch["incoming_audio"].to(device=device, dtype=torch.float32)
            controls = policy(state)
            rendered = mixer(outgoing, incoming, controls)
            mel, mir = projector(rendered)
            output = critic(mel, mir, torch.ones(rendered.shape[0], TOTAL_BARS, device=device))
            adversarial = F.softplus(-output["realism"]).mean()
            diagnostics = F.binary_cross_entropy_with_logits(
                output["auxiliary"], torch.ones_like(output["auxiliary"]),
            )
            peak_penalty = F.relu(rendered.abs().amax(dim=(1, 2)) - 0.98).mean()
            # Keep the relaxed phrase count near a legal 8-bar multiple.  The
            # live planner quantizes it and re-runs its hard residual gates.
            phrase_integer = torch.sin(controls[:, 0] * torch.pi).square().mean()
            bass_collision = (
                torch.sigmoid((controls[:, 1] - 0.72) * 10)
                + torch.sigmoid((0.42 - controls[:, 1]) * 10)
            ).mean()
            loop_cost = controls[:, 10].mean() * 0.02
            timing_calibration = F.mse_loss(controls[:, 11], torch.sigmoid(output["realism"]).detach())
            baseline = torch.tensor(
                [1.0, 0.56, 1.0, 1.0, -18.0, -18.0, -3.0, -3.0, -5.0, -5.0, 0.05, 0.5],
                device=device,
            )
            control_range = (policy.control_upper - policy.control_lower).clamp_min(1e-4)
            prior_cost = ((controls - baseline) / control_range).square().mean()
            loss = (
                adversarial + 0.35 * diagnostics + 2.0 * peak_penalty
                + 0.08 * phrase_integer + 0.05 * bass_collision + loop_cost
                + 0.12 * prior_cost + 0.15 * timing_calibration
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in policy.parameters()
            ):
                raise FloatingPointError("Policy optimization produced a non-finite gradient; no policy was exported")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        _report_epoch("policy", epoch, epochs, final_loss)
    checkpoint = _checkpoint_path(root, "policy.pt")
    torch.save({"model": policy.state_dict(), "epochs": epochs, "loss": final_loss}, checkpoint)
    export_policy(
        policy,
        policy_output,
        {
            "policyVersion": "transition-policy-v1",
            "epochs": epochs,
            "loss": final_loss,
            "professionalSamples": sum(row["domain"] == "professional" for row in read_rows(root)),
            "syntheticSamples": len(dataset),
        },
    )
    return {
        "stage": "policy", "device": str(device), "epochs": epochs, "loss": final_loss,
        "checkpoint": str(checkpoint), "export": str(policy_output),
    }


def read_rows(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        return []
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def tune_from_human_feedback(
    root: Path,
    feedback_path: Path,
    *,
    epochs: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Calibrate critic realism with 1–5 ratings before another policy stage."""
    device = training_device(device_name)
    feedback = {
        row["sampleId"]: (float(row["rating"]) - 1.0) / 4.0
        for line in feedback_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (row := json.loads(line))
    }
    dataset = TransitionDataset(root, {"synthetic"}, augment=True)
    selected = [(index, feedback[row["id"]]) for index, row in enumerate(dataset.rows) if row["id"] in feedback]
    if not selected:
        raise ValueError("Feedback contains no sampleId present in the synthetic dataset")
    critic = _load_critic(root, device)
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=5e-5, weight_decay=2e-3)
    final_loss = math.inf
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for index, rating in selected:
            item = dataset[index]
            mel = torch.from_numpy(item["mel"]).unsqueeze(0).to(device)
            mir = torch.from_numpy(item["mir"]).unsqueeze(0).to(device)
            mask = torch.from_numpy(item["bar_mask"]).unsqueeze(0).to(device)
            realism = critic(mel, mir, mask)["realism"]
            target = torch.tensor([rating], device=device)
            loss = F.binary_cross_entropy_with_logits(realism, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
        _report_epoch("human-feedback", epoch, epochs, final_loss)
    checkpoint = _checkpoint_path(root, "critic.pt")
    torch.save({"model": critic.state_dict(), "epochs": epochs, "loss": final_loss, "humanCalibrated": True}, checkpoint)
    return {"stage": "human-feedback", "device": str(device), "ratings": len(selected), "loss": final_loss, "checkpoint": str(checkpoint)}
