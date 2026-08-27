from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import (
    CONTEXT_BEFORE_BARS,
    FEATURE_SAMPLE_RATE,
    FEATURE_VERSION,
    MEL_BINS,
    MIR_DIM,
    POLICY_CONTROL_NAMES,
    POLICY_FEATURE_NAMES,
    SAMPLES_PER_BAR,
    TOTAL_BARS,
    TRANSITION_BARS,
)


class TemporalContextBlock(nn.Module):
    """Five-bar context projection without MPS Conv1d backward."""

    def __init__(self, inputs: int, outputs: int, radius: int = 2):
        super().__init__()
        self.radius = radius
        self.projection = nn.Linear(inputs * (radius * 2 + 1), outputs)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        length = sequence.shape[1]
        padded = F.pad(sequence, (0, 0, self.radius, self.radius))
        neighbours = torch.cat(
            [padded[:, offset:offset + length] for offset in range(self.radius * 2 + 1)],
            dim=-1,
        ).contiguous()
        return F.gelu(self.projection(neighbours))


class FeatureEncoder(nn.Module):
    def __init__(self, width: int = 96):
        super().__init__()
        self.mel_projection = nn.Sequential(nn.Linear(MEL_BINS, 56), nn.LayerNorm(56), nn.GELU())
        self.mir_projection = nn.Sequential(nn.Linear(MIR_DIM, 40), nn.LayerNorm(40), nn.GELU())
        self.temporal = nn.Sequential(
            TemporalContextBlock(96, width),
            TemporalContextBlock(width, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 3,
            dropout=0.10, batch_first=True, norm_first=True,
        )
        # The sequence length is fixed, so nested tensors provide no benefit.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False,
        )

    def forward(self, mel: torch.Tensor, mir: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat((self.mel_projection(mel), self.mir_projection(mir)), dim=-1)
        encoded = self.temporal(encoded.contiguous())
        return self.transformer(encoded)


class TransitionLocalizer(nn.Module):
    """Per-bar localizer trained with multiple-instance weak timestamp bags."""

    def __init__(self, width: int = 96):
        super().__init__()
        self.encoder = FeatureEncoder(width)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, mel: torch.Tensor, mir: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(mel, mir)).squeeze(-1)


class TransitionCritic(nn.Module):
    """Context-aware professional/artificial critic with diagnostic heads."""

    AUXILIARY_NAMES = [
        "phrase_alignment", "energy_smoothness", "bass_separation",
        "spectral_smoothness", "duration_fit", "transition_strength",
    ]

    def __init__(self, width: int = 96):
        super().__init__()
        self.encoder = FeatureEncoder(width)
        self.attention = nn.Linear(width, 1)
        self.realism = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.auxiliary = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(self.AUXILIARY_NAMES)))
        self.location = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(
        self, mel: torch.Tensor, mir: torch.Tensor, bar_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(mel, mir)
        attention_logits = self.attention(encoded).squeeze(-1)
        if bar_mask is not None:
            attention_logits = attention_logits.masked_fill(bar_mask <= 0, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return {
            "realism": self.realism(pooled).squeeze(-1),
            "auxiliary": self.auxiliary(pooled),
            "location": self.location(encoded).squeeze(-1),
        }


CONTROL_LOWER = torch.tensor([1.0, 0.35, 0.50, 0.50, -24.0, -24.0, -9.0, -9.0, -12.0, -12.0, 0.0, 0.0])
CONTROL_UPPER = torch.tensor([4.0, 0.80, 1.80, 1.80, -6.0, -6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])


class TransitionPolicy(nn.Module):
    """Compact runtime policy; hard musical timing constraints stay external."""

    def __init__(self, hidden: int = 48):
        super().__init__()
        inputs = len(POLICY_FEATURE_NAMES)
        outputs = len(POLICY_CONTROL_NAMES)
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, outputs)
        self.register_buffer("feature_mean", torch.zeros(inputs))
        self.register_buffer("feature_scale", torch.ones(inputs))
        self.register_buffer("control_lower", CONTROL_LOWER.clone())
        self.register_buffer("control_upper", CONTROL_UPPER.clone())
        # Start from the deterministic runtime controller. Adversarial training
        # must earn every deviation instead of beginning with random extreme EQ.
        initial = torch.tensor([1.0, 0.56, 1.0, 1.0, -18.0, -18.0, -3.0, -3.0, -5.0, -5.0, 0.05, 0.5])
        unit = ((initial - CONTROL_LOWER) / (CONTROL_UPPER - CONTROL_LOWER)).clamp(1e-4, 1 - 1e-4)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.logit(unit))

    def set_feature_statistics(self, features: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalised = ((features - self.feature_mean) / self.feature_scale).clamp(-5.0, 5.0)
        hidden = F.gelu(self.fc1(normalised), approximate="tanh")
        hidden = F.gelu(self.fc2(hidden), approximate="tanh")
        unit = torch.sigmoid(self.output(hidden))
        return self.control_lower + unit * (self.control_upper - self.control_lower)


class DifferentiableMixer(nn.Module):
    """Beat-warped three-band mixer used only while optimizing the policy."""

    def __init__(self):
        super().__init__()
        frequencies = torch.fft.rfftfreq(SAMPLES_PER_BAR, d=1.0 / FEATURE_SAMPLE_RATE)
        self.register_buffer("low_mask", (frequencies < 250).float())
        self.register_buffer("mid_mask", ((frequencies >= 250) & (frequencies < 4_000)).float())
        self.register_buffer("high_mask", (frequencies >= 4_000).float())
        bar_positions = torch.arange(TOTAL_BARS).float() + 0.5
        self.register_buffer("bar_positions", bar_positions)

    def _bands(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(audio, dim=-1)
        return tuple(
            torch.fft.irfft(spectrum * mask, n=SAMPLES_PER_BAR, dim=-1)
            for mask in (self.low_mask, self.mid_mask, self.high_mask)
        )  # type: ignore[return-value]

    @staticmethod
    def _db_gain(value: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, value / 20.0)

    def forward(self, outgoing: torch.Tensor, incoming: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if outgoing.shape[-2:] != (TOTAL_BARS, SAMPLES_PER_BAR) or outgoing.shape != incoming.shape:
            raise ValueError("Mixer inputs must be [batch, 64 bars, 4096 samples]")
        overlap_phrases = controls[:, 0:1]
        overlap_bars = overlap_phrases * 8.0
        transition_end = float(CONTEXT_BEFORE_BARS + TRANSITION_BARS)
        start = transition_end - overlap_bars
        progress = ((self.bar_positions.unsqueeze(0) - start) / overlap_bars.clamp_min(1.0)).clamp(0.0, 1.0)
        active = torch.sigmoid((self.bar_positions.unsqueeze(0) - start) * 3.0)
        progress = progress * active

        fade_out_curve = controls[:, 2:3].clamp(0.3, 2.5)
        fade_in_curve = controls[:, 3:4].clamp(0.3, 2.5)
        differentiable_progress = progress.clamp(1e-4, 1.0)
        outgoing_gain = torch.cos(differentiable_progress.pow(fade_out_curve) * torch.pi / 2)
        incoming_gain = torch.sin(differentiable_progress.pow(fade_in_curve) * torch.pi / 2)

        swap = controls[:, 1:2]
        bass_progress = torch.sigmoid((progress - swap) * 18.0)
        out_low_db = controls[:, 4:5] * bass_progress
        in_low_db = controls[:, 5:6] * (1.0 - bass_progress)
        out_mid_db = controls[:, 6:7] * progress
        in_mid_db = controls[:, 7:8] * (1.0 - progress)
        out_high_db = controls[:, 8:9] * progress
        in_high_db = controls[:, 9:10] * (1.0 - progress)

        out_low, out_mid, out_high = self._bands(outgoing)
        in_low, in_mid, in_high = self._bands(incoming)
        shaped_out = (
            out_low * self._db_gain(out_low_db).unsqueeze(-1)
            + out_mid * self._db_gain(out_mid_db).unsqueeze(-1)
            + out_high * self._db_gain(out_high_db).unsqueeze(-1)
        )
        shaped_in = (
            in_low * self._db_gain(in_low_db).unsqueeze(-1)
            + in_mid * self._db_gain(in_mid_db).unsqueeze(-1)
            + in_high * self._db_gain(in_high_db).unsqueeze(-1)
        )

        # Softly choose a four-bar outgoing loop candidate.  Later RL/Gumbel
        # stages can choose among multiple discrete loops without making the
        # basic adversarial mixer non-differentiable.
        looped = outgoing.clone()
        loop_source = outgoing[:, CONTEXT_BEFORE_BARS - 4:CONTEXT_BEFORE_BARS]
        looped[:, CONTEXT_BEFORE_BARS:CONTEXT_BEFORE_BARS + TRANSITION_BARS] = loop_source.repeat(1, TRANSITION_BARS // 4, 1)
        loop_probability = controls[:, 10:11].unsqueeze(-1)
        shaped_out = shaped_out * (1.0 - loop_probability) + self._bands(looped)[0] * loop_probability * 0.25 + shaped_out * loop_probability * 0.75

        mixed = shaped_out * outgoing_gain.unsqueeze(-1) + shaped_in * incoming_gain.unsqueeze(-1)
        return torch.tanh(mixed)


class DifferentiableFeatureProjector(nn.Module):
    """Approximate the saved canonical features while retaining gradients."""

    def __init__(self):
        super().__init__()
        frequencies = torch.fft.rfftfreq(SAMPLES_PER_BAR, d=1.0 / FEATURE_SAMPLE_RATE)
        mel = librosa.filters.mel(
            sr=FEATURE_SAMPLE_RATE, n_fft=SAMPLES_PER_BAR, n_mels=MEL_BINS,
            fmax=FEATURE_SAMPLE_RATE / 2,
        ).astype(np.float32)
        chroma = librosa.filters.chroma(
            sr=FEATURE_SAMPLE_RATE, n_fft=SAMPLES_PER_BAR, n_chroma=12,
        ).astype(np.float32)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("mel_filter", torch.from_numpy(mel))
        self.register_buffer("chroma_filter", torch.from_numpy(chroma))

    def forward(self, bars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        complex_spectrum = torch.fft.rfft(bars, dim=-1)
        spectrum = torch.sqrt(complex_spectrum.real.square() + complex_spectrum.imag.square() + 1e-8)
        power = spectrum.square() + 1e-8
        mel_power = torch.einsum("mf,btf->btm", self.mel_filter, power)
        mel_db = 10.0 * torch.log10(mel_power.clamp_min(1e-8))
        mel = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
        total = power.sum(dim=-1).clamp_min(1e-8)
        low = power[..., self.frequencies < 250].sum(dim=-1) / total
        mid = power[..., (self.frequencies >= 250) & (self.frequencies < 4_000)].sum(dim=-1) / total
        high = power[..., self.frequencies >= 4_000].sum(dim=-1) / total
        energy = torch.sqrt(bars.square().mean(dim=-1) + 1e-8).mul(4.0).clamp(max=1.0)
        energy_slope = F.pad((energy[:, 1:] - energy[:, :-1]) * 8.0, (1, 0)).clamp(-1.0, 1.0)
        normalised_spectrum = spectrum / spectrum.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        flux = F.pad((normalised_spectrum[:, 1:] - normalised_spectrum[:, :-1]).relu().norm(dim=-1), (1, 0)).clamp(max=1.0)
        nyquist = FEATURE_SAMPLE_RATE / 2
        centroid = (power * self.frequencies).sum(dim=-1) / total / nyquist
        variance = (power * (self.frequencies - centroid.unsqueeze(-1) * nyquist).square()).sum(dim=-1) / total
        bandwidth = torch.sqrt(variance.clamp_min(1e-8)) / nyquist
        cumulative = torch.cumsum(power, dim=-1) / total.unsqueeze(-1)
        # A smooth rolloff approximation avoids a non-differentiable argmax.
        rolloff_weights = torch.softmax(-(cumulative - 0.85).abs() * 20.0, dim=-1)
        rolloff = (rolloff_weights * self.frequencies).sum(dim=-1) / nyquist
        zcr_proxy = (bars[..., 1:] - bars[..., :-1]).abs().mean(dim=-1).mul(2.0).clamp(max=1.0)
        onset_proxy = flux
        vocal = (mid * (1.0 - low) * 1.7).clamp(0.0, 1.0)
        chroma = torch.einsum("cf,btf->btc", self.chroma_filter, power)
        chroma = chroma / chroma.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        mir = torch.cat(
            (
                energy.unsqueeze(-1), energy_slope.unsqueeze(-1), low.unsqueeze(-1), mid.unsqueeze(-1),
                high.unsqueeze(-1), flux.unsqueeze(-1), centroid.unsqueeze(-1), bandwidth.unsqueeze(-1),
                rolloff.unsqueeze(-1), zcr_proxy.unsqueeze(-1), onset_proxy.unsqueeze(-1),
                vocal.unsqueeze(-1), chroma,
            ),
            dim=-1,
        )
        return mel, mir


def export_policy(policy: TransitionPolicy, destination: Path, metadata: dict[str, Any] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = policy.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Refusing to export a transition policy with non-finite weights")
    payload = {
        "schemaVersion": 1,
        "featureVersion": FEATURE_VERSION,
        "policyVersion": (metadata or {}).get("policyVersion", "transition-policy-v1"),
        "featureNames": POLICY_FEATURE_NAMES,
        "controlNames": POLICY_CONTROL_NAMES,
        "activation": "gelu-tanh-approx",
        "featureMean": state["feature_mean"].detach().cpu().tolist(),
        "featureScale": state["feature_scale"].detach().cpu().tolist(),
        "controlLower": state["control_lower"].detach().cpu().tolist(),
        "controlUpper": state["control_upper"].detach().cpu().tolist(),
        "layers": [
            {"weight": state["fc1.weight"].detach().cpu().tolist(), "bias": state["fc1.bias"].detach().cpu().tolist()},
            {"weight": state["fc2.weight"].detach().cpu().tolist(), "bias": state["fc2.bias"].detach().cpu().tolist()},
            {"weight": state["output.weight"].detach().cpu().tolist(), "bias": state["output.bias"].detach().cpu().tolist()},
        ],
        "training": metadata or {},
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    return destination
