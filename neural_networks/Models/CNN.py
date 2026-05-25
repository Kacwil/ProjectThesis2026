import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from .utils import *

class CNNDataset(Dataset):
    def __init__(self, iq, angles_hat, positions, angles, train, seq_len=2, noise_mag=1.0, shuffle=True):
        
        if iq.shape[1] == 82:
            iq = average_iq_per_antenna(iq)  # (N, 82, 2) -> (N, 12, 2)

        self.iq = iq
        self.seq_len = seq_len

        iq_correlation = compute_cov_matrices(iq, seq_len)  # (N-T+1, 12, 12)
        I_correlation = torch.as_tensor(iq_correlation.real, dtype=torch.float32)
        Q_correlation = torch.as_tensor(iq_correlation.imag, dtype=torch.float32)
        phase = torch.as_tensor(np.angle(iq_correlation), dtype=torch.float32)
        self.iq_corr = torch.stack([I_correlation,Q_correlation,phase,], dim=1)  # (N-T+1, 3, 12, 12)

        angles_hat = torch.as_tensor(angles_hat, dtype=torch.float32)
        angles = torch.as_tensor(angles, dtype=torch.float32)

        noise_mag = 0.0 if train else noise_mag
        noise = np.random.uniform(-noise_mag, noise_mag, positions.shape)
        positions = torch.as_tensor(positions + noise, dtype=torch.float32)

        self.static = positions
        self.residuals = angles_hat
        self.targets = angles

        offset = seq_len - 1
        self.residuals = angles_hat[offset:]
        self.targets = angles[offset:]

    
    def concat(self, other, shuffle=True, seed=1111):
        """
        Concatenate another initialized CNNDataset into this one.
        """

        if not isinstance(other, CNNDataset):
            raise TypeError(f"Expected CNNDataset, got {type(other)}")

        if self.seq_len != other.seq_len:
            raise ValueError(
                f"Cannot concatenate datasets with different seq_len: "
                f"{self.seq_len} != {other.seq_len}"
            )

        if self.iq_corr.shape[1:] != other.iq_corr.shape[1:]:
            raise ValueError(
                f"corr_matrices shape mismatch: "
                f"{self.iq_corr.shape[1:]} != {other.iq_corr.shape[1:]}"
            )

        if self.residuals.shape[1:] != other.residuals.shape[1:]:
            raise ValueError(
                f"angles_hat shape mismatch: "
                f"{self.residuals.shape[1:]} != {other.residuals.shape[1:]}"
            )

        if self.targets.shape[1:] != other.targets.shape[1:]:
            raise ValueError(
                f"targets shape mismatch: "
                f"{self.targets.shape[1:]} != {other.targets.shape[1:]}"
            )

        self.iq_corr = torch.cat([self.iq_corr, other.iq_corr], dim=0)
        self.residuals = torch.cat([self.residuals, other.residuals], dim=0)
        self.targets = torch.cat([self.targets, other.targets], dim=0)

        return self


    def __len__(self):
        return self.iq_corr.shape[0]

    def __getitem__(self, idx):

        return {
            "features": {
                "corr_matrices": self.iq_corr[idx],
                "angles_hat": self.residuals[idx],
            },

            "targets": self.targets[idx],
            "target_idx": idx,
        }
    
class ElevationGate(nn.Module):
    def __init__(self, center_deg=15.0, sharpness=0.3):
        super().__init__()
        self.center = nn.Parameter(torch.tensor(np.deg2rad(center_deg)))
        self.sharpness = nn.Parameter(torch.tensor(sharpness))

    def forward(self, el_hat):
        # high gate at low elevation, low gate at high elevation
        return torch.sigmoid(self.sharpness * (self.center - el_hat))

class CNN(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()

        a = 8
        n_filters = 256//a

        # Input: (B, 3, 12, 12)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.GELU(),

            nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
            nn.BatchNorm2d(n_filters),
            nn.GELU(),

            nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
            nn.BatchNorm2d(n_filters,),
            nn.GELU(),

            nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
            nn.BatchNorm2d(n_filters),
            nn.GELU(),
        )

        aux_dim = 2

        if a == 1:
            filter_dim = 20736 
        elif a == 4:
            filter_dim = 5184
        elif a == 8:
            filter_dim = 2592
        else:
            raise ValueError(f"Unknown hardcoded filter_dim for model scaling factor {a}")


        self.head = nn.Sequential(
            nn.Linear(filter_dim + aux_dim, 4096//a),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(4096//a, 2048//a),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(2048//a, 1024//a),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024//a, 2),
        )

        self.elevation_gate = nn.Sequential(nn.Linear(1, 64),nn.GELU(),nn.Linear(64, 1), nn.Sigmoid())

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()
        return self

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.train()
        return self

    def forward(self, batch):

        corr = batch["corr_matrices"]
        angles_hat = batch["angles_hat"]

        z = self.cnn(corr) # (B, 256//a, 9, 9)
        z = z.flatten(start_dim=1) # (B, filter_dim)
        z = torch.cat([z, angles_hat], dim=1)

        delta = self.head(z)
        delta_az = delta[:, 0:1]
        delta_el = delta[:, 1:2]

        # Apply elevation gate
        elevation_scale = self.elevation_gate(angles_hat[:, 1:2])
        delta_el = elevation_scale * delta_el

        delta_gated = torch.cat([delta_az, delta_el], dim=1)

        return {
            "ang_hat": angles_hat + delta_gated,
            "gate": elevation_scale,
            "delta": delta_gated
        }
    
class DualExpertCNN(nn.Module):
    def __init__(self, out_dim=2, dropout=0.1):
        super().__init__()

        a = 8
        n_filters = 256 // a
        aux_dim = 2

        def make_cnn():
            return nn.Sequential(
                nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),
            )

        if a == 1:
            filter_dim = 20736
        elif a == 4:
            filter_dim = 5184
        elif a == 8:
            filter_dim = 2592
        else:
            raise ValueError(f"Unknown hardcoded filter_dim for model scaling factor {a}")

        in_dim = filter_dim + aux_dim

        def make_head():
            return nn.Sequential(
                nn.Linear(in_dim, 4096 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(4096 // a, 2048 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(2048 // a, 1024 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(1024 // a, out_dim),
            )

        self.low_el_cnn = make_cnn()
        self.high_el_cnn = make_cnn()

        self.low_el_expert = make_head()
        self.high_el_expert = make_head()

        self.el_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 2),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        corr = batch["corr_matrices"]      # (B, 3, 12, 12)
        angles_hat = batch["angles_hat"]   # (B, 2)

        low_z = self.low_el_cnn(corr)
        low_z = low_z.flatten(start_dim=1)
        low_z = torch.cat([low_z, angles_hat], dim=1)

        high_z = self.high_el_cnn(corr)
        high_z = high_z.flatten(start_dim=1)
        high_z = torch.cat([high_z, angles_hat], dim=1)

        low_delta = self.low_el_expert(low_z)      # (B, 2)
        high_delta = self.high_el_expert(high_z)   # (B, 2)

        gates = self.el_gate(angles_hat[:, 1:2])  # (B, 2)

        low_w = gates[:, 0:1]
        high_w = gates[:, 1:2]

        delta = low_w * low_delta + high_w * high_delta

        delta = low_w * low_delta + high_w * high_delta
        gates = torch.cat([low_w, high_w], dim=1)  # (B, 2)

        return {
            "ang_hat": angles_hat + delta,
            "delta": delta,
            "low_delta": low_delta,
            "high_delta": high_delta,
            "gate": gates,
            "gate_names": ["Low expert", "High expert"],
            "deltas": [low_delta, high_delta],
        }
    
class TrioExpertCNN(nn.Module):
    def __init__(self, out_dim=2, dropout=0.1):
        super().__init__()

        a = 8
        n_filters = 256 // a
        aux_dim = 2

        def make_cnn():
            return nn.Sequential(
                nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=0),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),
            )

        if a == 1:
            filter_dim = 20736
        elif a == 4:
            filter_dim = 5184
        elif a == 8:
            filter_dim = 2592
        else:
            raise ValueError(f"Unknown hardcoded filter_dim for model scaling factor {a}")

        in_dim = filter_dim + aux_dim

        def make_head():
            return nn.Sequential(
                nn.Linear(in_dim, 4096 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(4096 // a, 2048 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(2048 // a, 1024 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(1024 // a, out_dim),
            )

        self.low_el_cnn = make_cnn()
        self.mid_el_cnn = make_cnn()
        self.high_el_cnn = make_cnn()

        self.low_el_expert = make_head()
        self.mid_el_expert = make_head()
        self.high_el_expert = make_head()

        self.el_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        corr = batch["corr_matrices"]      # (B, 3, 12, 12)
        angles_hat = batch["angles_hat"]   # (B, 2)

        low_z = self.low_el_cnn(corr).flatten(start_dim=1)
        mid_z = self.mid_el_cnn(corr).flatten(start_dim=1)
        high_z = self.high_el_cnn(corr).flatten(start_dim=1)

        low_z = torch.cat([low_z, angles_hat], dim=1)
        mid_z = torch.cat([mid_z, angles_hat], dim=1)
        high_z = torch.cat([high_z, angles_hat], dim=1)

        low_delta = self.low_el_expert(low_z)       # (B, 2)
        mid_delta = self.mid_el_expert(mid_z)       # (B, 2)
        high_delta = self.high_el_expert(high_z)    # (B, 2)

        gates = self.el_gate(angles_hat[:, 1:2])  # (B, 3)

        low_w = gates[:, 0:1]
        mid_w = gates[:, 1:2]
        high_w = gates[:, 2:3]

        delta = (
            low_w * low_delta +
            mid_w * mid_delta +
            high_w * high_delta
        )

        return {
            "ang_hat": angles_hat + delta,
            "delta": delta,
            "low_delta": low_delta,
            "mid_delta": mid_delta,
            "high_delta": high_delta,
            "gate": gates,  # (B, 3)
            "gate_names": ["Low expert", "Mid expert", "High expert"],
            "deltas": [low_delta, mid_delta, high_delta],
        }
    

class ExpertCNN(nn.Module):
    def __init__(
        self,
        n_experts=3,
        out_dim=2,
        dropout=0.1,
        centers_deg=None,
        sharpness=10.0,
        mag_center_deg=15.0,
        mag_sharpness=1.0,
    ):
        super().__init__()

        if n_experts not in (1, 2, 3):
            raise ValueError("n_experts must be 1, 2, or 3")
        
        self.n_experts = n_experts

        a = 8
        n_filters = 256 // a
        aux_dim = 2

        def make_cnn():
            return nn.Sequential(
                nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),

                nn.Conv2d(n_filters, n_filters, kernel_size=2),
                nn.BatchNorm2d(n_filters),
                nn.GELU(),
            )

        filter_dim = {
            1: 20736,
            4: 5184,
            8: 2592,
        }[a]

        in_dim = filter_dim + aux_dim

        def make_head():
            return nn.Sequential(
                nn.Linear(in_dim, 4096 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(4096 // a, 2048 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(2048 // a, 1024 // a),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(1024 // a, out_dim),
            )

        self.cnn = nn.ModuleList([make_cnn() for _ in range(n_experts)])
        self.heads = nn.ModuleList([make_head() for _ in range(n_experts)])

        self.controller_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_experts + 1),
        )



    def forward(self, batch):
        corr = batch["corr_matrices"]
        angles_hat = batch["angles_hat"]

        z = self.cnn[0](corr)
        z = z.flatten(start_dim=1)
        z = torch.cat([z, angles_hat], dim=1)

        expert_deltas = torch.stack(
            [head(z) for head in self.heads],
            dim=1,
        )  # (B, E, 2)

        router_out = self.controller_head(z)

        expert_logits = router_out[:, :self.n_experts]
        confidence_logit = router_out[:, self.n_experts:self.n_experts + 1]

        expert_gate = F.softmax(expert_logits, dim=1)
        confidence = torch.sigmoid(confidence_logit)

        mixture_delta = (expert_gate[:, :, None] * expert_deltas).sum(dim=1)
        delta = confidence * mixture_delta

        return {
            "ang_hat": angles_hat + delta,
            "delta": delta,
            "mixture_delta": mixture_delta,
            "expert_deltas": expert_deltas,
            "expert_gate": expert_gate,
            "confidence": confidence,
            "gate": confidence,
        }