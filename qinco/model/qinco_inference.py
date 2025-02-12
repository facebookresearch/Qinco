# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch import nn

from ..utils import approx_compute_batch_distances, approx_pairwise_distance

#####################################################################
# QINCo inference model
#####################################################################


class QINCoInferenceStep(nn.Module):
    def __init__(self, cfg, qinco_model, i_step):
        super().__init__()

        self.i_step = i_step
        step = qinco_model.steps[i_step]
        self.concat = step.concat
        self.in_proj = step.in_proj
        self.out_proj = step.out_proj
        self.res_blocks = step.residual_blocks
        self.res_codeword_coeff = 0.0 if cfg.qinco1_mode else 1.0

    def forward(self, codewords_BD, xhat_BD):
        codewords_BD_in = codewords_BD
        codewords_BD = self.in_proj(codewords_BD)
        codewords_BD = self.concat(codewords_BD, xhat_BD)

        for res_block in self.res_blocks:
            codewords_BD = res_block(codewords_BD)

        codewords_BD = self.out_proj(codewords_BD)
        return codewords_BD + self.res_codeword_coeff * codewords_BD_in


class QINCoInferenceDecoder(nn.Module):
    STEP_CLS = QINCoInferenceStep

    def __init__(self, cfg, qinco_model):
        super().__init__()

        self.M = len(qinco_model.steps)
        self.d = qinco_model.steps[0].D
        self.K_vals = torch.tensor(cfg._K_vals)
        self.K_shifts: torch.Tensor = (
            torch.cumsum(torch.concat([torch.tensor([0]), self.K_vals[:-1]]), dim=0)
            .reshape(self.M, 1)
            .to(cfg._accelerator.device)
        )

        self.level_modules = nn.ModuleList()
        for i_step in range(1, self.M):
            self.level_modules.append(self.STEP_CLS(cfg, qinco_model, i_step))

        self.codebook = nn.Parameter(
            torch.concat([step.get_codebook_weight() for step in qinco_model.steps])
        )

    def forward(self, codes_MB: torch.Tensor):
        M, B = codes_MB.shape

        codes_MB_mapped = codes_MB + self.K_shifts
        rq_codewords_MBD = self.codebook[codes_MB_mapped]

        xhat = rq_codewords_MBD[0]
        for lvl in self.level_modules:
            xhat += lvl(rq_codewords_MBD[lvl.i_step], xhat)
        return xhat.reshape(B, self.d)


class QINCoInferenceStepEncoderNoSubstep(QINCoInferenceStep):
    def __init__(self, cfg, qinco_model, i_step):
        super().__init__(cfg, qinco_model, i_step)

        step = qinco_model.steps[i_step]
        self._ref_step = (step,)
        self.codebook = nn.Parameter(step.get_codebook_weight())

        self.F_beam_out: int = cfg.B if i_step < cfg._M_ivf - 1 else 1
        self.res_codeword_coeff = 0.0 if cfg.qinco1_mode else 1.0

    def forward(self, x_BD, xhat_BFD, codes_MBF):
        B, F, D = xhat_BFD.shape

        # Compute dimensions
        K, D = self.codebook.shape
        B_base, F_beam_in, D = xhat_BFD.shape
        B = B_base * F_beam_in
        F_beam_out = self.F_beam_out

        # Compute codewords and target we will use
        xhat_BD = xhat_BFD.reshape(B, D)

        codewords_r_BKD = self.codebook.reshape(1, K, D).broadcast_to(B, K, D)

        # Compute the mapped codes
        codewords_BFKD = codewords_r_BKD.reshape(B_base, F_beam_in, K, D)
        codewords_BFKD_in = codewords_BFKD
        codewords_BFKD = self.in_proj(codewords_BFKD)
        codewords_BFKD = self.concat(
            codewords_BFKD, xhat_BFD.unsqueeze(2).broadcast_to(B_base, F_beam_in, K, D)
        )
        for res_block in self.res_blocks:
            codewords_BFKD = res_block(codewords_BFKD)
        codewords_BFKD = (
            self.out_proj(codewords_BFKD) + self.res_codeword_coeff * codewords_BFKD_in
        )

        codewords_r_BKD = codewords_BFKD.reshape(B, K, D)
        codewords_r_BKD = codewords_r_BKD + xhat_BD.unsqueeze(-2)

        # Assign codes
        codewords_rgflat_BKD = codewords_r_BKD.reshape(B_base, F_beam_in * K, D)

        # Select argmin
        dists_BFk = approx_compute_batch_distances(
            x_BD.unsqueeze(1), codewords_rgflat_BKD
        ).squeeze(1)
        codes_BF_idx = dists_BFk.argmin(dim=-1).unsqueeze(-1)

        # Gather next xhat
        codewords_BFkD = codewords_r_BKD.reshape(B_base, F_beam_in * K, D)
        xhat_next_BFD = codewords_BFkD.gather(
            dim=-2,
            index=codes_BF_idx.reshape(B_base, F_beam_out, 1).broadcast_to(
                B_base, F_beam_out, D
            ),
        )

        # Append new codes & xhat
        codes_MBF = torch.concat([codes_MBF, codes_BF_idx.unsqueeze(0)])

        return xhat_next_BFD, codes_MBF


class QINCoInferenceStepEncoder(QINCoInferenceStep):
    def __init__(self, cfg, qinco_model, i_step):
        super().__init__(cfg, qinco_model, i_step)

        step = qinco_model.steps[i_step]
        self._ref_step = (step,)
        self.codebook = nn.Parameter(step.get_codebook_weight())
        self.codebook_rq = step.substep.codebook.weight

        self.F_beam_out: int = cfg.B if i_step < cfg._M_ivf - 1 else 1
        self.n_codes: int = qinco_model.steps[i_step].substep._n_codes
        self.res_codeword_coeff = 0.0 if cfg.qinco1_mode else 1.0

    def forward(self, x_BD, xhat_BFD, codes_MBF):
        B, F, D = xhat_BFD.shape
        M = len(codes_MBF)

        # Compute dimensions
        K, D = self.codebook.shape
        B_base, F_beam_in, D = xhat_BFD.shape
        B = B_base * F_beam_in
        F_beam_out = self.F_beam_out

        # Compute codewords and target we will use
        xhat_BD = xhat_BFD.reshape(B, D)
        codewords_KDg = self.codebook

        # Fetch sub-RQ candidates
        xtarget_BD = x_BD.unsqueeze(-2) - xhat_BFD
        dists_BK = approx_pairwise_distance(xtarget_BD.reshape(B, D), self.codebook_rq)
        top_codes_g_BK = dists_BK.topk(self.n_codes, dim=-1, largest=False).indices
        B, K = top_codes_g_BK.shape
        codewords_r_BKD = codewords_KDg[top_codes_g_BK]

        # Compute the mapped codes
        codewords_BFKD = codewords_r_BKD.reshape(B_base, F_beam_in, K, D)
        codewords_BFKD_in = codewords_BFKD
        codewords_BFKD = self.in_proj(codewords_BFKD)
        codewords_BFKD = self.concat(
            codewords_BFKD, xhat_BFD.unsqueeze(2).broadcast_to(B_base, F_beam_in, K, D)
        )
        for res_block in self.res_blocks:
            codewords_BFKD = res_block(codewords_BFKD)
        codewords_BFKD = (
            self.out_proj(codewords_BFKD) + self.res_codeword_coeff * codewords_BFKD_in
        )

        codewords_r_BKD = codewords_BFKD.reshape(B, K, D)
        codewords_r_BKD = codewords_r_BKD + xhat_BD.unsqueeze(-2)

        # Assign codes
        codewords_rgflat_BKD = codewords_r_BKD.reshape(B_base, F_beam_in * K, D)

        # Select topk
        dists_BFk = approx_compute_batch_distances(
            x_BD.unsqueeze(1), codewords_rgflat_BKD
        ).squeeze(1)
        codes_BF_idx = dists_BFk.topk(F_beam_out, dim=-1, largest=False).indices

        # Gather real code ids
        top_codes_g_BFk = top_codes_g_BK.reshape(B_base, F_beam_in * K)
        codes_g_BF = top_codes_g_BFk.gather(-1, codes_BF_idx)

        # Gather previous codes
        codes_MBFk = codes_MBF.repeat_interleave(self.n_codes, dim=-1)
        codes_MBF = codes_MBFk.gather(
            -1, codes_BF_idx.unsqueeze(0).broadcast_to(M, B_base, F_beam_out)
        )

        # Gather next xhat
        codewords_BFkD = codewords_r_BKD.reshape(B_base, F_beam_in * K, D)
        xhat_next_BFD = codewords_BFkD.gather(
            dim=-2,
            index=codes_BF_idx.reshape(B_base, F_beam_out, 1).broadcast_to(
                B_base, F_beam_out, D
            ),
        )

        # Append new codes
        codes_MBF = torch.concat([codes_MBF, codes_g_BF.unsqueeze(0)])

        return xhat_next_BFD, codes_MBF


class QINCoInferenceEncoder(QINCoInferenceDecoder):
    STEP_CLS = QINCoInferenceStepEncoder

    def __init__(self, cfg, qinco_model):
        if not cfg.A:
            self.STEP_CLS = QINCoInferenceStepEncoderNoSubstep
        super().__init__(cfg, qinco_model)

        self.K0 = int(self.K_vals[0])
        self.beam = cfg.B
        self.beam_0 = min(1 if cfg.ivf_in_use else self.beam, self.K0)

    def forward(self, x_BD: torch.Tensor):
        # Codebook 0
        dists_BK = approx_pairwise_distance(x_BD, self.codebook[: self.K0])
        if self.beam_0 == 1:
            codes0 = dists_BK.argmin(-1).unsqueeze(-1)
        else:
            codes0 = dists_BK.topk(self.beam_0, dim=-1, largest=False).indices
        xhat_MBD = self.codebook[codes0]

        # Beam search over all codebooks
        codes_MBF = codes0.unsqueeze(0)
        for lvl in self.level_modules:
            xhat_MBD, codes_MBF = lvl(x_BD, xhat_MBD, codes_MBF)

        codes_MB = codes_MBF.squeeze(-1)
        return codes_MB, xhat_MBD.squeeze(-2)


class QINCoInferenceWrapper(nn.Module):
    """
    QINCo decoder optimized for sequential decoding. Should only be used for evaluation.
    """

    def __init__(self, cfg, qinco_model):
        super().__init__()
        self.cfg = cfg
        self.print = self.cfg._accelerator.print

        self.qinco_model = qinco_model
        self.built = False
        self.data_mean = qinco_model.data_mean
        self.data_std = qinco_model.data_std

    def forward(self, x_in, *args, step="train", **kwargs):
        assert step in ["encode", "decode"]
        if step == "train":
            raise Exception("Don't use inference wrapper for training!")
        elif step == "encode":
            codes, xhat_BD = self.encode((x_in - self.data_mean) / self.data_std)
            return codes
        elif step == "decode":
            x_out = self.decode(x_in, *args, **kwargs)
            return x_out * self.data_std + self.data_mean
        else:
            raise ValueError(f"{step=}")

    def load_state_dict(self, state_dict, **kwargs):
        self.qinco_model.load_state_dict(state_dict, **kwargs)

        self.build()

    def build(self):
        self.encoder = QINCoInferenceEncoder(self.cfg, self.qinco_model)
        self.decoder = QINCoInferenceDecoder(self.cfg, self.qinco_model)
        self.optimize()
        self.built = True

    @torch.inference_mode
    def _optimize_model_part(self, model, decoder=False):
        NAME = "decoder" if decoder else "encoder"
        self.cfg._accelerator.print(f"Optimizing {NAME}")
        device = self.cfg._accelerator.device
        model = model.to(device).eval()

        self.float16 = device.type != "cpu"

        if decoder:
            input_rand = torch.zeros(
                (self.cfg._M_ivf, self.cfg.batch), dtype=int, device=device
            )
        else:
            input_rand = torch.rand(
                (self.cfg.batch, self.cfg._D),
                device=device,
                dtype=(torch.float16 if self.float16 else torch.float32),
            )

        if self.float16:
            model = model.half()

        model = torch.jit.script(model)
        if device.type != "cpu" and os.environ.get("PYTORCH_JIT") != "0":
            model = torch.jit.optimize_for_inference(model)

        self.cfg._accelerator.print(f"Warmup {NAME}")
        _ = model(input_rand)  # Warmup
        self.cfg._accelerator.print(f"Done for {NAME}")
        return model

    def optimize(self):
        self.decoder = self._optimize_model_part(self.decoder, True)
        self.encoder = self._optimize_model_part(self.encoder)

    @torch.inference_mode
    def decode(self, codes_MB):
        assert isinstance(codes_MB, torch.Tensor)
        if self.float16:
            return self.decoder(codes_MB).float()
        else:
            return self.decoder(codes_MB)

    @torch.inference_mode
    def encode(self, x_target_BD):
        if self.float16:
            codes_MB, xhat_MBD = self.encoder(x_target_BD.half())
            return codes_MB, xhat_MBD.float()
        else:
            with torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):  # Mixed precision beneficial only during encoding
                codes_MB, xhat_MBD = self.encoder(x_target_BD)
                return codes_MB, xhat_MBD

    def get_codebooks_refs(self):
        return self.qinco_model.get_codebooks_refs()
