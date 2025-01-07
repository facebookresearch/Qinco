from collections.abc import Iterable

import torch
from einops import rearrange, repeat
from torch import nn

import qinco

from ..utils import (
    approx_pairwise_distance,
    compute_batch_distances,
    merge_losses,
    pairwise_distances,
)

#####################################################################
# Shared model initizalition
#####################################################################


@torch.no_grad
def initialize_qinco_codebooks(cfg, model, rq_centroids):
    """Initialize the codebook weights from RQ"""
    if rq_centroids is not None:
        for i_stage, codebooks in enumerate(model.get_codebooks_refs()):
            step_centroid: torch.Tensor = rq_centroids[i_stage]
            if i_stage == 0 and not cfg.ivf_in_use:
                step_centroid = (step_centroid - model.data_mean) / model.data_std
            else:
                step_centroid = step_centroid / model.data_std
            step_centroid = step_centroid + torch.randn_like(
                step_centroid
            ) * step_centroid.std() * (
                cfg.codebook_noise_init if not cfg.qinco1_mode else 1
            )

            for weight in codebooks:
                weight.copy_(step_centroid)


#####################################################################
# Block Components
#####################################################################


class QConcat(nn.Module):
    def __init__(self, cfg, d, embed_dim=None):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Linear(embed_dim + d, embed_dim)

        self.initialize_weights()

    def forward(self, zqs, xhat):
        cc = torch.concatenate((zqs, xhat), dim=-1)
        cc = self.mlp(cc)
        cc = zqs + cc
        return cc

    def initialize_weights(self):
        if not self.cfg.qinco1_mode:
            nn.init.constant_(self.mlp.bias, 0)
            nn.init.constant_(self.mlp.weight, 0)


class QBlockFFN(nn.Module):
    """
    Basic residual FFN block for QINCo.
    Works on the last dimension of the vectors, whatever their size is.
    """

    def __init__(self, cfg, hidden_dim, embed_dim):
        super().__init__()
        self.cfg = cfg

        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.act = torch.nn.ReLU()
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        if not self.cfg.qinco1_mode:
            nn.init.kaiming_uniform_(self.up_proj.weight, nonlinearity="relu")
            nn.init.constant_(self.down_proj.weight, 0)

    def forward(self, x_in: torch.Tensor):
        x = self.up_proj(x_in)
        x = self.act(x)
        x_out = self.down_proj(x)
        return x_in + x_out


class QincoSubstep(nn.Module):
    """
    QINCo sub-step without adaptive codewords
    """

    def __init__(self, cfg, K, i_step, d):
        super().__init__()
        self.codebook = nn.Embedding(K, d)
        self._n_codes = cfg.A
        if i_step == 0 or (
            i_step == 1 and cfg._ivf_book
        ):  # Ensure we have enough candidates, either for the first step (QINCo), or second (IVF-QINCo)
            self._n_codes = max(cfg.B, self._n_codes)

    def get_distances_for_codes(self, xhat_BD: torch.Tensor, xtarget_BD: torch.Tensor):
        codewords_KD = self.codebook.weight
        xtarget_remain_BD = xtarget_BD.detach() - xhat_BD.detach()
        return pairwise_distances(xtarget_remain_BD, codewords_KD)

    def select_code_candidates(self, xhat_BD: torch.Tensor, xtarget_BD: torch.Tensor):
        dists_BK = self.get_distances_for_codes(xhat_BD, xtarget_BD)
        return dists_BK.topk(self._n_codes, dim=-1, largest=False).indices

    def get_loss(self, last_x_BD, xhat_in_BD, codes_B):
        xhat_out_BD = xhat_in_BD + self.codebook(codes_B)
        return ((xhat_out_BD - last_x_BD) ** 2).mean()


class IVFBook(nn.Module):
    """Special QINCo step class, optimized to handle very large codebooks without beam search, for the IVF codebook step"""

    IVF_BS_MAX = 2**30

    def __init__(self, cfg, np_centroids=None):
        super().__init__()

        self.cfg = cfg

        self.K, self.D = cfg.ivf_K, cfg._D
        self.ivf_centroids = nn.Embedding(self.K, self.D)
        self.ivf_centroids.requires_grad_(False)
        if np_centroids is not None:
            self.ivf_centroids.weight.copy_(
                torch.tensor(np_centroids, device=cfg._accelerator.device)
            )

    def quantize(self, x_BD):
        # Do a batched pairwise distance computation
        codes_B = []
        cur_ivf_bs = self.IVF_BS_MAX // len(self.ivf_centroids.weight)
        B, D = x_BD.shape
        for i in range(0, B, cur_ivf_bs):
            dists = approx_pairwise_distance(
                x_BD[i : i + cur_ivf_bs].to(self.cfg._accelerator.device),
                self.ivf_centroids.weight,
            )
            codes_B.append(dists.argmin(-1))

        codes_B = torch.concatenate(codes_B, dim=0)
        quant_x_BD = self.ivf_centroids(codes_B)
        return quant_x_BD, codes_B

    def get_codebook_weight(self):
        return self.ivf_centroids.weight.to(self.cfg._accelerator.device)

    ##### Function to allow to use IVFBook as a QINCo step, or during RQ training #####

    @torch.no_grad
    def encode(self, xhat_BFD, x_BD, codes_prev=None):
        # Same signature as step.encode
        B, F, D = xhat_BFD.shape

        xhat_next_BD, codes_B = self.quantize(x_BD)
        xhat_next = xhat_next_BD.reshape(B, F, D)
        return xhat_next, [codes_B.reshape(B, F)]

    @torch.no_grad
    def decode(self, codes_B, xhat_list=None, codes_Ll_B=None):
        if codes_Ll_B is not None:  # Called from QINCo
            codes_B = codes_Ll_B[0].reshape(-1)
        xhat = self.ivf_centroids(codes_B)
        if codes_Ll_B is not None:  # Called from QINCo
            xhat = xhat.reshape(xhat.shape[0], 1, 1, xhat.shape[-1])
        return xhat

    @torch.no_grad
    def codebook(self, codes):
        return self.ivf_centroids(codes)

    def get_centroids(self):
        return "ivf"

    def reset_unused_codebooks(self, *args, **kwargs):
        return 0, 0  # Don't reset, as the IVF is frozen

    def collect_losses(self, *args, **kwargs):
        return {}  # No loss, as the IVF is frozen


#####################################################################
# The base QINCo model
#####################################################################


class QINCoStep(nn.Module):
    """
    One quantization step for QINCo.
    Contains the codebook, concatenation block, and residual blocks
    """

    def __init__(self, cfg, qinco_model, D, i_step):
        nn.Module.__init__(self)

        # Step initialization
        self.cfg = cfg
        self._qinco_ref = (qinco_model,)
        self.D = D
        self.i_step = i_step
        self.codebook_only = i_step == 0
        self.beam_size = cfg.B

        De = cfg.de or self.D
        Dh = cfg.dh

        self.has_substep = bool(cfg.A) and not self.codebook_only
        self.substep = (
            QincoSubstep(self.cfg, cfg.K, i_step, self.D) if self.has_substep else None
        )

        self.codebook = nn.Embedding(cfg.K, self.D)

        if not self.codebook_only:
            self.concat = QConcat(cfg, self.D, embed_dim=De)

            self.residual_blocks = nn.Sequential(
                *[QBlockFFN(cfg, Dh, De) for _ in range(cfg.L)]
            )

            if De != self.D:
                self.in_proj = nn.Linear(self.D, De, bias=False)
                self.out_proj = nn.Linear(De, self.D, bias=False)
                if not cfg.qinco1_mode:  # Initialize weights
                    nn.init.kaiming_uniform_(self.in_proj.weight, nonlinearity="relu")
                    nn.init.kaiming_uniform_(self.out_proj.weight, nonlinearity="relu")
            else:
                self.in_proj = nn.Identity()
                self.out_proj = nn.Identity()

            if cfg._qinco_jit:
                self.concat = torch.jit.script(self.concat)
                self.in_proj = torch.jit.script(self.in_proj)
                self.out_proj = torch.jit.script(self.out_proj)
                self.residual_blocks = torch.jit.script(self.residual_blocks)

        self.stats_momentum = 0.1
        self.register_buffer(
            "xtarget_mean", torch.zeros(self.D).to(cfg._accelerator.device)
        )
        self.register_buffer(
            "xtarget_var", torch.ones(self.D).to(cfg._accelerator.device)
        )

    def forward(self, codewords_BKD, prev_xhat_BKD):
        if not self.codebook_only:
            # Then, compute the conditioned predicted codeword
            assert (
                codewords_BKD.shape == prev_xhat_BKD.shape
            ), f"{codewords_BKD.shape=} != {prev_xhat_BKD.shape=}"
            codewords_BKD_at_start = codewords_BKD
            codewords_BKE = self.in_proj(codewords_BKD)

            codewords_BKE = self.concat(codewords_BKE, prev_xhat_BKD)
            # for residual_block in self.residual_blocks:
            #     codewords_BKE = residual_block(codewords_BKE)
            codewords_BKE = self.residual_blocks(codewords_BKE)

            codewords_BKD = self.out_proj(codewords_BKE)
            if not self.cfg.qinco1_mode:
                codewords_BKD = codewords_BKD_at_start + codewords_BKD

        return codewords_BKD

    def decode(self, codes_B, xhat_BD):
        assert codes_B.ndim == 1
        assert xhat_BD.ndim == 2
        assert len(codes_B) == len(xhat_BD)

        # Add extra dimension
        codewords_BKD = self.codebook(codes_B.unsqueeze(1))

        return self(codewords_BKD, xhat_BD.unsqueeze(1)).squeeze(1)

    def encode(self, xhat_BFD, x_BD, codes_Ll_BF):
        # Update running mean / var
        x_target: torch.Tensor = x_BD.unsqueeze(1) - xhat_BFD
        if self.training:
            assert not torch.is_inference_mode_enabled()
            self.xtarget_mean.copy_(
                self.stats_momentum * (x_target.mean(dim=(0, 1)))
                + (1.0 - self.stats_momentum) * self.xtarget_mean
            )
            self.xtarget_var.copy_(
                self.stats_momentum * (x_target.std(dim=(0, 1)))
                + (1.0 - self.stats_momentum) * self.xtarget_var
            )

        # Compute dimensions
        K, D = self.codebook.weight.shape
        B_base, F_beam_in, D = xhat_BFD.shape
        B = B_base * F_beam_in
        F_beam_out = self.beam_size if self.i_step < self.cfg._M_ivf - 1 else 1
        assert x_BD.shape == (B_base, D)

        # Compute codewords and target we will use
        xhat_BD = xhat_BFD.reshape(B, D)

        # If we use a substep, fetch candidates
        if self.has_substep:  # Alway do this step, even if not used, to allow training
            x_BfD = repeat(x_BD, "B D -> (B F) D", F=F_beam_in)
            assert x_BfD.shape == xhat_BD.shape
            top_codes_BK = self.substep.select_code_candidates(xhat_BD, x_BfD)
            assert len(top_codes_BK) == B

            B, K = top_codes_BK.shape
            codewords_r_BKD = self.codebook.weight[top_codes_BK]
        else:  # If we don't use a substep, use all codes as possibles candidates
            codewords_r_BKD = self.codebook.weight.view(1, K, D).broadcast_to(B, K, D)

        # Compute the mapped codes, with a possible filtering of code candidates
        assert codewords_r_BKD.shape == (B, K, D)
        codewords_r_BKD = self(
            codewords_r_BKD, xhat_BD.reshape(B, 1, D).broadcast_to(B, K, D)
        )
        _, K, _ = codewords_r_BKD.shape  # K might have been updated

        codewords_r_BKD = codewords_r_BKD + xhat_BD.unsqueeze(-2)

        ### Assign codes
        codewords_rflat_BKD = rearrange(
            codewords_r_BKD, "(B F) K D -> B (F K) D", F=F_beam_in
        )

        # Select topk
        dists_BFk = compute_batch_distances(
            x_BD.unsqueeze(1), codewords_rflat_BKD
        ).squeeze(1)
        codes_BF = dists_BFk.topk(F_beam_out, dim=-1, largest=False).indices

        # Gather real code ids
        if self.has_substep:
            realidx_codes_BF = rearrange(
                top_codes_BK, "(B F) K -> B (F K)", F=F_beam_in
            ).gather(-1, codes_BF)
        else:
            realidx_codes_BF = codes_BF % K

        # Gather previous codes
        codes_Ll_BF = [
            repeat(step_codes_BF, "B F -> B (F K)", K=K).gather(-1, codes_BF)
            for step_codes_BF in codes_Ll_BF
        ]

        # Gather next xhat
        codewords_BFkD = rearrange(
            codewords_r_BKD, "(B F) K D -> B (F K) D", F=F_beam_in
        )
        assert codewords_BFkD.shape == (B_base, F_beam_in * K, D)
        xhat_next_BFD = codewords_BFkD.gather(
            dim=-2, index=repeat(codes_BF, "B F -> B F D", D=D)
        )

        # Append new codes
        codes_Ll_BF.append(realidx_codes_BF)

        return xhat_next_BFD, codes_Ll_BF

    def collect_losses(self, last_x_BD, xhat_BD, xhat_prev_BD, codes):
        losses = {"mse_loss": ((xhat_BD - last_x_BD) ** 2).mean()}
        if self.has_substep:
            losses["loss_substep"] = self.substep.get_loss(
                last_x_BD, xhat_prev_BD.detach(), codes
            )

        return losses

    def reset_unused_codebooks(self, codebook_usage):
        assert len(codebook_usage) == len(self.codebook.weight)
        codebook_usage = (codebook_usage > 0).unsqueeze(-1)

        # Initialize a new random codebook
        new_codebook = torch.rand(
            self.codebook.weight.shape, device=self.codebook.weight.device
        )
        new_codebook = (new_codebook - new_codebook.mean()) / new_codebook.std()
        new_codebook = new_codebook * self.xtarget_var + self.xtarget_mean

        # Reset unused codewords in the codebook (and in substep codebook if it exists)
        self.codebook.weight.copy_(
            self.codebook.weight * codebook_usage + new_codebook * (~codebook_usage)
        )
        if self.has_substep and self.substep.codebook is not None:
            sub_book = self.substep.codebook.weight
            new_codebook = (
                new_codebook
                + torch.randn_like(new_codebook)
                * new_codebook.std()
                * (self.cfg.codebook_noise_init if not self.cfg.qinco1_mode else 1)
                / 4
            )
            sub_book.copy_(sub_book * codebook_usage + new_codebook * (~codebook_usage))

        # Return number of reset codebooks
        n_reset = int((~codebook_usage).to(int).sum().cpu().float())
        return n_reset, len(self.codebook.weight)

    def get_codebook_weight(self):
        return self.codebook.weight.to(self.cfg._accelerator.device)


class QINCo(nn.Module):
    """
    QINCo quantizer, built from a chain of residual quantization steps
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.print = self.cfg._accelerator.print
        self.D = cfg._D
        self.M = cfg._M_ivf
        self.device = cfg._accelerator.device
        self.data_mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.data_std = nn.Parameter(torch.zeros(()), requires_grad=False)

        if cfg.task == "train":
            self.data_mean.copy_(torch.from_numpy(self.cfg._data_mean))
            self.data_std.copy_(float(self.cfg._data_std))

        self.steps = nn.ModuleList()
        for m in range(self.M):
            if m == 0 and cfg._ivf_book:
                step = cfg._ivf_book
            else:
                step = QINCoStep(cfg, self, self.D, i_step=m)
            self.steps.append(step)

    def decode(self, codes_MB):
        xhat = torch.zeros((len(codes_MB[0]), self.D), device=self.device)
        assert len(codes_MB) == len(self.steps)
        for step, codes_B in zip(self.steps, codes_MB):
            xhat += step.decode(codes_B, xhat)
        return xhat

    def encode(self, x_target_BD):
        torch.cuda.empty_cache()
        if self.cfg.enc_max_bs:
            max_beam = (
                max(self.cfg.B) if isinstance(self.cfg.B, Iterable) else self.cfg.B
            )
            xin_bs = self.cfg.enc_max_bs // (max_beam * (self.cfg.A or 1))
            if xin_bs < len(
                x_target_BD
            ):  # Encode by parts to avoid memory issues with large beam size
                all_codes_Bl_Mb, all_xhat_Bl_bD = [], []
                for i_bs in range(0, len(x_target_BD), xin_bs):
                    codes_MB1, xhat_BD = self.encode(x_target_BD[i_bs : i_bs + xin_bs])
                    all_codes_Bl_Mb.append(codes_MB1)
                    all_xhat_Bl_bD.append(xhat_BD)

                codes_MB1 = torch.concat(all_codes_Bl_Mb, axis=1)
                xhat_BFD = torch.concat(all_xhat_Bl_bD)
                return codes_MB1, xhat_BFD.squeeze(1)

        codes_Ll_BF = []
        xhat_BFD = torch.zeros_like(x_target_BD, device=x_target_BD.device).unsqueeze(1)

        for step in self.steps:
            xhat_BFD, codes_Ll_BF = step.encode(xhat_BFD, x_target_BD, codes_Ll_BF)

        assert codes_Ll_BF[0].shape[1] == 1  # Assert beam size 1 at the start
        assert codes_Ll_BF[-1].shape[1] == 1  # Assert beam size 1 at the end
        assert xhat_BFD.shape[1] == 1

        codes_MB1 = torch.stack(codes_Ll_BF).squeeze(2)
        return codes_MB1, xhat_BFD.squeeze(1)

    def _train_encode_decode(self, x_BD):
        with torch.no_grad():
            codes, _ = self.encode(x_BD)

        # then decode step by step and collect losses
        losses = {}
        xhat_BD = torch.zeros_like(x_BD, device=x_BD.device)
        xhat_list = []

        for i, step in enumerate(self.steps):
            xhat_prev_BD = xhat_BD  # Used for substep loss
            xhat_BD = xhat_BD + step.decode(codes[i], xhat_BD)
            xhat_list.append(xhat_BD)
            step_losses = step.collect_losses(x_BD, xhat_BD, xhat_prev_BD, codes[i])
            losses = merge_losses(losses, step_losses)

        return codes, xhat_BD, losses

    def reset_unused_codebooks(self, codebook_usage):
        if not self.cfg.qinco1_mode:
            assert len(codebook_usage) == len(self.steps)
            step_resets, step_tot = [], []
            for step, usage in zip(self.steps, codebook_usage):
                n_reset, n_tot = step.reset_unused_codebooks(usage)
                step_resets.append(n_reset)
                step_tot.append(n_tot)

            if sum(step_resets):
                step_reset_str = [f"{r}/{t}" for r, t in zip(step_resets, step_tot)]
                self.cfg._accelerator.print(
                    f"Reset {sum(step_resets)}/{sum(step_tot)} codewords at the end of epoch {self.cfg._cur_epoch} (for each step: {step_reset_str})"
                )
            else:
                self.cfg._accelerator.print(
                    f"No codeword to reset after epoch {self.cfg._cur_epoch}"
                )

    def forward(self, x_in, step="train"):
        assert step in ["train", "encode", "decode"]
        assert self.data_std.item() > 0
        if step == "train":
            codes, xhat_BD, losses = self._train_encode_decode(
                (x_in - self.data_mean) / self.data_std
            )
            return codes, (xhat_BD * self.data_std + self.data_mean), losses
        elif step == "encode":
            codes, xhat_BD = self.encode((x_in - self.data_mean) / self.data_std)
            return codes
        elif step == "decode":
            x_out = self.decode(x_in)
            return x_out * self.data_std + self.data_mean
        else:
            raise ValueError(f"{step=}")

    def get_codebooks_refs(self):
        """Used for initialization of the codebooks"""
        codebooks_refs = []
        for step in self.steps:
            if not isinstance(step, IVFBook):
                codebooks_refs.append([step.codebook.weight])
                if step.substep is not None and step.substep.codebook is not None:
                    codebooks_refs[-1].append(step.substep.codebook.weight)
        return codebooks_refs
