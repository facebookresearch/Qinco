import numpy as np
import torch
from torch import nn

from ..utils import corrected_mean_squared_error, approx_pairwise_distance, torch_sum
from ..metrics import Timer


class PairwiseDecoderIVF(nn.Module):
    IVF_M = 5
    ITER_STEPS = 50

    def __init__(self, cfg, train_codes_MB=None, trainset_BD=None, load=True, ivf_centroids=None, ivf_codes=None, valset_BD=None, valset_BD_past_ivf=None, val_codes_MB=None):
        super().__init__()


        self.K = cfg.K
        self.K_base = self.K
        self.M = cfg.M
        self.M_base = cfg.M
        self.D = cfg._D
        self.M_target = round(cfg.n_pairwise_codebooks * cfg.M)
        self.K_target = self.K**2

        self.cfg = cfg
        self.accelerator = cfg._accelerator
        self.acc_print = self.accelerator.print
        self.device = self.accelerator.device

        self.pre_train_init()
        if load:
            self.set_loading_parameters()
            load_path = load if isinstance(load, str) else cfg.pairwise_decoder
            state_dict = torch.load(load_path, map_location=torch.device('cpu'), weights_only=True)
            self.load_state_dict(state_dict, strict=True)
        else:
            self.train_B = len(trainset_BD)
            self.train_codes_MB = train_codes_MB.to(self.device)
            self.trainset_BD = trainset_BD.to(self.device)
            self.codebook_MKD = None
            self.ivf_centroids = ivf_centroids
            self.ivf_codes = ivf_codes

            self.train()

            del self.train_codes_MB
            del self.trainset_BD
        
            self.valset_BD = valset_BD
            self.valset_BD_past_ivf = valset_BD_past_ivf
            self.val_codes_MB = val_codes_MB
    
    def compute_ds_mse(self, dataset, datacodes, ivf_codes=None):
        mse_vals = []
        bs = 2**14
        for k in range(0, len(dataset), bs):
            if ivf_codes is not None:
                quant_x = self(datacodes[:,k:k+bs], ivf_codes[k:k+bs])
            else:
                quant_x = self(datacodes[:,k:k+bs])
            ref_x = torch.from_numpy(dataset[k:k+bs]).to(quant_x.device)
            err = corrected_mean_squared_error(self.cfg, ref_x, quant_x)
            mse_vals.append(err * len(ref_x))
        return torch_sum(mse_vals) / len(dataset)

    def pre_train_init(self):
        self.M = self.M_target
        self.K = self.K_target

        self.M_base_combined = self.M_base + self.IVF_M
        self.ivf_small_codebooks = torch.empty((self.IVF_M, self.K_base, self.D), device=self.device)
        self.ivf_code_map = nn.Parameter(torch.empty((self.cfg.ivf_K, self.IVF_M), device=self.device, dtype=int), requires_grad=False)
        self.combine_mvals_m = nn.Parameter(torch.empty((2, self.M_target), device=self.device, dtype=int), requires_grad=False)

    def set_loading_parameters(self):
        self.M = self.M_target
        self.K = self.K_target

        self.combine_mvals_m = nn.Parameter(torch.empty((2, self.M_target), device=self.device, dtype=int), requires_grad=False)
        self.codebook_MKD = nn.Parameter(torch.empty((self.M, self.K, self.D), dtype=torch.float32, device=self.device), requires_grad=False)

    def forward(self, codes_MB, ivf_codes=None):
        codes_MB = self.map_codes(codes_MB, ivf_codes)
        xhat_BD = self.codebook_MKD[0][codes_MB[0]]
        for codebook_KD, codes_B in zip(self.codebook_MKD[1:], codes_MB[1:]):
            xhat_BD += codebook_KD[codes_B]
        return xhat_BD

    
    def get_combined(self, m1, m2, N=None):
        if N: return self.train_codes_MB[m1][:N] * self.K_base + self.train_codes_MB[m2][:N]
        return self.train_codes_MB[m1] * self.K_base + self.train_codes_MB[m2]

    def build_combined_codebook(self, m1, m2, x_remain_BD):
        codes_combined = self.get_combined(m1, m2)
        counts_K2 = torch.zeros(self.K, device=self.device).scatter_add_(0, codes_combined, torch.ones(len(x_remain_BD), device=self.device))
        book_K2D = torch.zeros((self.K, self.D), device=self.device).index_add_(0, codes_combined, x_remain_BD)
        book_K2D /= counts_K2.unsqueeze(-1).clamp(min=1)

        max_N = 100_000
        codes_combined = self.get_combined(m1, m2, N=max_N)
        local_error = self.compute_mse(x_remain_BD[:max_N] - book_K2D[codes_combined])

        return book_K2D, local_error

    def compute_mse(self, x_remain):
        mse_vals = []
        bs = 2**14
        for k in range(0, len(x_remain), bs):
            err = float((x_remain[k:k+bs] ** 2).sum()) * self.cfg.mse_scale
            mse_vals.append(err)
        return torch_sum(mse_vals) / len(x_remain)

    def apply_book_on_trainset(self, book_K2D, m1, m2):
        codes_combined_B = self.get_combined(m1, m2)
        bs = 2**14
        for k in range(0, len(codes_combined_B), bs):
            self.trainset_BD[k:k+bs] -= book_K2D[codes_combined_B[k:k+bs]]

    def map_codes(self, codes_MB, ivf_codes):
        assert len(ivf_codes.shape) == 1
        codes_MB = torch.concat([codes_MB, self.ivf_code_map[ivf_codes].T])
        comb_codes_MB = codes_MB[self.combine_mvals_m[0]] * self.K_base + codes_MB[self.combine_mvals_m[1]]
        return comb_codes_MB

    def train_make_ivf_small_codesbooks(self):
        IVF_K = self.cfg.ivf_K
        ivf_mean_count = torch.zeros(IVF_K, device=self.device).scatter_add_(0, self.ivf_codes, torch.ones(len(self.trainset_BD), device=self.device))

        self.acc_print(f"{self.ivf_centroids.shape=}")
        self.acc_print(f"Error before IVF optim:", {(self.ivf_centroids**2).sum(-1).mean()})
        for ivf_m in range(self.IVF_M):
            remain_std_D = self.ivf_centroids.std(axis=0)
            remain_avg_D = self.ivf_centroids.mean(axis=0)

            book_Chat_KD = torch.randn((self.K_base, self.D), device=self.device) * remain_std_D + remain_avg_D
            book_Gmap_K2 = torch.zeros((IVF_K), device=self.device, dtype=int)

            # Simple initialization: simply select random points
            probas = (ivf_mean_count / ivf_mean_count.sum()).cpu().numpy()
            init_k_idx = np.random.choice(IVF_K, size=self.K_base, replace=False, p=probas)
            book_Chat_KD = self.ivf_centroids[init_k_idx]

            for cur_iter in range(self.ITER_STEPS):
                has_assignement_changed = False
                bs = 2**14
                for i_bs in range(0, IVF_K, bs):
                    new_segment = approx_pairwise_distance(self.ivf_centroids[i_bs:i_bs+bs], book_Chat_KD).argmin(-1)
                    if not has_assignement_changed and (new_segment != book_Gmap_K2[i_bs:i_bs+bs]).any():
                        has_assignement_changed = True
                    book_Gmap_K2[i_bs:i_bs+bs] = new_segment
                sum_book_Chat_KD = torch.zeros((self.K_base, self.D), device=self.device).index_add_(0, book_Gmap_K2, self.ivf_centroids * ivf_mean_count.unsqueeze(1))
                counts_Chat_K = torch.zeros(self.K_base, device=self.device).scatter_add_(0, book_Gmap_K2, ivf_mean_count)
                book_Chat_KD = sum_book_Chat_KD / counts_Chat_K.unsqueeze(-1).clamp(min=1)

                if not has_assignement_changed:
                    self.acc_print(f"Assignement has not changed between steps, stopping at iteration {cur_iter+1}/{self.ITER_STEPS}")
                    break

            self.ivf_small_codebooks[ivf_m] = book_Chat_KD
            self.ivf_code_map[:,ivf_m] = book_Gmap_K2
            self.ivf_centroids -= book_Chat_KD[book_Gmap_K2]
            uniques = len(book_Gmap_K2.unique())
            self.acc_print(f"Error after IVF optim step {ivf_m}:", {(self.ivf_centroids**2).sum(-1).mean()})
            self.acc_print(f"IVF small codebook usage: {uniques}/{self.K_base}")

        self.train_codes_MB = torch.concat([self.train_codes_MB, self.ivf_code_map[self.ivf_codes].T])
        assert len(self.train_codes_MB) == self.M_base_combined

    def train(self):
        self.train_make_ivf_small_codesbooks()
        self.acc_print(f"Intialize with MSE= {self.compute_mse(self.trainset_BD):.6f}")
        print(f"{self.M=} {self.M_target=}")

        # Create new variables for codebook & codes
        self.codebook_MKD = nn.Parameter(torch.empty((self.M, self.K, self.D), device=self.device), requires_grad=False)

        codes_timer = Timer()
        codes_timer.start()
        for i_new_code in range(self.M):
            self.acc_print(f"\n\n======= [{codes_timer}] Finding new code {i_new_code}")
            # Find m1, m2 candidates
            min_err, best_m1, best_m2, best_book_K2D = float("inf"), None, None, None
            for m1 in range(self.M_base_combined):
                for m2 in range(m1+1, self.M_base_combined):
                    book_K2D, local_error = self.build_combined_codebook(m1, m2, self.trainset_BD)
                    if local_error < min_err:
                        min_err, best_m1, best_m2, best_book_K2D = local_error, m1, m2, book_K2D

            self.acc_print(f"----- [Selects pair {best_m1}-{best_m2}] -----")
            m1, m2, book_K2D = best_m1, best_m2, best_book_K2D # Ensure I don't mix variables later
            self.codebook_MKD[i_new_code] = book_K2D
            self.apply_book_on_trainset(book_K2D, m1, m2)

            self.combine_mvals_m[0][i_new_code] = m1
            self.combine_mvals_m[1][i_new_code] = m2

            usage = (book_K2D.abs().sum(-1) != 0.0).sum().item()
            self.acc_print(f"\n[{codes_timer}] With code {i_new_code}, pair {m1}-{m2}: MSE= {self.compute_mse(self.trainset_BD):.6f} [usage={usage/self.K:.2%}]")
