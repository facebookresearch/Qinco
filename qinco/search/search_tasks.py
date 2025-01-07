import json
import math
from pathlib import Path

import faiss
import numpy as np
import torch
from faiss.contrib.evaluation import OperatingPointsWithRanges

from ..datasets import load_queries_data, load_vec_db, load_vec_trainset
from ..metrics import Timer, TimersManager
from ..qinco_tasks import BaseTask
from ..utils import (
    approx_pairwise_distance,
    compute_batch_distances,
    corrected_mean_squared_error,
)
from .optim_order import PARETO_FRONT_SETTINGS
from .pairwise_decoder import PairwiseDecoderIVF
from .search_utils import (
    EncodedDBIterator,
    add_to_ivfaq_index,
    batched_db,
    compute_fixed_aq_codebooks,
    reconstruct_from_fixed_codebooks,
    show_mem,
)


def load_encoded_trainset(cfg):
    assert (
        cfg.encoded_trainset
    ), "Please specify the path to the encoded trainset using the 'encoded_trainset' argument"

    (train_vecs, val_vecs), (train_dataset, val_dataset) = load_vec_trainset(cfg)

    train_codes_it = EncodedDBIterator(cfg, cfg.encoded_trainset)
    trainset_all_codes = train_codes_it.load_all()
    assert cfg.ds.trainset + cfg.ds.valset <= len(
        trainset_all_codes
    ), f"Size of the training ({cfg.ds.trainset}) + validation ({cfg.ds.valset}) set doesn't match the set of the encoded vectors {len(trainset_all_codes)}"

    train_codes = trainset_all_codes[: cfg.ds.trainset]
    val_codes = trainset_all_codes[-cfg.ds.valset :]

    cfg._accelerator.print(
        f"Loaded encoded training vectors of shape {train_codes.shape}"
    )
    cfg._accelerator.print(
        f"Loaded encoded validation vectors of shape {val_codes.shape}"
    )

    return (train_vecs, val_vecs), (train_codes, val_codes)


####################################################################
# Centroids training phase
####################################################################


def train_ivf_centroids(cfg, xt):
    acc_print = cfg._accelerator.print

    acc_print(f"Loading {cfg.ds.trainset} training vectors")
    xt = xt.astype(np.float32)
    acc_print(f"Training set of size {xt.shape=:}")
    d = xt.shape[-1]
    # km = faiss.Kmeans(d, cfg.ivf_K, niter=100, verbose=True, gpu=True)
    km = faiss.Kmeans(d, cfg.ivf_K, niter=100, verbose=True, gpu=False)
    km.train(xt)
    return km.centroids


####################################################################
# Encode training set (1B)
####################################################################


@torch.inference_mode
def encode_database(cfg, model, db_vecs):
    cfg._accelerator.wait_for_everyone()
    assert cfg.output.endswith(".npz")
    output_base_path = cfg.output[:-4]

    acc_print = cfg._accelerator.print
    accelerator = cfg._accelerator

    db_size = len(db_vecs)
    nproc = cfg._accelerator.num_processes
    proc_id = cfg._accelerator.process_index

    acc_print(f"Encoding {db_size} vectors using {nproc} processes")
    timers = TimersManager("encode", "saving")

    all_codes = []

    start_id = (db_size // nproc) * proc_id
    end_id = (db_size // nproc) * (proc_id + 1) if proc_id < nproc - 1 else db_size

    with timers.encode:
        for i0 in range(start_id, end_id, cfg.batch):
            i1 = min(end_id, i0 + cfg.batch)
            batch = db_vecs[i0:i1]
            batch = torch.from_numpy(batch).to(accelerator.device, torch.float32)

            acc_print(
                f"[T_encode={timers.encode}] Encoding batch {(i0-start_id)//cfg.batch+1}/{(end_id-start_id)//cfg.batch+1}"
            )
            codes = model(batch, step="encode").T
            all_codes.append(codes.cpu().numpy())

        acc_print(f"Waiting for all encoding processes to be done...")
        cfg._accelerator.wait_for_everyone()
        acc_print(f"Encoding done in {timers.encode}")

    with timers.saving:
        acc_print("Saving encoded vectors...")
        if cfg._accelerator.is_main_process:
            np.savez_compressed(cfg.output, n_parts=nproc, K=cfg.K, M=cfg.M, D=cfg._D)

        all_codes = np.concatenate(all_codes)
        print(
            f"Storing encoding of size {all_codes.shape} for database part {proc_id+1}/{nproc} to file {output_base_path + f'.{proc_id}.npz'}"
        )
        np.savez_compressed(output_base_path + f".part_{proc_id}.npz", codes=all_codes)

        acc_print(f"Waiting for all encoded files to be saved...")
        cfg._accelerator.wait_for_everyone()
        acc_print(
            f"Stored codes into {cfg.output} and {nproc} associated part files [done in {timers.saving}]"
        )


####################################################################
# Create faiss index
####################################################################


def pair_codes_ivf(codes, K2):
    K = int(K2**0.5)
    ivf_codes, codes = codes[:, :1], codes[:, 1:]
    M = codes.shape[-1]
    codes = codes.reshape((-1, 2, M // 2))
    codes = codes[:, 0] * K + codes[:, 1]
    return np.concatenate([ivf_codes, codes], axis=-1)


@torch.inference_mode
def build_index_training_phase(cfg, ivf_codebook, vec_data, encoded_data):
    acc_print = cfg._accelerator.print
    b_timer = Timer()
    K = cfg.K

    with b_timer:
        if cfg._pair_codes:
            acc_print(f"[{b_timer}] Pairing training codes")
            K = K**2
            encoded_data = pair_codes_ivf(encoded_data, K)

        assert vec_data.dtype == np.float32
        acc_print(
            f"[{b_timer}] TLoaded IVF centroids of shape {ivf_codebook.shape} from model"
        )

        acc_print(f"[{b_timer}] Training fixed codebooks on residuals")

        acc_print(f"[{b_timer}] Computing residuals...")
        max_nt = min(len(vec_data), cfg.search.aq_training_samples)
        xt_residuals = vec_data - ivf_codebook[encoded_data[:, 0]]
        acc_print(
            f"[{b_timer}] Training codebooks with AQ as approximate quantizer... (using {max_nt}/{len(vec_data)} data points)"
        )
        codebooks = compute_fixed_aq_codebooks(
            xt_residuals[:max_nt], encoded_data[:max_nt, 1:], k=K
        )
        acc_print(f"[{b_timer}] Training done, codebooks of shape {codebooks.shape}")

        xt_fixed_recons = reconstruct_from_fixed_codebooks(
            encoded_data[:, 1:], codebooks
        )
        acc_print(
            f"[{b_timer}] Computed fixed reconstructions {xt_fixed_recons.shape=}"
        )

        MSE = corrected_mean_squared_error(cfg, xt_fixed_recons, xt_residuals)
        acc_print(f"[{b_timer}] MSE with fixed codebooks on training set: {MSE:g}")

        # not really useful for _Nfloat encoding and even less for ST_decompress
        norms = ((xt_fixed_recons - xt_residuals) ** 2).sum(-1)
        acc_print(f"[{b_timer}] Computed norms")

        acc_print(f"[{b_timer}] Construct the index", cfg.search.index_key)
        index = faiss.index_factory(vec_data.shape[-1], cfg.search.index_key)
        quantizer = faiss.downcast_index(index.quantizer)

        quantizer.hnsw.efConstruction = 20
        acc_print(
            f"[{b_timer}] Set quantizer efConstruction to",
            quantizer.hnsw.efConstruction,
        )

        acc_print(f"[{b_timer}] Setting IVF centroids and RQ codebooks")
        acc_print(f"[{b_timer}] IVF centroids")
        assert ivf_codebook.shape[0] == index.nlist
        quantizer.add(ivf_codebook)
        acc_print(f"[{b_timer}] Set codebook")
        assert codebooks.shape[0] == index.rq.M
        assert codebooks.shape[2] == index.rq.d
        rq_Ks = list(2 ** faiss.vector_to_array(index.rq.nbits))
        assert rq_Ks == [codebooks.shape[1]] * index.rq.M
        faiss.copy_array_to_vector(codebooks.ravel(), index.rq.codebooks)
        acc_print(f"[{b_timer}] Train norms")
        index.rq.train_norm(len(norms), faiss.swig_ptr(norms))

        index.rq.is_trained = True
        index.is_trained = True
        acc_print(f"[{b_timer}] Index ready")

    return index


@torch.inference_mode
def build_index_adding_phase(cfg, encoded_db, index):
    acc_print = cfg._accelerator.print
    timers = TimersManager("adding")

    quantizer = faiss.downcast_index(index.quantizer)
    codebooks = faiss.vector_to_array(index.rq.codebooks)
    k = 1 << index.rq.nbits.at(0)
    M = index.rq.M
    codebooks = codebooks.reshape(M, k, cfg._D)

    if len(cfg.search.quantizer_efSearch) > 0:
        ef = cfg.search.quantizer_efSearch[0]
        acc_print(f"Set quantizer efSearch to {ef}")
        quantizer.hnsw.efSearch = ef

    with timers.adding:
        BS = 1_000_000
        acc_print(f"[{timers.adding}] Adding codes")
        for xb_batch in encoded_db.iter(BS):
            if cfg._pair_codes:
                xb_batch = pair_codes_ivf(xb_batch, k)
            assert xb_batch.shape[1] == M + 1
            acc_print(
                f"[{timers.adding}] Adding batch {encoded_db.cur_i_batch}/{encoded_db.total_n_batches}"
                f" ({encoded_db.batch_start_id}:{encoded_db.batch_end_id} over {encoded_db.n_samples})"
            )
            xb_fixed_recons = reconstruct_from_fixed_codebooks(
                xb_batch[:, 1:], codebooks
            )
            xb_norms = (xb_fixed_recons**2).sum(1)
            acc_print(f"[{timers.adding}] Reconstructed, adding to index")
            add_to_ivfaq_index(
                index,
                xb_batch[:, 1:],
                xb_batch[:, 0],
                xb_norms,
                i_base=encoded_db.batch_start_id,
            )

    acc_print(f"Adding encoded vectors to index done in {timers}")


####################################################################
# Searching
####################################################################


def compute_recalls(I, gt):
    assert I.ndim == 2 and gt.ndim == 2
    recalls = {}
    for rank in [1, 10, 100]:
        recall = (I[:, :rank] == gt[:, :1]).sum() / gt.shape[0]
        recalls[rank] = float(recall)
    return recalls


def sort_experiments_pareto_front(op, experiments):
    expe_tab = []
    for i_exp, cno in enumerate(experiments):
        key = op.cno_to_key(cno)
        cur_p = op.get_parameters(key)
        min_dist = float("inf")
        for p2 in PARETO_FRONT_SETTINGS:
            dist = sum(
                [
                    abs(math.log2(max(1, p2[key])) - math.log2(max(1, cur_p[key])))
                    for key in p2.keys()
                ]
            )
            min_dist = min(dist, min_dist)
        expe_tab.append((min_dist, i_exp, cno))
    expe_tab.sort()

    experiments = [e[-1] for e in expe_tab]
    return experiments


@torch.inference_mode
def run_search_ivf(cfg, qinco_model, index, xq, gt):
    acc_print = cfg._accelerator.print
    accelerator = cfg._accelerator

    if cfg.pairwise_decoder:
        mid_reranker = PairwiseDecoderIVF(cfg, load=True)
        acc_print(f"Will use middle reranker {mid_reranker}")
        mid_reranker = torch.jit.script(mid_reranker)
    else:
        mid_reranker = None

    seen_parameters, ivf_real_res = [], []
    json_results = {
        "ivf_real_res": ivf_real_res,
    }
    if cfg.resume:
        if Path(cfg.output).exists():
            with open(cfg.output, "r") as sf:
                json_results = json.load(sf)
            ivf_real_res = json_results["ivf_real_res"]
            seen_parameters = [r["parameters"] for r in ivf_real_res]
            acc_print(
                f"Resuming from previous experiment results {cfg.output} (results of {len(seen_parameters)} experiments)"
            )
        else:
            acc_print(f"Can't find file {cfg.output} to resume from")

    if cfg.output:
        acc_print(f"Will write search results to {cfg.output}")
    else:
        acc_print("No outpout file specified, results will only be displayed in logs")

    quantizer = faiss.downcast_index(index.quantizer)

    device = accelerator.device
    acc_print(f"Putting model on device {device}")
    qinco_model = qinco_model.to(device)

    index.parallel_mode = 3
    acc_print("Loading queries")
    xq_distances = torch.from_numpy(xq).to(device, torch.float32)

    acc_print(f"queries {xq.shape=:} ground-truth {gt.shape=:}")

    cc = index.coarse_code_size()
    cc1 = index.sa_code_size()
    nq, d = xq.shape
    M = cfg.M  # M without IVF
    acc_print("Start experiments")

    op = OperatingPointsWithRanges()
    op.add_range("nprobe", list(cfg.search.nprobe))
    if len(cfg.search.quantizer_efSearch) > 0:
        op.add_range("quantizer_efSearch", list(cfg.search.quantizer_efSearch))
    op.add_range("nshort", list(cfg.search.nshort))
    nmid_short_list = list(cfg.search.nmid_short) if mid_reranker is not None else [0]
    op.add_range("nmid_short", nmid_short_list)

    experiments = op.sample_experiments(0, rs=np.random.RandomState(123))
    acc_print(
        f"Total nb experiments {op.num_experiments()}, running {len(experiments)}"
    )

    # Sort experiments depeding on probability of being on Pareto-optimal front
    experiments = sort_experiments_pareto_front(op, experiments)

    # Warmup model
    with Timer() as t:
        qinco_model(
            torch.randn((cfg.batch, cfg._D), dtype=torch.float32, device=device),
            step="encode",
        )
        qinco_model(
            torch.zeros((cfg._M_ivf, cfg.search.batch_size), dtype=int, device=device),
            step="decode",
        )
    print(f"Warmup in {t.ms()}")

    for i_exp, cno in enumerate(experiments):
        key = op.cno_to_key(cno)
        parameters = op.get_parameters(key)
        acc_print(f"-")
        acc_print(f"Experiment {i_exp+1}/{len(experiments)} {key}")

        if parameters in seen_parameters:
            for result in ivf_real_res:
                if result["key"] == list(key):
                    break
            else:
                raise Exception(f"Can't find result with {key=}")
            op.add_operating_point(key, result["recalls"]["1"], result["t_total"])
            acc_print(f"Loaded from previous checkpoint")
            continue
        seen_parameters.append(parameters)

        acc_print(f"{cno=:4d} {str(parameters):50}")
        exp_timers = TimersManager("search", "mid_rerank", "decode", "rerank")

        # Get parameters
        index.nprobe = parameters["nprobe"]
        nshort = parameters["nshort"]
        nmid_short = parameters["nmid_short"] * nshort
        if "quantizer_efSearch" in parameters:
            quantizer.hnsw.efSearch = parameters["quantizer_efSearch"]

        (max_perf, min_time) = op.predict_bounds(key)
        if not op.is_pareto_optimal(max_perf, min_time):
            acc_print(f"SKIP, {max_perf=:.3f} {min_time=:.3f}")
            continue
        ##### Part 1: Use IVF to get a shortlist of codes #####
        n_short_ivf = min(max(nmid_short, nshort), 8000)

        with exp_timers.search:
            D, I, codes = index.search_and_return_codes(
                xq, n_short_ivf, include_listnos=True
            )
            I = torch.from_numpy(I).to(device)
            codes = torch.from_numpy(codes)

        ##### Part 2: Map splitted IVF codes to full codes #####
        with exp_timers.decode:
            codes2 = codes.reshape(n_short_ivf * nq, cc1).to(device, torch.int32)
            codes_int32 = torch.zeros(
                (n_short_ivf * nq, M + 1), dtype=torch.int32, device=device
            )
            if cc == 2:
                codes_int32[:, 0] = codes2[:, 0] | (codes2[:, 1] << 8)
            elif cc == 3:
                codes_int32[:, 0] = (
                    codes2[:, 0] | (codes2[:, 1] << 8) | (codes2[:, 2] << 16)
                )
            else:
                raise NotImplementedError

            # to avoid decode errors on -1 (missing shortlist result)
            codes_int32[:, 0] = torch.clip(codes_int32[:, 0], min=0, max=cfg.ivf_K - 1)

            # Codebook codes
            codes_int32[:, 1:] = codes2[:, cc : M + cc]

        ##### Part 3: Re-rank using approximate decoding #####
        if nshort < n_short_ivf:  # Using nmid_short in previous steps
            ivf_book = qinco_model.qinco_model.steps[0].ivf_centroids.weight
            codes_int32_T = codes_int32.T
            _ = mid_reranker(codes_int32_T[1:], codes_int32_T[0])  # Warmup

            with exp_timers.mid_rerank:
                shortlist = mid_reranker(codes_int32_T[1:], codes_int32_T[0])
                shortlist += ivf_book[codes_int32[:, 0]]
                shortlist = shortlist.reshape(nq, n_short_ivf, d)

                D_refined = compute_batch_distances(
                    xq_distances.reshape(nq, 1, d), shortlist, approx=True
                ).reshape(nq, n_short_ivf)
                idx = torch.argsort(D_refined, axis=1)
                codes_refined = torch.take_along_dim(
                    codes_int32.reshape(nq, n_short_ivf, M + 1),
                    idx[:, :nshort, None],
                    dim=1,
                )
                I_refined = torch.take_along_dim(I, idx[:, :nshort], dim=1)
                _ = I_refined[-1][-1].item()  # Forces CUDA synchronisation

            I = I_refined
            codes_int32 = codes_refined.reshape(nq * nshort, M + 1)

        ##### Part 4: Decode codes from IVF #####
        # Ensure the QINCo model is properly optimized with jit and warmup
        _ = qinco_model.decode(
            codes_int32[: len(codes_int32) % cfg.search.batch_size].T
        )
        _ = qinco_model.decode(codes_int32[: cfg.search.batch_size].T)  # Warmup

        ### Decode the codes using QINCo
        with exp_timers.decode:
            with torch.no_grad():
                shortlist = []
                for i in range(0, len(codes_int32), cfg.search.batch_size):
                    code_batch = codes_int32[i : i + cfg.search.batch_size]
                    x_batch = qinco_model.decode(code_batch.T)
                    shortlist.append(x_batch)

            _ = shortlist[-1][-1, -1].cpu().numpy()  # Forces CUDA synchronisation

        if (
            nshort == 1
        ):  # We don't use rerank if shortlise is of size 1 (we could just not execute the previous functions ; this snippet is to minimize code complexity)
            exp_timers.decode.reset()
            exp_timers.decode.reset()

        ##### Part 5: Re-rank the entries using QINCo model #####
        with exp_timers.rerank:
            shortlist_t = torch.concatenate(shortlist).reshape(nq, nshort, d)

            D_refined = compute_batch_distances(
                xq_distances.reshape(nq, 1, d), shortlist_t, approx=True
            ).reshape(nq, nshort)

            idx = torch.argsort(D_refined, axis=1)
            I_refined = torch.take_along_dim(I, idx[:, :100], dim=1)
            I_refined = I_refined.cpu().numpy()  # Forces CUDA synchronisation

        ##### END: Compute recall and time stats #####
        recalls_orig = compute_recalls(I.cpu().numpy(), gt)
        recalls = compute_recalls(I_refined, gt)

        acc_print(
            f"Achieved R@1={recalls[1]*100:.2} R@10={recalls[10]*100:.2} R@100={recalls[100]*100:.2} in {exp_timers.sum().s()}"
        )
        acc_print(
            f"Timers: search={exp_timers.search.s()} + mid_rerank={exp_timers.mid_rerank.s()} + decode={exp_timers.decode.s()} + rerank={exp_timers.rerank.s()} (Total={exp_timers.sum().s()})"
        )
        acc_print(f"Recalls: {recalls}  |  Recalls without QINCo step: {recalls_orig}")

        total_time = exp_timers.sum().get()
        op.add_operating_point(key, recalls[1], total_time)

        ivf_real_res.append(
            dict(
                parameters=parameters,
                cno=cno,
                t_search=exp_timers.search.get(),
                t_mid_rerank=exp_timers.mid_rerank.get(),
                t_decode=exp_timers.decode.get(),
                t_rerank=exp_timers.rerank.get(),
                t_total=exp_timers.sum().get(),
                recalls=recalls,
                recalls_orig=recalls_orig,
                key=list(map(int, key)),
            )
        )
        # acc_print("Added result", ivf_real_res[-1])
        if cfg.output:
            with open(cfg.output, "w") as sf:
                json.dump(json_results, sf)

    return json_results


#####################################################################
# Full search (no IVF) over 1M vectors
#####################################################################


@torch.inference_mode
def run_search_full_direct_small_db(cfg, qinco_model, db, queries_QD, gt_Q):
    acc_print = cfg._accelerator.print
    device = cfg._accelerator.device
    qinco_model = qinco_model.to(device)

    show_mem()

    with Timer() as search_timer:
        ###### Load queries & dataset ######
        acc_print(f"[{search_timer}] Loading queries")
        queries_QD = torch.from_numpy(queries_QD).to(device, torch.float32)
        gt_Q = torch.from_numpy(gt_Q)

        ###### Encode database ######
        acc_print(f"[{search_timer}] Encoding database")

        encoded_dataset_ND = []
        for i_batch, batch_BD in batched_db(db, cfg.batch):
            if i_batch % 10 == 0:
                acc_print(
                    f"[{search_timer}] Encoding database, batch {i_batch+1}/{len(db)//cfg.batch}"
                )
            batch_BD = torch.from_numpy(batch_BD).to(device, torch.float32)
            codes_MB = qinco_model(batch_BD, step="encode")
            xhat_BD = qinco_model(codes_MB, step="decode")
            encoded_dataset_ND.append(xhat_BD.cpu())

        encoded_dataset_ND = torch.concat(encoded_dataset_ND, dim=0).to(device)
        acc_print(f"[{search_timer}] Encoding done")

        ###### Find top vectors for each queries ######
        acc_print(f"[{search_timer}] Computing top query answers")
        bs = min(100, cfg.batch)
        nshort = 100
        shortlists_QN = []
        for i_batch, batch_queries_QD in batched_db(queries_QD, bs):
            if i_batch % 10 == 0:
                acc_print(
                    f"[{search_timer}] Computing distances to queries, batch {i_batch+1}/{len(queries_QD)//bs}"
                )
            dists_QN = approx_pairwise_distance(
                batch_queries_QD.unsqueeze(1), encoded_dataset_ND
            ).squeeze(1)
            closest_vecs_loc_idx_qN = dists_QN.argsort(dim=-1)[:, :nshort]
            shortlists_QN.append(closest_vecs_loc_idx_qN.cpu())
        shortlists_QN = torch.concat(shortlists_QN)

        ###### Compute retrieval score ######
        acc_print(f"[{search_timer}] Computing Recall")
        recalls = compute_recalls(shortlists_QN, gt_Q)
        recalls = {k: f"{v*100:.2f}" for k, v in recalls.items()}
        acc_print(f"R@1={recalls[1]}    R@10={recalls[10]}    R@100={recalls[100]}")


#####################################################################
# Search tasks
#####################################################################


class BaseSearchTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self):
        super().setup()

        if self.cfg.search.nthreads != -1:
            self.accelerator.print(
                f"set number of threads to {self.cfg.search.nthreads}"
            )
            faiss.omp_set_num_threads(self.cfg.search.nthreads)
            torch.set_num_threads(self.cfg.search.nthreads)

    def init_config(self):
        super().init_config()
        self.cfg._cpu_model = [
            l for l in open("/proc/cpuinfo", "r") if "model name" in l
        ][0]

        self.cfg._cuda_devices = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

        if self.accelerator.device.type == "cpu":
            self.accelerator.print(
                f"CPU, using a batch size of {self.cfg.search.batch_size} instead of {self.cfg.search.batch_size}"
            )
            self.cfg.search.batch_size = self.cfg.search.batch_size

        self.cfg._pair_codes = False


class IVFTrainTask(BaseSearchTask):
    USE_QINCO_MODEL = False

    def setup(self):
        super().setup()
        assert (
            self.accelerator.num_processes == 1
        ), "IVF centroids training should use a single process"
        assert (
            self.cfg.output is not None
        ), "Please specify the outpout path to store the IVF centroids using the 'output' argument"
        assert self.cfg.output.endswith(
            ".npy"
        ), "Please specify a .npy file for the 'output' argument"

    def load_data(self):
        self.cfg._accelerator.print(f"Loading training data from {self.cfg.trainset}")
        (self.train_vecs, self.val_vecs), (self.train_dataset, self.val_dataset) = (
            load_vec_trainset(self.cfg)
        )
        self.cfg._accelerator.print(f"Training set: {self.train_vecs.shape}")

    def run(self):
        ivf_centroids = train_ivf_centroids(self.cfg, self.train_vecs)

        self.accelerator.print(f"Storing IVF centroids to {self.cfg.output}")
        np.save(self.cfg.output, ivf_centroids)


class EncodeDBTask(BaseSearchTask):
    def setup(self):
        super().setup()
        assert (
            self.cfg.output is not None
        ), "Please specify a path to store the pairwise decoder using the 'output' argument"
        assert self.cfg.output.endswith(
            ".npz"
        ), "Please specify a .npz file for the 'output' argument"

    def load_data(self):
        if self.cfg.encode_trainset:
            (train_vecs, val_vecs), _ = load_vec_trainset(self.cfg)
            self.db_vecs = np.concatenate((train_vecs, val_vecs), axis=0)

            self.cfg._accelerator.print(
                f"Training + validation sets: {self.db_vecs.shape}"
            )
        else:
            self.cfg._accelerator.print(f"Loading database from {self.cfg.db}")
            self.db_vecs, _ = load_vec_db(self.cfg)
            self.cfg._accelerator.print(f"Database: {self.db_vecs.shape}")

    def run(self):
        return encode_database(self.cfg, self.qinco_model, self.db_vecs)


class BuildIndexTask(BaseSearchTask):
    USE_QINCO_MODEL = False

    def setup(self):
        super().setup()
        assert (
            self.accelerator.num_processes == 1
        ), "Build-index should use a single GPU and a single process"
        assert self.cfg.ivf_centroids, "Please specify a path to the IVF centroids"
        assert (
            self.cfg.encoded_db
        ), "Please specify the path to the encoded database using the 'encoded_db' argument"
        assert (
            self.cfg.output
        ), "Please specify a path to store the pairwise decoder using the 'output' argument"
        assert self.cfg.output.endswith(
            ".faissindex"
        ), "Please specify a .faissindex file for the 'output' argument"

    def load_data(self):
        (self.train_vecs, self.val_vecs), (self.train_codes, self.val_codes) = (
            load_encoded_trainset(self.cfg)
        )
        self.encoded_db = EncodedDBIterator(self.cfg, self.cfg.encoded_db)
        self.ivf_centroids = self.cfg._ivf_centroids_preloaded.astype(np.float32)

    def run(self):
        index = build_index_training_phase(
            self.cfg, self.ivf_centroids, self.train_vecs, self.train_codes
        )
        build_index_adding_phase(self.cfg, self.encoded_db, index)
        self.accelerator.print(f"Saving index to {self.cfg.output}")
        faiss.write_index(index, self.cfg.output)


class TrainPairwiseDecoderTask(BaseSearchTask):
    USE_QINCO_MODEL = False

    def setup(self):
        super().setup()
        assert (
            self.accelerator.num_processes == 1
        ), "Reranker training should use a single GPU and a single process"
        assert (
            self.cfg.ivf_centroids
        ), "Please specify a path to the IVF centroids using the 'ivf_centroids' argument"
        assert (
            self.cfg.encoded_trainset
        ), "Please specify the path to the encoded trainset using the 'encoded_trainset' argument"
        assert (
            self.cfg.output
        ), "Please specify a path to store the pairwise decoder using the 'output' argument"
        assert self.cfg.output.endswith(
            ".pt"
        ), "Please specify a .pt file for the 'output' argument"

    def load_data(self):
        (self.train_vecs, self.val_vecs), (self.train_codes, self.val_codes) = (
            load_encoded_trainset(self.cfg)
        )
        self.train_codes, self.val_codes = self.train_codes.T, self.val_codes.T

    @torch.inference_mode()
    def run(self):
        acc_print = self.cfg._accelerator.print

        ivf_centroids = torch.from_numpy(self.cfg._ivf_centroids_preloaded)
        train_codes_MB = torch.from_numpy(self.train_codes[1:]).to(
            self.accelerator.device
        )
        ivf_codes_B = torch.from_numpy(self.train_codes[0]).to(self.accelerator.device)
        assert self.train_vecs.dtype == np.float32
        trainset_BD_past_ivf = (
            self.train_vecs - ivf_centroids.numpy()[self.train_codes[0]]
        )
        trainset_BD = torch.from_numpy(trainset_BD_past_ivf.copy()).to(
            self.accelerator.device
        )
        acc_print(f"Train with {train_codes_MB.shape=} {trainset_BD.shape=}")

        val_codes_MB = torch.from_numpy(self.val_codes[1:]).to(self.accelerator.device)
        ivf_val_codes_B = torch.from_numpy(self.val_codes[0]).to(
            self.accelerator.device
        )
        assert self.val_vecs.dtype == np.float32
        valset_BD_past_ivf = self.val_vecs - ivf_centroids.numpy()[self.val_codes[0]]
        valset_BD = torch.from_numpy(valset_BD_past_ivf.copy()).to(
            self.accelerator.device
        )
        acc_print(f"Val with {val_codes_MB.shape=} {valset_BD.shape=}")

        acc_print(
            "Min / max train_codes_MB:", train_codes_MB.min(), train_codes_MB.max()
        )
        estimated_mse_0 = (trainset_BD[: 2**14] ** 2).sum(
            -1
        ).mean() * self.cfg.mse_scale
        estimated_val_mse_0 = (valset_BD[: 2**14] ** 2).sum(
            -1
        ).mean() * self.cfg.mse_scale

        reranker = PairwiseDecoderIVF(
            cfg=self.cfg,
            train_codes_MB=train_codes_MB,
            trainset_BD=trainset_BD,
            load=False,
            ivf_centroids=ivf_centroids.to(self.accelerator.device),
            ivf_codes=ivf_codes_B,
            valset_BD=valset_BD,
            valset_BD_past_ivf=valset_BD_past_ivf,
            val_codes_MB=val_codes_MB,
        )

        estimated_mse_1 = (trainset_BD[: 2**14] ** 2).sum(
            -1
        ).mean() * self.cfg.mse_scale
        acc_print(
            f"Estimated train_MSE={estimated_mse_1:.3f} (starting from train_MSE={estimated_mse_0:.3f}, val_MSE={estimated_val_mse_0:.3f})"
        )
        acc_print(f"Saving re-ranker to {self.cfg.output}")
        torch.save(reranker.state_dict(), self.cfg.output)

        acc_print(
            f"After training: train_MSE={reranker.compute_ds_mse(trainset_BD_past_ivf, train_codes_MB, ivf_codes_B):.10f} (computed on training set)"
        )
        acc_print(
            f"After training: val_MSE={reranker.compute_ds_mse(valset_BD_past_ivf, val_codes_MB, ivf_val_codes_B):.10f} (computed on test set)"
        )

        reranker = PairwiseDecoderIVF(self.cfg, load=self.cfg.output)
        acc_print(
            f"MSE after loading: {reranker.compute_ds_mse(trainset_BD_past_ivf, train_codes_MB, ivf_codes_B):.10f} (computed on training set)"
        )


class SearchTask(BaseSearchTask):
    def setup(self):
        super().setup()

        if self.cfg.index:
            assert (
                self.accelerator.num_processes == 1
            ), "Search within an index should use a single process"
            assert (
                self.accelerator.device.type == "cpu"
            ), f"Search within an index should only be done on CPU for accurate timing. Current device is {self.accelerator.device}"
            self.accelerator.print(f"Will search in index {self.cfg.index}")

            if self.cfg.output:
                assert self.cfg.output.endswith(
                    ".json"
                ), "Please specify a .json file for the 'output' argument"
        elif self.cfg.db:
            assert (
                self.accelerator.num_processes == 1
            ), "Search should use a single GPU and a single process"
            self.accelerator.print(
                f"No index used, will search within full database {self.cfg.db}"
            )
        else:
            raise Exception(
                "Please specify either the 'db' argument for full database search, or the 'index' argument forlarge-scale search with optimized faiss index"
            )

    def load_data(self):
        self.xq, self.xq_gt = load_queries_data(self.cfg)

        if self.cfg.index:
            self.accelerator.print(f"Reading index from {self.cfg.index}")
            self.faiss_index = faiss.read_index(self.cfg.index)
        else:
            self.accelerator.print(f"Reading database from {self.cfg.db}")
            self.db, _ = load_vec_db(self.cfg)
            self.cfg._accelerator.print(f"Database: {self.db.shape}")

    def run(self):
        if self.cfg.index:
            search_results = run_search_ivf(
                self.cfg, self.qinco_model, self.faiss_index, self.xq, self.xq_gt
            )
            print("JSON results:", json.dumps(search_results))
        else:
            run_search_full_direct_small_db(
                self.cfg, self.qinco_model.eval(), self.db, self.xq, self.xq_gt
            )
