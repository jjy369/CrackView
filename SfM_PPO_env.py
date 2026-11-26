"""Scene observations, joint subset scoring and proposal-specific bandit rewards."""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from score.dataset import build_pose_features, _build_transform
from score.model import SubsetScoringNet, with_composite


class SceneEnvironment:
    def __init__(self, features, scorer, seed=2024, scaling=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = np.random.default_rng(seed)
        self.scaling = scaling
        self.n_heads, self.select_k, self.n_path_bins = 12, 25, 10
        self.n_fps_anchors = 6
        self.use_geo_obs = self.use_difficulty_obs = self.use_image_obs = True
        self.use_selection_memory = self.use_crack_obs = self.use_curriculum = True
        self.mmr_selection = True
        self.mmr_div_scale, self.head_score_temperature, self.head_union_weight = 3., .60, .50
        self.coverage_bonus_weight, self.angle_bonus_weight, self.sharpness_bonus_weight = .075, .04, .04
        self.div_tau = .92
        self.curriculum_end_frac, self.curriculum_alpha = .55, .1
        self.global_total = self.episode_idx = 0
        self._scene_delta_ema = {}
        self._scene_visit_n = defaultdict(int)
        self._scene_q_ema = defaultdict(lambda: np.zeros(2, np.float32))
        self.baselines, self.moments = {}, {}
        self.img_transform = _build_transform(224, False)
        self._load_npz(features)
        self._validate_scenes()
        ck = torch.load(scorer, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        args = {k: cfg[k] for k in ("pose_dim", "hidden_dim", "backbone_name", "view_layers",
                                     "scene_layers", "num_heads", "dropout")}
        self.model = SubsetScoringNet(**args, out_dim=2, pretrained=False).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval().requires_grad_(False)

    def _load_npz(self, npz_path: str) -> None:
        print(f"📦 加载特征文件: {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)

        pair_keys   = npz["pair_keys"].tolist()
        scene_ids   = npz["scene_ids"].tolist()
        image_paths = npz["image_paths"].tolist()
        cam_paths   = npz["camera_paths"].tolist()
        features    = npz["features"].astype(np.float32)
        extrinsics  = npz["extrinsics"].astype(np.float32)

        self.d = int(features.shape[1])

        if self.d != 397:
            raise ValueError("R3 requires 397 features per view")
        if len(set(pair_keys)) != len(pair_keys) or not np.isfinite(features).all():
            raise ValueError("Duplicate view keys or nonfinite features")
        self._has_crack_feats = False
        if self.scaling is None:
            self.scaling = (features.mean(0), features.std(0))
        self._feat_mean, self._feat_std = self.scaling
        self._feat_std = np.where(self._feat_std < 1e-6, 1., self._feat_std)
        self.feature_dict:    Dict[str, np.ndarray] = {}
        self.image_path_dict: Dict[str, str]        = {}
        self.cam_path_dict:   Dict[str, str]        = {}
        self.extrinsic_dict:  Dict[str, np.ndarray] = {}
        self.scene_pair_keys: Dict[str, List[str]]  = defaultdict(list)

        for pk, sid, ip, cp, feat, ext in zip(
            pair_keys, scene_ids, image_paths, cam_paths, features, extrinsics,
        ):
            self.feature_dict[pk]    = feat
            self.image_path_dict[pk] = str(ip)
            self.cam_path_dict[pk]   = str(cp)
            self.extrinsic_dict[pk]  = ext
            self.scene_pair_keys[sid].append(pk)

        self.all_scene_ids = list(self.scene_pair_keys.keys())

        print("   预计算场景特征矩阵、obs 统计、FPS 锚点...")
        self._scene_feat_mat:        Dict[str, np.ndarray] = {}
        self._scene_feat_mat_policy: Dict[str, np.ndarray] = {}
        self._scene_feat_mat_norm:   Dict[str, np.ndarray] = {}
        self._scene_ext_mat:         Dict[str, np.ndarray] = {}
        self._scene_img_obs:         Dict[str, np.ndarray] = {}
        self._scene_crack_obs:       Dict[str, np.ndarray] = {}
        self._scene_anchor_obs:      Dict[str, np.ndarray] = {}
        self._scene_centers:         Dict[str, np.ndarray] = {}
        self._scene_center_unit:     Dict[str, np.ndarray] = {}
        self._scene_dirs:            Dict[str, np.ndarray] = {}
        self._scene_path_u:          Dict[str, np.ndarray] = {}
        self._scene_perp_u:          Dict[str, np.ndarray] = {}
        self._scene_perp_valid:      Dict[str, bool]       = {}
        self._scene_quality_prior:   Dict[str, np.ndarray] = {}

        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            if a.size < 2 or a.std() < 1e-8 or b.std() < 1e-8:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        for sid in self.all_scene_ids:
            self.scene_pair_keys[sid].sort()
            keys  = self.scene_pair_keys[sid]
            X     = np.stack([self.feature_dict[pk] for pk in keys]).astype(np.float32)
            X_policy = np.clip(
                ((X - self._feat_mean) / self._feat_std).astype(np.float32),
                -5.0, 5.0
            )
            dino = X[:, :384].astype(np.float32)
            X_n  = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)
            ext  = np.stack([self.extrinsic_dict[pk] for pk in keys]).astype(np.float32)
            N    = X.shape[0]

            centers = X[:, 388:391].astype(np.float32)
            dirs    = X[:, 391:394].astype(np.float32)
            dirs    = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
            c_min   = centers.min(0, keepdims=True)
            c_span  = centers.max(0, keepdims=True) - c_min
            c_span[c_span < 1e-6] = 1.0
            center_unit = (centers - c_min) / c_span

            path_u = np.zeros(N, dtype=np.float32)
            perp_u = np.zeros(N, dtype=np.float32)
            perp_valid = False
            geom_eigval_ratios = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if N >= 2:
                C0 = centers - centers.mean(0, keepdims=True)
                try:
                    _, svals, vh = np.linalg.svd(C0, full_matrices=False)
                    proj   = C0 @ vh[0].astype(np.float32)
                    p_span = float(proj.max() - proj.min())
                    if p_span >= 1e-6:
                        path_u = ((proj - proj.min()) / p_span).astype(np.float32)

                    if vh.shape[0] > 1:
                        perp_raw  = C0 @ vh[1].astype(np.float32)
                        perp_span = float(perp_raw.max() - perp_raw.min())
                        if perp_span > 1e-6 and (
                            len(svals) < 2 or svals[1] / (svals[0] + 1e-8) > 0.015
                        ):
                            perp_u     = ((perp_raw - perp_raw.min()) / perp_span).astype(np.float32)
                            perp_valid = True

                    sv_sum = float(svals.sum()) + 1e-8
                    r1 = float(svals[0]) / sv_sum
                    r2 = float(svals[1]) / sv_sum if len(svals) > 1 else 0.0
                    r3 = float(svals[2]) / sv_sum if len(svals) > 2 else 0.0
                    geom_eigval_ratios = np.array([r1, r2, r3], dtype=np.float32)
                except Exception:
                    pass

            sharp, exposure, contrast, sat = (
                X[:, 384], X[:, 385], X[:, 386], X[:, 387]
            )

            def _z(v):
                return (v - v.mean()) / (v.std() + 1e-6)

            quality_prior = .60 * _z(sharp) + .25 * _z(contrast) - .30 * _z(sat)
            quality_prior = _z(quality_prior).astype(np.float32)

            self._scene_feat_mat[sid]        = X
            self._scene_feat_mat_policy[sid] = X_policy
            self._scene_feat_mat_norm[sid]   = X_n
            self._scene_ext_mat[sid]         = ext
            self._scene_centers[sid]         = centers
            self._scene_center_unit[sid]     = center_unit.astype(np.float32)
            self._scene_dirs[sid]            = dirs
            self._scene_path_u[sid]          = path_u
            self._scene_perp_u[sid]          = perp_u
            self._scene_perp_valid[sid]      = bool(perp_valid)
            self._scene_quality_prior[sid]   = quality_prior

            # ── image_obs（18 维） ────────────────────────────────────
            q_feat  = X[:, 384:388].astype(np.float64)
            q_std   = q_feat.std(0).astype(np.float32)
            q_min   = q_feat.min(0).astype(np.float32)
            iu      = np.triu_indices(N, k=1)
            sim     = (X_n @ X_n.T).astype(np.float64)
            sv      = sim[iu] if len(iu[0]) else np.array([0.0], dtype=np.float64)
            sim_obs = np.array([sv.mean(), sv.std(), sv.min()], dtype=np.float32)
            fwd_n   = dirs.astype(np.float32)
            av      = (fwd_n @ fwd_n.T)[iu] if len(iu[0]) else np.array([1.0], dtype=np.float32)
            ang_obs = np.array([av.mean(), av.std()], dtype=np.float32)
            quality_pcts = np.percentile(quality_prior, [0, 25, 50, 75, 100]).astype(np.float32)
            self._scene_img_obs[sid] = np.concatenate(
                [q_std, q_min, sim_obs, ang_obs, quality_pcts]
            )

            # ── crack_obs（19 维 baseline + 可选 5 维裂缝场景统计） ───
            n_bins_obs = max(2, int(self.n_path_bins))
            bins_all = np.clip((path_u * n_bins_obs).astype(int), 0, n_bins_obs - 1)
            bin_sharp, bin_quality = [], []
            for b in range(n_bins_obs):
                m = bins_all == b
                if m.any():
                    bin_sharp.append(float(sharp[m].mean()))
                    bin_quality.append(float(quality_prior[m].mean()))
            if not bin_sharp:
                bin_sharp   = [float(sharp.mean())]
                bin_quality = [float(quality_prior.mean())]
            bin_sharp   = np.asarray(bin_sharp,   dtype=np.float32)
            bin_quality = np.asarray(bin_quality, dtype=np.float32)
            dir_spread = float(
                np.mean(np.linalg.norm(dirs - dirs.mean(0, keepdims=True), axis=1))
            )
            crack_obs_base = np.array([
                float(sharp.mean()), float(sharp.std()),
                float(sharp.min()),  float(sharp.max()),
                float(contrast.mean()), float(contrast.std()),
                float(sat.mean()),   float(sat.max()),
                _safe_corr(path_u, sharp),
                _safe_corr(path_u, quality_prior),
                float(bin_sharp.min()),   float(bin_sharp.max()),
                float(bin_sharp.std()),   float(bin_quality.std()),
                1.0 if perp_valid else 0.0,
                dir_spread,
            ], dtype=np.float32)

            self._scene_crack_obs[sid] = np.concatenate([crack_obs_base, geom_eigval_ratios])

            # ── FPS 路径锚点 ──────────────────────────────────────────
            if self.n_fps_anchors > 0:
                k_anc   = min(self.n_fps_anchors, N)
                targets = np.linspace(0.0, 1.0, k_anc)
                sel_anc = []
                for tt in targets:
                    order = np.argsort(np.abs(path_u - tt))
                    pick  = next(
                        (int(i) for i in order if int(i) not in sel_anc),
                        int(order[0])
                    )
                    sel_anc.append(pick)
                while len(sel_anc) < self.n_fps_anchors:
                    sel_anc.append(sel_anc[-1])
                self._scene_anchor_obs[sid] = (
                    X_policy[sel_anc[:self.n_fps_anchors]].flatten().astype(np.float32)
                )

        n_pairs = sum(len(v) for v in self.scene_pair_keys.values())
        print(f"   场景={len(self.all_scene_ids)}  pair={n_pairs}  D={self.d}  "
              f"crack={'on' if self._has_crack_feats else 'off'}")

    def _validate_scenes(self) -> None:
        if not self.all_scene_ids:
            raise ValueError("NPZ 中没有任何场景数据。")
        bad = [s for s in self.all_scene_ids
               if len(self.scene_pair_keys[s]) != 50]
        if bad:
            raise ValueError(
                f"{len(bad)} 个场景未覆盖完整的 50 个视图: "
                + ", ".join(bad[:3])
            )

    # ════════════════════════════════════════════════════════════════════
    # 图像访问
    # ════════════════════════════════════════════════════════════════════

    def _score_with_action(
        self,
        action: np.ndarray,
        X: Optional[np.ndarray] = None,
        return_heads: bool = False,
    ) -> Tuple:
        if X is None:
            X = self._X_policy
        D, H   = self.d, self.n_heads
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        heads  = action[:H * D].reshape(H, D)
        betas  = action[H * D: H * D + H]

        heads_n = heads / (np.linalg.norm(heads, axis=1, keepdims=True) + 1e-8)
        hs      = X @ heads_n.T
        hs      = (hs - hs.mean(0, keepdims=True)) / (hs.std(0, keepdims=True) + 1e-6)

        bs      = betas - betas.max()
        weights = np.exp(bs) / (np.exp(bs).sum() + 1e-8)

        mix     = hs @ weights
        tau     = max(float(self.head_score_temperature), 1e-3)
        logits  = (hs + np.log(weights.reshape(1, -1) + 1e-8)) / tau
        zmax    = logits.max(axis=1, keepdims=True)
        union   = tau * (zmax[:, 0] + np.log(
            np.exp(logits - zmax).sum(axis=1) + 1e-8
        ))
        a           = float(np.clip(self.head_union_weight, 0.0, 1.0))
        base_scores = ((1.0 - a) * mix + a * union).astype(np.float32)

        if self.mmr_selection and len(action) > H * D + H:
            raw_div          = float(action[H * D + H])
            diversity_weight = float(self.mmr_div_scale / (1.0 + np.exp(-raw_div)))
        else:
            diversity_weight = 0.0

        if return_heads:
            return base_scores, hs.astype(np.float32), weights.astype(np.float32), diversity_weight
        return base_scores, diversity_weight

    # ════════════════════════════════════════════════════════════════════
    # Greedy MMR
    # ════════════════════════════════════════════════════════════════════

    def _greedy_mmr_select(
        self,
        base_scores: np.ndarray,
        X_norm: np.ndarray,
        k: int,
        diversity_weight: float,
    ) -> np.ndarray:
        N = base_scores.shape[0]
        k = min(k, N)

        selected:    List[int] = []
        remaining:   List[int] = list(range(N))
        max_sim_to_S = np.full(N, -np.inf, dtype=np.float32)

        for step in range(k):
            rem_arr = np.array(remaining)
            if step == 0 or diversity_weight < 1e-6:
                best_local = int(np.argmax(base_scores[rem_arr]))
            else:
                mmr_scores = base_scores[rem_arr] - diversity_weight * max_sim_to_S[rem_arr]
                best_local = int(np.argmax(mmr_scores))
            best_idx = remaining[best_local]

            selected.append(best_idx)
            remaining.pop(best_local)

            if diversity_weight > 1e-6 and remaining:
                new_sims = X_norm[remaining] @ X_norm[best_idx]
                for li, ri in enumerate(remaining):
                    if new_sims[li] > max_sim_to_S[ri]:
                        max_sim_to_S[ri] = new_sims[li]

        return np.array(selected, dtype=np.int64)

    @staticmethod
    def _path_max_gap(u_values: np.ndarray) -> float:
        if len(u_values) == 0:
            return 1.0
        us   = np.sort(np.asarray(u_values, dtype=np.float32))
        gaps = [float(us[0]), float(1.0 - us[-1])]
        if len(us) > 1:
            gaps.extend(np.diff(us).astype(float).tolist())
        return float(max(gaps))

    def _policy_select(
        self, action: np.ndarray, scene_id: str, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        base_scores, diversity_weight = self._score_with_action(
            action_arr,
            X=self._scene_feat_mat_policy[scene_id],
        )

        if self.mmr_selection:
            top_idx = self._greedy_mmr_select(
                base_scores,
                self._scene_feat_mat_norm[scene_id],
                k,
                diversity_weight,
            )
        else:
            top_idx = np.argsort(base_scores)[-k:].astype(np.int64)

        return top_idx, base_scores

    # ════════════════════════════════════════════════════════════════════
    # 批量 reward model 评分
    # ════════════════════════════════════════════════════════════════════

    def _build_tensors(
        self, pair_keys: List[str], cache: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = torch.stack([cache[pk] for pk in pair_keys])
        ext_mats = []
        for pk in pair_keys:
            e     = self.extrinsic_dict[pk].reshape(3, 4)
            m     = np.eye(4, dtype=np.float32)
            m[:3] = e
            ext_mats.append(m)
        poses = torch.tensor(build_pose_features(ext_mats).astype(np.float32))
        return imgs, poses

    def _get_scene_obs(self, scene_id: str) -> np.ndarray:
        X   = self._scene_feat_mat_policy[scene_id]
        obs = np.concatenate([X.mean(0), X.var(0)])
        if self.use_geo_obs:
            t   = self._scene_centers[scene_id]
            obs = np.concatenate([obs, t.mean(0), t.var(0), t.max(0) - t.min(0)])
        if self.use_difficulty_obs:
            obs = np.concatenate([obs, [
                float(self._scene_delta_ema.get(scene_id, 0.0)),
                float(min(self._scene_visit_n[scene_id] / 50.0, 1.0)),
            ]])
        if self.use_image_obs:
            obs = np.concatenate([obs, self._scene_img_obs[scene_id]])
        if self.use_crack_obs:
            obs = np.concatenate([obs, self._scene_crack_obs[scene_id]])
        if self.use_selection_memory:
            obs = np.concatenate([obs, self._scene_q_ema[scene_id]])
        if self.n_fps_anchors > 0 and scene_id in self._scene_anchor_obs:
            obs = np.concatenate([obs, self._scene_anchor_obs[scene_id]])
        return obs.astype(np.float32)

    # ════════════════════════════════════════════════════════════════════
    # 裂缝专项奖励
    # ════════════════════════════════════════════════════════════════════

    def _crack_bonus(
        self,
        top_idx: np.ndarray,
        scene_id: str,
        update_last: bool = True,
    ) -> Tuple[float, float, float]:
        X   = self._scene_feat_mat[scene_id]
        idx = np.asarray(top_idx, dtype=np.int64)
        if idx.size == 0:
            return 0.0, 0.0, 0.0

        sharp_all  = X[:, 384]
        sharp_sel  = float(sharp_all[idx].mean())
        sharp_gain = (sharp_sel - float(sharp_all.mean())) / (float(sharp_all.std()) + 1e-6)
        sharp_bonus = self.sharpness_bonus_weight * float(np.tanh(sharp_gain))

        centers    = self._scene_centers[scene_id]
        path_u     = self._scene_path_u[scene_id]
        perp_u     = self._scene_perp_u[scene_id]
        perp_valid = self._scene_perp_valid.get(scene_id, False)
        u_sel      = path_u[idx]

        path_span      = float(u_sel.max() - u_sel.min()) if len(idx) > 1 else 0.0
        endpoint_score = (
            float(1.0 - 0.5 * (u_sel.min() + (1.0 - u_sel.max())))
            if len(idx) > 0 else 0.0
        )
        endpoint_score = float(np.clip(endpoint_score, 0.0, 1.0))

        n_bins  = max(2, int(self.n_path_bins))
        bins    = np.clip((u_sel * n_bins).astype(int), 0, n_bins - 1)
        counts  = np.bincount(bins, minlength=n_bins).astype(np.float32)
        bin_cov = float(np.count_nonzero(counts) / min(n_bins, len(idx)))
        balance = float(np.clip(1.0 - counts.std() / (counts.mean() + 1e-6), 0.0, 1.0))

        max_gap        = self._path_max_gap(u_sel)
        path_gap_score = float(np.clip(1.0 - max_gap / 0.20, 0.0, 1.0))

        pos_sel    = centers[idx]
        span_sel   = float((pos_sel.max(0) - pos_sel.min(0)).max()) if len(idx) > 1 else 0.0
        span_all   = float((centers.max(0) - centers.min(0)).max()) + 1e-8
        spatial_span = float(np.clip(span_sel / span_all, 0.0, 1.0))

        perp_span  = (
            float(perp_u[idx].max() - perp_u[idx].min())
            if perp_valid and len(idx) > 1 else 0.0
        )

        dirs = self._scene_dirs[scene_id][idx]
        if len(idx) > 1:
            iu  = np.triu_indices(len(idx), k=1)
            cos = np.clip(dirs @ dirs.T, -1.0, 1.0)
            dir_angle_div = float(np.mean(np.arccos(cos[iu]) / np.pi))
            xy   = np.stack([u_sel, perp_u[idx] if perp_valid else np.zeros_like(u_sel)], axis=1)
            xy   = xy - xy.mean(0, keepdims=True)
            norm = np.linalg.norm(xy, axis=1, keepdims=True)
            valid_ = norm[:, 0] > 1e-6
            if valid_.sum() > 1:
                unit = xy[valid_] / norm[valid_]
                c2   = np.clip(unit @ unit.T, -1.0, 1.0)
                iu2  = np.triu_indices(unit.shape[0], k=1)
                baseline_angle_div = float(np.mean(np.arccos(c2[iu2]) / np.pi))
            else:
                baseline_angle_div = 0.0
        else:
            dir_angle_div = baseline_angle_div = 0.0

        view_div = float(np.clip(
            max(dir_angle_div, 0.55 * baseline_angle_div + 0.45 * perp_span),
            0.0, 1.0
        ))
        coverage = float(np.clip(
            0.24 * endpoint_score + 0.18 * bin_cov + 0.26 * path_gap_score
            + 0.17 * balance + 0.15 * spatial_span,
            0.0, 1.0
        ))
        cov_bonus = (
            self.coverage_bonus_weight * (coverage - 0.72)
            + self.angle_bonus_weight  * (view_div - 0.35)
        )

        if update_last:
            self._last_crack_metrics = {
                "sharpness_mean": sharp_sel,
                "sharpness_gain": float(sharp_gain),
                "path_span": path_span,
                "path_endpoint_score": endpoint_score,
                "path_bin_coverage": bin_cov,
                "path_balance": balance,
                "path_max_gap": float(max_gap),
                "path_gap_score": path_gap_score,
                "spatial_span": spatial_span,
                "perp_span": perp_span,
                "dir_angle_diversity": dir_angle_div,
                "baseline_angle_diversity": baseline_angle_div,
                "angle_diversity": view_div,
                "view_diversity": view_div,
                "angle_bonus": self.angle_bonus_weight * (view_div - 0.35),
            }
        return float(sharp_bonus), float(cov_bonus), float(coverage)

    def _redundancy_score(
        self,
        sel_keys: List[str],
        pair_keys_list: Optional[List[str]] = None,
        X_norm: Optional[np.ndarray] = None,
    ) -> float:
        if len(sel_keys) <= 1:
            return 0.0
        pl  = pair_keys_list if pair_keys_list is not None else self._pair_keys_list
        xn  = X_norm if X_norm is not None else self._X_norm
        k2i = {pk: i for i, pk in enumerate(pl)}
        idx = [k2i[pk] for pk in sel_keys]
        X   = xn[idx]
        sim = X @ X.T
        k   = len(sel_keys)
        iu  = np.triu_indices(k, k=1)
        return float(np.maximum(0.0, sim[iu] - self.div_tau).mean())

    # ════════════════════════════════════════════════════════════════════
    # 自适应 MC 基线
    # ════════════════════════════════════════════════════════════════════

    def _sample_scene(self) -> str:
        if not self.use_curriculum or not self._scene_delta_ema:
            return str(self.rng.choice(self.all_scene_ids))
        weights = np.ones(len(self.all_scene_ids), dtype=np.float32)
        for i, sid in enumerate(self.all_scene_ids):
            if sid not in self._scene_delta_ema:
                weights[i] = 1.5
            else:
                e = self._scene_delta_ema[sid]
                weights[i] = 0.4 if e < -0.25 else (2.5 if e < 0.08 else 1.0)
        if self.global_total > 0:
            frac = min(1.0, self.episode_idx / max(self.global_total, 1))
            if frac > self.curriculum_end_frac:
                d = (frac - self.curriculum_end_frac) / max(
                    1.0 - self.curriculum_end_frac, 1e-8
                )
                weights = weights * (1 - d) + np.ones_like(weights) * d
        weights /= weights.sum()
        return self.all_scene_ids[self.rng.choice(len(self.all_scene_ids), p=weights)]

    def _update_curriculum(self, scene_id: str, delta: float) -> None:
        if scene_id not in self._scene_delta_ema:
            self._scene_delta_ema[scene_id] = delta
        else:
            a = self.curriculum_alpha
            self._scene_delta_ema[scene_id] = (
                a * delta + (1 - a) * self._scene_delta_ema[scene_id]
            )


    @torch.no_grad()
    def score_group(self, sid, indices, cache):
        keys = self.scene_pair_keys[sid]
        pairs = [self._build_tensors([keys[j] for j in ids], cache) for ids in indices]
        images, poses = [torch.stack(group).unsqueeze(0).to(self.device) for group in zip(*pairs)]
        return with_composite(self.model(images, poses))[0].cpu().numpy()

    def proposals(self, sid, actions):
        keys = self.scene_pair_keys[sid]
        cache = {}
        for key in keys:
            with Image.open(self.image_path_dict[key]) as image:
                cache[key] = self.img_transform(image.convert("RGB"))
        indices = [self._policy_select(action, sid, 25)[0] for action in actions]
        return indices, self.score_group(sid, indices, cache), cache

    def rewards(self, sid, actions):
        indices, scores, cache = self.proposals(sid, actions)
        keys = self.scene_pair_keys[sid]
        random_sets = [self.rng.choice(len(keys), 25, replace=False) for _ in range(4)]
        mc = self.score_group(sid, random_sets, cache)[:, 2].mean()
        visits = self._scene_visit_n[sid]
        rate = 1 / (visits + 1) if visits < 8 else .1
        baseline = self.baselines.get(sid, mc)
        baseline += rate * (mc - baseline)
        self.baselines[sid] = baseline
        delta = scores[:, 2] - baseline
        shaped = delta.copy()
        for i, ids in enumerate(indices):
            sharp, coverage, _ = self._crack_bonus(ids, sid, False)
            redundancy = self._redundancy_score([keys[j] for j in ids], keys, self._scene_feat_mat_norm[sid])
            shaped[i] += sharp + coverage - .03 * redundancy
        n, mean, m2, ema = self.moments.get(sid, (0, 0., 0., 0.))
        # Merge all ten samples simultaneously. Permuting proposals cannot change their rewards.
        count = len(shaped)
        batch_mean = float(shaped.mean())
        diff = batch_mean - mean
        m2 += float(((shaped - batch_mean)**2).sum()) + diff**2 * n * count / (n + count)
        mean += diff * count / (n + count)
        n += count
        center = ema if visits else batch_mean
        rate = 1 / (visits + 1) if visits < 6 else .07
        self.moments[sid] = (n, mean, m2, ema + rate * (batch_mean - ema))
        self._scene_visit_n[sid] += 1
        self._scene_q_ema[sid] = .85 * self._scene_q_ema[sid] + .15 * scores[:, :2].mean(0)
        self._update_curriculum(sid, float(delta.mean()))
        return np.clip((shaped - center) / (np.sqrt(m2 / max(n-1, 1)) + 1e-8), -2., 2.)
