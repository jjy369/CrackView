import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from SfM_PPO_env import SceneEnvironment
from SfM_PPO_interaction import Policy, collect, update


def digest(path):
    with open(path, "rb") as stream:
        h = hashlib.sha256()
        for data in iter(lambda: stream.read(2**20), b""):
            h.update(data)
        return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer"])
    parser.add_argument("--features", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", default="selected_views.json", type=Path)
    parser.add_argument("--rollouts", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ck = None
    if args.mode == "infer" or args.resume:
        ck = torch.load(args.policy, map_location="cpu", weights_only=False)
        if ck["scorer"] != digest(args.scorer):
            raise ValueError("Scorer differs from the policy training checkpoint")
    elif args.policy.exists():
        raise FileExistsError("Choose a new policy path or use --resume")
    env = SceneEnvironment(args.features, args.scorer, seed=args.seed, scaling=ck["scaling"] if ck else None)
    policy = Policy(len(env._get_scene_obs(env.all_scene_ids[0]))).to(env.device)
    if ck:
        policy.load_state_dict(ck["policy"])
    if args.mode == "infer":
        policy.eval()
        result = {}
        for sid in sorted(env.all_scene_ids):
            obs = torch.as_tensor(env._get_scene_obs(sid), device=env.device)
            actions, _, _ = policy.draw(obs)
            ids, scores, _ = env.proposals(sid, actions.cpu().numpy())
            winner = int(scores[:, 2].argmax())
            keys = [env.scene_pair_keys[sid][i] for i in ids[winner]]
            result[sid] = dict(winner=winner, scores=scores.tolist(), views=[
                dict(image=env.image_path_dict[k], camera=env.cam_path_dict[k]) for k in keys])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
        return
    if len(env.all_scene_ids) < 2:
        raise ValueError("Training requires a main and an auxiliary scene")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)
    fingerprint = digest(args.features)
    scorer_hash = digest(args.scorer)
    state_keys = ("baselines","moments","_scene_visit_n","_scene_delta_ema","_scene_q_ema")
    start = 0
    if ck:
        if fingerprint != ck["features"] or args.rollouts != ck["total"]:
            raise ValueError("Resume requires the same training features and schedule")
        start = ck["rollout"]
        optimizer.load_state_dict(ck["optimizer"])
        env.rng.bit_generator.state = ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if env.device == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        for key in state_keys:
            getattr(env,key).update(ck["env"][key])
    env.global_total, env.episode_idx = 32 * args.rollouts, 32 * start
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    for i in range(start, args.rollouts):
        batch = collect(env, policy)
        progress = i / max(args.rollouts - 1, 1)
        entropy = .006 + max(0., (progress - .1) / .9) * (.0004 - .006)
        update(policy, optimizer, batch, entropy)
        if (i+1) % 5 == 0 or i+1 == args.rollouts:
            state = dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), rollout=i+1,
                         total=args.rollouts, scaling=env.scaling, features=fingerprint, scorer=scorer_hash,
                         env={key:dict(getattr(env,key)) for key in state_keys}, numpy_rng=env.rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            temp = args.policy.with_suffix(".tmp")
            torch.save(state, temp)
            os.replace(temp, args.policy)
        print(f"Rollout {i+1}/{args.rollouts}, mean reward {batch[3].mean().item():.4f}", flush=True)


if __name__ == "__main__":
    main()
