"""Run from project root: python -m smoke_test"""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from score.dataset import build_pose_features
from score.model import SubsetScoringNet, with_composite
from score.train_scoring_net import calc_loss
from SfM_PPO_env import SceneEnvironment
from SfM_PPO_interaction import Policy, collect, update

def main():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    torch.set_num_threads(2)
    cfg = dict(pose_dim=13, hidden_dim=32, backbone_name="resnet18", view_layers=1, scene_layers=1,
               num_heads=4, dropout=0.)
    scorer = SubsetScoringNet(**cfg, pretrained=False)
    x = torch.randn(1,10,2,3,32,32)
    poses = torch.randn(1,10,2,13)
    scores = with_composite(scorer(x, poses))
    assert scores.shape == (1,10,3)
    assert torch.allclose(scores[...,2], .4*scores[...,0]+.6*scores[...,1])
    loss, _ = calc_loss(scores, torch.rand_like(scores), torch.zeros(1,10,dtype=torch.bool), torch.nn.SmoothL1Loss())
    loss.backward()
    assert scorer.score_head[-1].weight.grad.isfinite().all()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        torch.save(dict(config=cfg,model=scorer.state_dict()),root/"scorer.pt")
        image = root/"view.png"
        Image.fromarray(rng.integers(0,255,(32,32,3),dtype=np.uint8)).save(image)
        ext = np.tile(np.eye(4,dtype=np.float32)[:3],(100,1,1))
        ext[:,:,3] = rng.normal(size=(100,3))
        assert build_pose_features(ext[:25]).shape == (25,13)
        np.savez(root/"views.npz",features=rng.normal(size=(100,397)),extrinsics=ext.reshape(100,12),
                 scene_ids=["a"]*50+["b"]*50,pair_keys=[str(i) for i in range(100)],
                 image_paths=[str(image)]*100,camera_paths=["synthetic"]*100)
        env = SceneEnvironment(root/"views.npz",root/"scorer.pt")
        from score.dataset import _build_transform
        env.img_transform = _build_transform(32,False)
        obs = torch.tensor(env._get_scene_obs("a"),device=env.device)
        policy = Policy(len(obs),actor=(32,),critic=(16,)).to(env.device)
        batch = collect(env,policy,contexts=1)
        assert batch[1].shape == (10,4777) and torch.isfinite(batch[3]).all()
        optimizer = torch.optim.AdamW(policy.parameters(),lr=5e-5)
        old = policy.actor[-1].weight.clone()
        update(policy,optimizer,batch,.006)
        assert not torch.equal(old,policy.actor[-1].weight)
        actions,_,_ = policy.draw(obs)
        indices, scores, _ = env.proposals("a",actions.cpu().numpy())
        assert scores.shape == (10,3) and all(len(set(ids))==25 for ids in indices)
        assert np.isfinite(scores).all()
        print("PASS scorer backward, 10 joint proposals, auxiliary rewards, PPO update, inference")
if __name__ == "__main__":
    main()
