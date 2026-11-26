"""Score registered point clouds in metres. Use the ICP diagnostics from registration."""
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree


def score(pred, reference, core, fitness, inlier_rmse):
    if fitness < .1 or len(pred) == 0:
        return dict(global_q_geo_raw=0., crack_core_q_geo_raw=0., q_final=0., reliability=0.)
    def distances(a, b):
        return cKDTree(b).query(a)[0]
    def rmse(d):
        return np.sqrt(np.mean(d**2))
    def norm(error, limit):
        return 1 - np.clip(error / limit, 0, 1)
    acc, comp = distances(pred, reference), distances(reference, pred)
    precision, recall = (acc <= .01).mean(), (comp <= .01).mean()
    f10 = 2*precision*recall/max(precision+recall, 1e-12)
    global_q = .3*norm(rmse(acc),.02) + .3*norm(rmse(comp),.02) + .4*f10
    near = distances(pred, core)
    local = near[near <= .01]
    missing = distances(core, pred)
    core_q = (.3*norm(rmse(missing),.01) + .25*(missing <= .003).mean()
              + .2*(missing <= .005).mean())
    if len(local):
        core_q += .15*norm(rmse(local),.01) + .1*(local <= .005).mean()
    base = .4 if fitness < .3 else (.75 if fitness < .5 else 1.)
    reliability = base*np.clip(1-.5*(inlier_rmse/.01-1),.5,1)
    return dict(global_q_geo_raw=global_q, crack_core_q_geo_raw=core_q,
                q_final=(.4*global_q+.6*core_q)*reliability, reliability=reliability)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("pred","reference","core"):
        p.add_argument("--"+key, required=True, help="Registered N x 3 NPY array, metres")
    p.add_argument("--fitness",required=True,type=float)
    p.add_argument("--inlier-rmse",required=True,type=float)
    a = p.parse_args()
    print(json.dumps(score(np.load(a.pred),np.load(a.reference),np.load(a.core),a.fitness,a.inlier_rmse)))
