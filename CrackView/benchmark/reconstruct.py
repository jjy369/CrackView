"""Run with Metashape Professional 2.x Python after view selection.

Coordinates of exported points are in the chunk frame. Register them to the
metric reference before calculating distances. Run full and subset jobs with
the same installation and parameters.
"""
import argparse
import json
from pathlib import Path


def reconstruct(images, destination):
    import Metashape as ms
    doc = ms.Document()
    chunk = doc.addChunk()
    chunk.addPhotos([str(Path(path).resolve(strict=True)) for path in images])
    if len(chunk.sensors) != 1:
        raise ValueError("The paper assumes one shared sensor; check image calibration groups.")
    chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False)
    chunk.alignCameras(adaptive_fitting=True)
    if sum(camera.transform is not None for camera in chunk.cameras) < 2:
        return False
    chunk.optimizeCameras(adaptive_fitting=True)
    chunk.buildDepthMaps(downscale=2, filter_mode=ms.MildFiltering)
    chunk.buildPointCloud()
    chunk.exportPointCloud(path=str(destination), source_data=ms.PointCloudData,
                           format=ms.PointCloudFormatPLY)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selected", required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for index,(sid,scene) in enumerate(json.loads(args.selected.read_text(encoding="utf8")).items()):
        destination = args.out / f"scene_{index:04d}.ply"
        ok = reconstruct([v["image"] for v in scene["views"]],destination)
        report.append(dict(scene=sid, aligned=ok, cloud=str(destination) if ok else None))
    (args.out/"reconstructions.json").write_text(json.dumps(report,indent=2), encoding="utf8")
