"""LLM affine initialization adapter; shared by preparation and Painter."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml

NS = "{http://www.w3.org/2000/svg}"

def compile_plan(plan, svg_path, num_frames, variant="affine"):
    assert variant in ("affine", "translation") and num_frames >= 2
    root = ET.parse(svg_path).getroot()
    ids = [p.get("id") for p in root.findall(".//" + NS + "path")]
    assert len(ids) == len(set(ids)) and all(ids)
    groups = plan["groups"]
    assert groups and len({g["id"] for g in groups}) == len(groups)
    assigned = [i for g in groups for i in g["path_ids"]]
    assert len(assigned) == len(set(assigned)) and set(assigned) == set(ids), "incomplete path partition"
    assert plan["canvas"] == {"width":256, "height":256}
    result = []
    for group in groups:
        assert group["path_ids"]
        pivot = np.asarray(group["pivot"], float)
        keys = group["keyframes"]
        times = np.asarray([k["t"] for k in keys], float)
        assert pivot.shape == (2,) and np.isfinite(pivot).all()
        assert len(keys) >= 2 and times[0] == 0 and times[-1] == 1 and (np.diff(times) > 0).all()
        values = np.asarray([[*k["translation"], k["rotation_deg"], *k["scale"], *k["shear"]] for k in keys], float)
        assert values.shape == (len(keys), 7) and np.isfinite(values).all()
        assert np.allclose(values[0], [0,0,0,1,1,0,0], atol=1e-8), "nonidentity first frame"
        assert (values[:,3:5] > 0).all()
        states = np.stack([np.interp(np.linspace(0,1,num_frames),times,values[:,i]) for i in range(7)],axis=1)
        matrices = []
        for dx,dy,degrees,sx,sy,hx,hy in states:
            if variant == "translation":
                linear = np.eye(2)
            else:
                a = np.deg2rad(degrees)
                rotation = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                linear = rotation @ np.array([[1,hx],[hy,1]]) @ np.diag([sx,sy])
            matrix = np.eye(3)
            matrix[:2,:2] = linear
            matrix[:2,2] = pivot + [dx,dy] - linear @ pivot
            assert np.isfinite(matrix).all() and abs(np.linalg.det(linear)) > 1e-6, "singular sampled transform"
            matrices.append(matrix.tolist())
        result.append({"id":group["id"],"path_ids":group["path_ids"],"matrices":matrices})
    return {"schema_version":1,"num_frames":num_frames,"variant":variant,"groups":result}

def load_affine_prior(path, num_frames, svg_path):
    data = yaml.safe_load(Path(path).read_text())
    assert data["schema_version"] == 1 and data["num_frames"] == num_frames
    root = ET.parse(svg_path).getroot()
    svg_groups = root.findall(".//" + NS + "g")
    assert len(svg_groups) == len(data["groups"]), "group count mismatch"
    for group, record in zip(svg_groups, data["groups"]):
        assert [p.get("id") for p in group.findall(NS+"path")] == record["path_ids"], "group path order mismatch"
    matrices = np.asarray([g["matrices"] for g in data["groups"]], dtype=np.float32)
    assert matrices.shape == (len(svg_groups),num_frames,3,3) and np.isfinite(matrices).all()
    assert np.allclose(matrices[:,:,2,:],[0,0,1])
    assert np.allclose(matrices[:,0],np.eye(3),atol=1e-6)
    assert (np.abs(np.linalg.det(matrices[:,:,:2,:2])) > 1e-6).all()
    return matrices
