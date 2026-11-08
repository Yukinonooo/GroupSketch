"""Prepare an LLM plan through existing Canvas grouping and Painter export."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import pydiffvg
import yaml
import app as canvas
from animate_svg import parse_arguments
from motion_prior import compile_plan
from painter import Painter
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--caption",required=True)
    parser.add_argument("--num-frames",type=int,required=True)
    parser.add_argument("--variant",choices=["affine","translation"],required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists(), "do not overwrite a prepared input"
    plan = json.loads(args.plan.read_text())
    prior = compile_plan(plan,args.svg,args.num_frames,args.variant)
    root = ET.parse(args.svg).getroot()
    assert root.get("viewBox") == "0 0 256 256"
    # Input is already preprocessed into cubic paths, as used for LLM context.
    _,_,shapes,_ = pydiffvg.svg_to_scene(str(args.svg))
    assert len(shapes) == sum(len(g["path_ids"]) for g in plan["groups"])
    assert all(len(s.points)==4 and list(s.num_control_points)==[2] for s in shapes)
    file_id = "llm-plan"
    tree = ET.ElementTree(copy.deepcopy(root))
    canvas.app.config["AFTER_MOVE_FOLDER"] = str(out.parent)
    canvas.PROJECT_ROOT = str(out/".canvas-work")
    canvas.svg_files[file_id] = {"filename":out.name+".svg","path":str(args.svg),"tree":tree,"root":tree.getroot(),"namespace":"http://www.w3.org/2000/svg"}
    canvas.groups_info[file_id] = []
    canvas.frame_positions[file_id] = {}
    client = canvas.app.test_client()
    for group in plan["groups"]:
        response = client.post("/create_group",json={"file_id":file_id,"group_name":group["name"],"path_ids":group["path_ids"]})
        assert response.status_code == 200, response.get_data(as_text=True)
    response = client.post("/generate_animation",json={"file_id":file_id,"total_frames":args.num_frames})
    assert response.status_code == 200, response.get_data(as_text=True)
    prior_path = out/"input/affine-prior.yaml"
    prior_path.write_text(yaml.safe_dump(prior,sort_keys=False))
    (out/"input/planner-output.json").write_bytes(args.plan.read_bytes())
    # Reuse the real parser, initialization and scene exporter; no SDS request here.
    sys.argv = ["animate_svg.py","--target",str(out/"input"),"--caption",args.caption,
                "--num_frames",str(args.num_frames),"--output_folder","llm-prior-preview/"+out.name,
                "--affine_prior",str(prior_path)]
    cfg = parse_arguments()
    cfg.device = torch.device("cpu")
    cfg.optim_points_with_mlp = False
    pydiffvg.set_use_gpu(False)
    painter = Painter(cfg,str(out/"input/postprocess"),args.num_frames,cfg.device)
    with torch.no_grad():
        frames,scene,_ = painter.render_frames_to_tensor_mlp(apply_mlp=False)
    utils.save_vid_svg(scene,str(Path(cfg.output_folder)/"svg_logs"),0,painter.canvas_width,painter.canvas_height)
    utils.save_mp4_from_tensor(frames,str(Path(cfg.output_folder)/"coarse.mp4"))
    utils.save_hq_video(cfg.output_folder,iter_=0)
    # Coarse frame SVGs used by DINO originate from exactly the same initial geometry.
    exported = Path(cfg.output_folder)/"svg_logs/svg_step0"
    errors = []
    for frame in range(args.num_frames):
        svg = exported/f"frame{frame:03d}.svg"
        _,_,reloaded,_ = pydiffvg.svg_to_scene(str(svg))
        expected = torch.cat([s.points for g in sorted(scene) for s in scene[g][frame][0]])
        actual = torch.cat([s.points for s in reloaded])
        error = (expected-actual).abs().max().item()
        assert error < 1e-3, error
        errors.append(error)
        shutil.copy2(svg,out/f"frame-{frame+1}.svg")
    if (out/".canvas-work").exists():
        shutil.rmtree(out/".canvas-work")
    (out/"frame-1_cleaned.svg").unlink(missing_ok=True)
    evidence = {"variant":args.variant,"num_frames":args.num_frames,"groups":len(plan["groups"]),
        "source_svg_sha256":hashlib.sha256(args.svg.read_bytes()).hexdigest(),
        "plan_sha256":hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "sampled_transforms_nonsingular":True,"coarse_and_painter_point_error_max":max(errors),
        "preview":str(Path(cfg.output_folder).resolve()),"affine_prior":str(prior_path),
        "source_entry":"Canvas create_group/generate_animation -> Painter initialization -> utils.save_vid_svg/save_hq_video"}
    (out/"preparation.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(json.dumps(evidence,ensure_ascii=False))
if __name__ == "__main__":
    main()
