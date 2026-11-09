"""Normalize planner SVG through the same DiffVG importer used by Canvas/Painter."""
import argparse, json, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import numpy as np
from pydiffvg.parse_svg import parse_transform
import pydiffvg
NS='{http://www.w3.org/2000/svg}'
def normalize(source, output):
    assert not output.exists(), output
    pydiffvg.set_use_gpu(False)
    w,h,shapes,groups=pydiffvg.svg_to_scene(str(source))
    assert w==h and len(shapes)==len(groups)
    scale=256.0/w
    transforms=[]
    def visit(node, inherited):
        matrix=inherited @ np.asarray(parse_transform(node.get('transform')))
        if node.tag==NS+'path': transforms.append(matrix)
        for child in node: visit(child,matrix)
    visit(ET.parse(source).getroot(),np.eye(3))
    assert len(transforms)==len(shapes)
    stroke_scales=[float(np.sqrt(abs(np.linalg.det(m[:2,:2])))) for m in transforms]
    for index,(shape,group) in enumerate(zip(shapes,groups)):
        assert isinstance(shape,pydiffvg.Path) and len(shape.points)==4
        assert shape.num_control_points.tolist()==[2] and not shape.is_closed
        assert group.shape_ids.tolist()==[index] and group.fill_color is None
        assert torch.allclose(group.stroke_color,torch.tensor([0.,0.,0.,1.]))
        shape.points=shape.points*scale
        shape.stroke_width=shape.stroke_width*scale*stroke_scales[index]
    output.parent.mkdir(parents=True,exist_ok=True)
    pydiffvg.save_svg(str(output),256,256,shapes,groups)
    root=ET.parse(output).getroot()
    root.set('viewBox','0 0 256 256')
    paths=root.findall('.//'+NS+'path')
    assert len(paths)==len(shapes)
    for i,path in enumerate(paths):path.set('id',f'path_{i}')
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    return dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),paths=len(paths),
                scale=scale,stroke_policy='scale inherited scalar stroke by sqrt(abs(det(transform))); anisotropic stroke approximated, centerline unchanged',stroke_scales=stroke_scales,
                geometry='existing pydiffvg.svg_to_scene bakes inherited transforms/styles')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();print(json.dumps(normalize(args.source,args.output)))
