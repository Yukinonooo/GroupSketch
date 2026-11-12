"""验证删网而非置零、保留网络初始化一致、原双分支兼容和优化器归属。无需GPU或模型API。"""
import json,subprocess,types,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
from models import PointMLP
from painter import Painter, PainterOptimizer
root=Path.cwd();original={"__name__":"frozen_models"};exec(compile(subprocess.check_output(['git','show','bd4891a:models.py'],text=True),'frozen_models','exec'),original)

def networks(cls):
 return [cls(input_dim=4*n*2,inter_dim=16,num_points_per_frame=n,num_frames=4,device='cpu',predict_global_frame_deltas=1,predict_only_global=False,dino_fusion=False,timeE=False) for n in (4,7)]
torch.manual_seed(1000);old=networks(original['PointMLP']);old_rng=torch.get_rng_state().clone()
records=[]
for mode in ['both','local','global']:
 torch.manual_seed(1000);new=networks(PointMLP)
 for net in new:net.select_network_paths(mode)
 assert torch.equal(old_rng,torch.get_rng_state()),'initialization RNG changed'
 for net,base in zip(new,old):
  for k,v in net.state_dict().items():assert torch.equal(v,base.state_dict()[k]),('retained initialization changed',mode,k)
  x=torch.linspace(-1,1,base.input_dim).reshape(1,-1,2)
  rng=torch.get_rng_state().clone()
  if mode=='both':expected=base(x)
  else:
   features=base.get_position_encoding_representation(x)
   expected=base.get_frame_deltas(x,features).squeeze(0) if mode=='global' else base.last_linear_layer(base.model(base.project_xy(features))).reshape(x.shape).squeeze(0)
  torch.set_rng_state(rng)
  assert torch.equal(net(x),expected),(mode,'incorrect branch output')
  if mode=='global':
   assert not any(hasattr(net,k) for k in ['project_xy','model','last_linear_layer']);assert net.get_points_params()==[]
  if mode=='local':
   assert not any(hasattr(net,k) for k in ['global_layers','frames_rigid_shared','frames_rigid_translation','frames_rigid_rotation','frames_rigid_scale','frames_rigid_shear']);assert net.get_global_params()==[]
 args=SimpleNamespace(lr_local=.005,lr_base_global=.0001,lr_init=.002,lr_final=.0008,lr_delay_mult=.1,lr_delay_steps=100,num_iter=1000,optim_points=True,split_global_loss=True,network_paths=mode)
 optimizer=PainterOptimizer(args);optimizer.init_optimizers([n.get_global_params() for n in new],[n.get_points_params() for n in new]);optimizer.zero_grad_()
 loss=sum(n(torch.linspace(-1,1,n.input_dim).reshape(1,-1,2)).square().mean() for n in new);loss.backward();optimizer.step_();optimizer.update_lr()
 assert optimizer.optim_points==(mode!='global');assert optimizer.optim_global==(mode!='local')
 assert all(torch.isfinite(p).all() for n in new for p in n.parameters())
 records.append(dict(mode=mode,parameters=sum(sum(p.numel() for p in n.parameters()) for n in new),optimizer_local=optimizer.optim_points,optimizer_global=optimizer.optim_global,remaining_weights_match_baseline=True,forward_matches_original_component=True))
# Real case: same grouping and initialized affine control points in all conditions.
runs=json.loads(Path('/data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-30case-24f-ablation/conf/runs.json').read_text())
run=next(r for r in runs if r['name']=='case01-NoCCFE');args_dict=run['arguments'];reference=None
for mode in ['both','local','global']:
 args=SimpleNamespace(**dict(args_dict,network_paths=mode));args.device=torch.device('cpu');args.output_folder=str(root/'tmp/branch-check'/mode);Path(args.output_folder).mkdir(parents=True,exist_ok=True)
 torch.manual_seed(1000);painter=Painter(args,args.target,args.num_frames,args.device)
 points={g:x.clone() for g,x in painter.points_mlp_input_by_g.items()}
 if reference is None:reference=points
 else:
  assert points.keys()==reference.keys()
  assert all(torch.equal(points[g],reference[g]) for g in points)
 records[['both','local','global'].index(mode)]['real_case_groups']=len(points)
 del painter
print(json.dumps({'passed':True,'checks':records},indent=2))
