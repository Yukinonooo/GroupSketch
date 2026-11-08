import unittest
from unittest.mock import patch
from losses import SDSVideoLoss, SDSLossBase
from types import SimpleNamespace
import torch
from models import PointMLP
from painter import PainterOptimizer

class RuntimeCorrectness(unittest.TestCase):
    def test_local_and_global_losses_share_one_pipeline(self):
        cfg=SimpleNamespace(model_name="test-model",use_xformers=False,del_text_encoders=False,caption="motion")
        def make_pipeline(*a,**kw):
            pipe=SimpleNamespace(scheduler=SimpleNamespace(alphas_cumprod=torch.ones(1000)))
            pipe.to=lambda device:pipe
            return pipe
        SDSLossBase._global_pipe=None
        try:
            with patch("losses.DiffusionPipeline.from_pretrained",side_effect=make_pipeline) as load, patch.object(SDSLossBase,"embed_text",return_value=torch.zeros(1)):
                local=SDSVideoLoss(cfg,torch.device("cpu"))
                global_=SDSVideoLoss(cfg,torch.device("cpu"),reuse_pipe=True)
                self.assertIs(local.pipe,global_.pipe)
                self.assertEqual(load.call_count,1)
        finally:
            SDSLossBase._global_pipe=None

    def test_scale_head_receives_gradient(self):
        torch.manual_seed(0)
        model = PointMLP(input_dim=16, inter_dim=128, num_points_per_frame=4,
            num_frames=2, device=torch.device("cpu"),predict_global_frame_deltas=1,
            predict_only_global=False, dino_fusion=False, timeE=False)
        points=torch.arange(1,17,dtype=torch.float32).reshape(1,8,2)
        features=model.get_position_encoding_representation(points)
        model.get_frame_deltas(points,features).square().sum().backward()
        gradients=[p.grad for p in model.frames_rigid_scale.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum().item()>0 for g in gradients),
                        "scale branch must independently affect output")
    def test_global_gradients_do_not_accumulate_across_loss_calls(self):
        cfg=SimpleNamespace(lr_local=.005,lr_base_global=.0001,lr_init=.002,
            lr_final=.0008,lr_delay_mult=.1,lr_delay_steps=100,num_iter=1000,
            optim_points=True,split_global_loss=True)
        local=torch.nn.Parameter(torch.tensor(1.))
        global_=torch.nn.Parameter(torch.tensor(1.))
        shared=torch.nn.Parameter(torch.tensor(1.))
        opt=PainterOptimizer(cfg);opt.init_optimizers([[global_,shared]],[[local,shared]])
        (local+global_+shared).backward()
        opt.zero_grad_()
        (local+global_+shared).backward()
        self.assertEqual(global_.grad.item(),1.)
        self.assertEqual(local.grad.item(),1.)
        self.assertEqual(shared.grad.item(),1.)
if __name__ == "__main__":
    unittest.main()
