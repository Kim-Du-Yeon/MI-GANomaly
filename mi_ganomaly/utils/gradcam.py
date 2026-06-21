import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x):
        """x: (1, C, H, W). Backprops the GANomaly anomaly score (TotalLoss) to the
        hooked layer to produce a Grad-CAM++ heatmap explaining that score."""
        self.model.zero_grad()

        x_hat, z, z_hat, feat_real, feat_fake = self.model(x)
        score, _, _, _ = self.model.criterion(x, x_hat, feat_real, feat_fake, z, z_hat)
        score.backward()

        acts = self.activations
        grads = self.gradients

        eps = 1e-8
        grads_sq = grads.pow(2)
        grads_cube = grads.pow(3)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        alpha_denom = 2 * grads_sq + sum_acts * grads_cube
        alpha_denom = torch.where(alpha_denom.abs() > eps, alpha_denom, torch.full_like(alpha_denom, eps))
        alpha = grads_sq / alpha_denom

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))

        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu()

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo) if (hi - lo) > eps else torch.zeros_like(cam)

        return cam.numpy()
