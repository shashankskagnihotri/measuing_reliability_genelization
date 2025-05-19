from typing import Literal

import torch
from mmengine.evaluator import Evaluator
from mmengine.model import BaseModel
from torchvision import transforms

from .common import Attack


class PGD(Attack):
    """Projected Gradient Descent (PGD) attack."""

    def __init__(
        self,
        epsilon: float = 8,
        alpha: float = 2.55,
        steps: int = 20,
        target: bool | int = False,
        random_start: bool = False,
        norm: Literal["inf", "two"] = "inf",
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.target = target
        self.random_start = random_start
        self.norm = norm

    def run_batch(self, data_batch: dict, model: BaseModel, evaluators: list[Evaluator]):
        """"""

        data_batch_prepro, images = self._preprocess_data(data_batch, model)
        adv_images = images.clone().float().detach().to("cuda")
        self._evaluator_process(evaluators[0], data_batch_prepro, model)

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_images = torch.clamp(adv_images, min=0, max=255).detach()

        for step in range(self.steps):
            adv_images.requires_grad = True
            data_batch_prepro["inputs"][0] = adv_images
            model.training = True  # avoid missing arguments error in some models

            if self.target:
                if isinstance(self.target, bool):
                    target = torch.full_like(
                        data_batch_prepro["data_samples"][0].gt_instances.labels, 42
                    )
                elif isinstance(self.target, int):
                    target = torch.full_like(
                        data_batch_prepro["data_samples"][0].gt_instances.labels, self.target
                    )
                elif isinstance(self.target, torch.Tensor):
                    assert (
                        self.target.shape
                        == data_batch_prepro["data_samples"][0].gt_instances.labels.shape
                    )
                    target = self.target
                else:
                    raise ValueError("Invalid target type")

                original_labels = data_batch_prepro["data_samples"][0].gt_instances.labels
                data_batch_prepro["data_samples"][0].gt_instances.labels = target

            losses = model(**data_batch_prepro, mode="loss")
            loss, _ = model.parse_losses(losses)

            model.zero_grad()
            adv_images.grad = None

            loss.backward(retain_graph=True)
            grad = adv_images.grad
            assert grad is not None

            if self.target:
                grad *= -1

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            if self.norm == "inf":
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            elif self.norm == "two":
                delta = adv_images - images
                batch_size = images.shape[0]
                delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                factor = self.epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)
            else:
                raise NotImplementedError
            adv_images = torch.clamp(images + delta, min=0, max=255).detach()

            data_batch_prepro["inputs"][0] = adv_images
            transforms.Normalize(
                model.data_preprocessor.mean,
                model.data_preprocessor.std,
                inplace=True,
            )(data_batch_prepro["inputs"][0])

            if self.target:
                data_batch_prepro["data_samples"][0].gt_instances.labels = original_labels

            model.training = False  # avoid missing arguments error in some models
            self._evaluator_process(evaluators[step + 1], data_batch_prepro, model)

        return data_batch_prepro
