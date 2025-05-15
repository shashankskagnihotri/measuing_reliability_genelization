# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
import torch
from torch import Tensor
import torchvision
from cospgd import functions as attack

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from mmseg.models.utils import resize

import math

@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 attack_loss: ConfigType=None,
                 attack_cfg: ConfigType=None,
                 normalize_mean_std: ConfigType=None,
                 enable_normalization: bool = False,
                 perform_attack: bool=False):
        # import pdb
        # pdb.set_trace()
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if attack_cfg is not None:
            if "targeted" not in attack_cfg:
                attack_cfg["targeted"] = False
        self.attack_cfg = attack_cfg
        self.attack_loss = attack_loss
        self.criterion = MODELS.build(self.attack_loss) 
        self.perform_attack=perform_attack
        self.enable_normalization = enable_normalization
        
        self.mean=normalize_mean_std['mean']
        self.std=normalize_mean_std['std']
        self.counter=0

        assert self.with_decode_head

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("Number of parameters: ", count_parameters(self))

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def segpgd_scale(
            self,
            predictions,
            labels,
            loss,
            iteration,
            iterations,
            targeted=False,
        ):
        lambda_t = iteration/(2*iterations)
        output_idx = torch.argmax(predictions, dim=1)
        if targeted:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    lambda_t*loss,
                    (1-lambda_t)*loss
                )
            ) / (predictions.shape[-2]*predictions.shape[-1])
        else:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    (1-lambda_t)*loss,
                    lambda_t*loss
                )
            ) / (predictions.shape[-2]*predictions.shape[-1])
        return loss
    
    def cospgd_scale(
            self,
            predictions,
            labels,
            loss,
            num_classes=None,
            targeted=False,
            one_hot=True,
        ):
        if one_hot:
            transformed_target = torch.nn.functional.one_hot(
                torch.clamp(labels, 0, num_classes-1),
                num_classes = num_classes
            ).permute(0,3,1,2)
        else:
            transformed_target = torch.nn.functional.softmax(labels, dim=1)
        cossim = torch.nn.functional.cosine_similarity(
            torch.nn.functional.softmax(predictions, dim=1),
            transformed_target,
            dim = 1
        )
        # import pdb
        # pdb.set_trace()
        if targeted:
            cossim = 1 - cossim # if performing targeted attacks, we want to punish for dissimilarity to the target        
        
        del transformed_target
        del predictions
        torch.cuda.empty_cache()
        
        return cossim.detach()        

    def apgd(
            self,
            model,
            normalize_inputs,
            batch_img_metas,
            data_samples,
            n_iter,
            x,
            y,
            device,
            eps,
            criterion_indiv = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none'),
        ):
        
        def normalize(x, ndims):
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * ndims)) + 1e-12)
        
        def check_oscillation(x, j, k, y5, device, k3=0.75):
            t = torch.zeros(x.shape[1]).to(device)
            for counter5 in range(k):
                t += (x[j - counter5] > x[j - counter5 - 1]).float()

            return (t <= k * k3 * torch.ones_like(t)).float()
        
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        
        thr_decr = 0.75
        
        
        orig_dim = list(x.shape[1:])
        ndims = len(orig_dim)
        
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x + eps * torch.ones_like(x).detach() * normalize(t, ndims)
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros(
            [n_iter, x.shape[0]]
        ).to(device)
        loss_best_steps = torch.zeros(
            [n_iter + 1, x.shape[0]]
        ).to(device)
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        with torch.enable_grad():
        
            loss_indiv = self.loss(normalize_inputs(x_adv*255), data_samples)['decode.loss_ce']
            loss = loss_indiv.sum()
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        
        grad_best = grad.clone()
        
        # acc = logits.detach().max(1)[1] == y
        # acc_steps[0] = acc.mean()
        loss_best = loss_indiv.detach().view(len(loss_indiv),-1).mean(dim=1)
        
        alpha = 2.
        
        step_size = alpha * eps * torch.ones(
            [x.shape[0], *([1] * ndims)]
        ).to(device).detach()
        x_adv_old = x_adv.clone()
        
        counter = 0
        k = n_iter_2
        n_fts = math.prod(orig_dim)
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0
        
        u = torch.arange(x.shape[0], device=device)
        for i in range(n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(x_adv_1, x - eps),
                        x + eps
                    ), 0.0, 1.0
                )
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                            x - eps
                        ), x + eps
                    ), 0.0, 1.0
                )
                
                x_adv = x_adv_1 + 0.
            
            ### get grad
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            with torch.enable_grad():
                loss_indiv = self.loss(normalize_inputs(x_adv*255), data_samples)['decode.loss_ce']
                loss = loss_indiv.sum()

                grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            
            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().view(len(loss_indiv),-1).mean(dim=1)
                loss_steps[i] = y1
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best_adv[ind] = x_adv[ind] + 0.
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = check_oscillation(
                        loss_steps, i, k, loss_best, device=device, k3=thr_decr
                    )
                    fl_reduce_no_impr = (1. - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(
                        fl_oscillation,
                        fl_reduce_no_impr
                    )
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - size_decr, n_iter_min)
        
                    counter3 = 0
                    
        return x_best_adv*255
        
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """

        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        # print(f"old: epsilon={self.attack_cfg['epsilon']}, alpha={self.attack_cfg['alpha']}")

        # old version: 
        # self.attack_cfg["epsilon"] = self.attack_cfg["epsilon"]/255 if inputs.max() <= 1 else self.attack_cfg["epsilon"]
        # self.attack_cfg["alpha"] = self.attack_cfg["alpha"]*255 if inputs.max() > 1 else self.attack_cfg["alpha"]


        # propose:
        if self.attack_cfg is not None:
            epsilon = self.attack_cfg["epsilon"]/255 if inputs.max() <= 1 else self.attack_cfg["epsilon"]
            alpha = self.attack_cfg["alpha"]*255 if inputs.max() > 1 else self.attack_cfg["alpha"]

        # print(f"new: epsilon={epsilon}, alpha={alpha}")

        

        # import pdb
        # pdb.set_trace()

        normalize = torchvision.transforms.Normalize(mean = self.mean, std=self.std) if not self.enable_normalization else torch.nn.Identity()
        
        
        if self.perform_attack:
            if self.attack_cfg['name']=='apgd':
                inputs = self.apgd(model = self.inference, normalize_inputs=normalize, batch_img_metas = batch_img_metas, data_samples=data_samples, n_iter=self.attack_cfg['iterations'], x = inputs/255., y = data_samples[-1].gt_sem_seg, device = inputs.device, eps=epsilon/255.)
            else:
                orig_inputs = inputs.clone().detach()
                
                # import pdb
                # pdb.set_trace()
                if 'pgd' in self.attack_cfg['name']:
                    if self.attack_cfg['norm'] == 'linf':
                        inputs = attack.init_linf(inputs, epsilon, clamp_min = 0, clamp_max=255)
                    elif self.attack_cfg['norm'] == 'l2':
                        inputs = attack.init_l2(inputs, epsilon, clamp_min = 0, clamp_max=255)
                    else:
                        raise NotImplementedError('Norm ' + str(self.attack_cfg['norm']) +' not implemented, only linf and l2 norm implemented')                                               
                
                for itr in range(self.attack_cfg['iterations']):
                    inputs.requires_grad = True
                    
                    self.zero_grad()
                    
                
                    with torch.enable_grad():
                        seg_logits = self.inference(normalize(inputs), batch_img_metas)

                        # loss = self.loss(normalize(inputs), data_samples)['decode.loss_ce']
                        loss_temp = self.loss(normalize(inputs), data_samples)

                        if 'decode.loss_ce' in loss_temp:
                            loss = loss_temp['decode.loss_ce']
                        elif 'decode.loss_dice' in loss_temp:
                            loss = loss_temp['decode.loss_dice']
                        elif 'decode.loss_boundary' in loss_temp:
                            loss = loss_temp['decode.loss_boundary']
                        elif 'decode.loss_focal' in loss_temp:
                            loss = loss_temp['decode.loss_focal']
                        elif 'decode.loss_huasdorff_disstance' in loss_temp:
                            loss = loss_temp['decode.loss_huasdorff_disstance']
                        elif 'decode.loss_kld' in loss_temp:
                            loss = loss_temp['decode.loss_kld']
                        elif 'decode.loss_lovasz' in loss_temp:
                            loss = loss_temp['decode.loss_lovasz']
                        elif 'decode.loss_ohem' in loss_temp:
                            loss = loss_temp['decode.loss_ohem']
                        elif 'decode.loss_silog' in loss_temp:
                            loss = loss_temp['decode.loss_silog']
                        elif 'decode.loss_tversky' in loss_temp:
                            loss = loss_temp['decode.loss_tversky']
                        else:
                            raise AttributeError('Decoder has no loss defined')
         
                        img_meta = batch_img_metas[0]
                        batch_size, C, H, W = seg_logits.shape
                        if 'img_padding_size' not in img_meta:
                            padding_size = img_meta.get('padding_size', [0] * 4)
                        else:
                            padding_size = img_meta['img_padding_size']
                        padding_left, padding_right, padding_top, padding_bottom =\
                            padding_size
                        # i_seg_logits shape is 1, C, H, W after remove padding
                        i_seg_logits = seg_logits[:, :,
                                                padding_top:H - padding_bottom,
                                                padding_left:W - padding_right]
                        
                        resized_seg_logits = resize(
                                        i_seg_logits,
                                        size=batch_img_metas[0]['ori_shape'],
                                        mode='bilinear',
                                        align_corners=self.align_corners,
                                        warning=False)
                        
                        
                        
                        if self.attack_cfg['name'] == 'cospgd':
                        
                            with torch.no_grad():
                                
                                cossim = self.cospgd_scale(resized_seg_logits.detach(), data_samples[-1].gt_sem_seg.data.detach(), loss, num_classes=resized_seg_logits.shape[1], targeted=self.attack_cfg['targeted'], one_hot=True)
                            loss = cossim.detach() * loss
                        elif self.attack_cfg['name'] == 'segpgd':
                            loss = self.segpgd_scale(resized_seg_logits, data_samples[-1].gt_sem_seg.data, loss, iteration=itr, iterations=self.attack_cfg['iterations'], targeted=self.attack_cfg['targeted'])
                    
                    
                        loss.mean().backward()
                    
                    if self.attack_cfg['norm'] == 'linf':
                        inputs = attack.step_inf(inputs, epsilon, data_grad=inputs.grad, orig_image=orig_inputs, alpha=alpha, targeted=self.attack_cfg['targeted'], clamp_min = 0, clamp_max=255)
                    elif self.attack_cfg['norm'] == 'l2':
                        inputs = attack.step_l2(inputs, epsilon, data_grad=inputs.grad, orig_image=orig_inputs, alpha=alpha, targeted=self.attack_cfg['targeted'], clamp_min = 0, clamp_max=255)
                    else:
                        raise NotImplementedError('Only linf and l2 norm implemented')
        
        seg_logits = self.inference(normalize(inputs), batch_img_metas)


        # seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
