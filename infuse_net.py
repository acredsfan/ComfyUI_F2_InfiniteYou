# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from torch import Tensor, nn
from einops import rearrange, repeat

import comfy
from comfy.controlnet import controlnet_config, controlnet_load_state_dict, ControlNet, StrengthType
from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import (timestep_embedding)
import comfy.ldm.common_dit
import comfy.sampler_helpers


def _first_present(mapping, *keys):
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is not None:
                return value
    return None


def _normalize_condition_tensor(value, *, like_tensor=None, name="conditioning"):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # ComfyUI may store scalar conditioning values (e.g. guidance scale) as
        # plain Python numbers in the conditioning dict.  Convert to a 0-dim
        # tensor here; the caller is responsible for broadcasting to batch size.
        value = torch.tensor(float(value))
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected '{name}' to be a torch.Tensor, got {type(value).__name__}.")

    if like_tensor is not None:
        if value.device != like_tensor.device:
            value = value.to(device=like_tensor.device)
        if value.dtype != like_tensor.dtype:
            value = value.to(dtype=like_tensor.dtype)
    return value


def _infer_main_model_block_count(value, fallback):
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, (tuple, list)):
        numeric_values = [item for item in value if isinstance(item, int) and item > 0]
        if numeric_values:
            return max(numeric_values)
    return fallback


def _infer_linear_input_features(module, fallback):
    if module is None:
        return fallback

    direct_value = getattr(module, "in_features", None)
    if isinstance(direct_value, int) and direct_value > 0:
        return direct_value

    for child in module.modules():
        child_value = getattr(child, "in_features", None)
        if isinstance(child_value, int) and child_value > 0:
            return child_value

    return fallback


def _infer_patch_size(feature_width, base_channels, fallback=2):
    if not isinstance(feature_width, int) or feature_width <= 0:
        return fallback
    if not isinstance(base_channels, int) or base_channels <= 0:
        return fallback

    patch_area = feature_width / base_channels
    if patch_area <= 0:
        return fallback

    patch_size = int(round(math.sqrt(patch_area)))
    if patch_size > 0 and (patch_size * patch_size * base_channels) == feature_width:
        return patch_size
    return fallback


def _infer_linear_weight_shape(module):
    if module is None:
        return None, None

    weight = getattr(module, "weight", None)
    if isinstance(weight, Tensor) and weight.ndim == 2:
        return int(weight.shape[1]), int(weight.shape[0])

    for child in module.modules():
        child_weight = getattr(child, "weight", None)
        if isinstance(child_weight, Tensor) and child_weight.ndim == 2:
            return int(child_weight.shape[1]), int(child_weight.shape[0])

    return None, None


class InfuseNetFluxCompatibilityError(RuntimeError):
    pass

class InfuseNet(ControlNet):
    def __init__(self, 
                 control_model=None, 
                 id_embedding = None,
                 global_average_pooling=False, 
                 compression_ratio=8, 
                 latent_format=None, 
                 load_device=None, 
                 manual_cast_dtype=None, 
                 extra_conds=["y"], 
                 strength_type=StrengthType.CONSTANT, 
                 concat_mask=False, 
                 preprocess_image=lambda a: a):
        super().__init__(control_model=control_model, 
                         global_average_pooling=global_average_pooling, 
                         compression_ratio=compression_ratio, 
                         latent_format=latent_format, 
                         load_device=load_device, 
                         manual_cast_dtype=manual_cast_dtype, 
                         extra_conds=extra_conds, 
                         strength_type=strength_type, 
                         concat_mask=concat_mask, 
                         preprocess_image=preprocess_image)
        self.id_embedding = id_embedding

    def copy(self):
        c = InfuseNet(None,
                      global_average_pooling=self.global_average_pooling,
                      compression_ratio=self.compression_ratio,
                      latent_format=self.latent_format,
                      load_device=self.load_device,
                      manual_cast_dtype=self.manual_cast_dtype,
                      extra_conds=list(self.extra_conds) if self.extra_conds is not None else None,
                      strength_type=self.strength_type,
                      concat_mask=self.concat_mask)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        c.id_embedding = self.id_embedding
        self.copy_to(c)
        return c
    
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        cond = cond.copy()
        cond['crossattn_controlnet'] = self.id_embedding
        cond['c_crossattn'] = self.id_embedding
        return super().get_control(x_noisy, t, cond, batched_number, transformer_options)
    
class InfuseNetFlux(Flux):
    def __init__(self, latent_input=False, num_union_modes=0, mistoline=False, control_latent_channels=None, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        main_model_double = kwargs.pop("main_model_double", None)
        main_model_single = kwargs.pop("main_model_single", None)
        patch_size = kwargs.pop("patch_size", 2)
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, patch_size=patch_size, **kwargs)

        self.main_model_double = _infer_main_model_block_count(main_model_double, 19)
        self.main_model_single = _infer_main_model_block_count(main_model_single, 38)
        self.patch_size = patch_size if isinstance(patch_size, int) and patch_size > 0 else 2
        self.timestep_embedding_dim = _infer_linear_input_features(getattr(self, "time_in", None), 256)

        self.mistoline = mistoline
        # add ControlNet blocks
        if self.mistoline:
            control_block = lambda : MistolineControlnetBlock(self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            control_block = lambda : operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(self.params.depth):
            self.controlnet_blocks.append(control_block())

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(self.params.depth_single_blocks):
            self.controlnet_single_blocks.append(control_block())

        self.num_union_modes = num_union_modes
        self.controlnet_mode_embedder = None
        if self.num_union_modes > 0:
            self.controlnet_mode_embedder = operations.Embedding(self.num_union_modes, self.hidden_size, dtype=dtype, device=device)

        self.gradient_checkpointing = False
        self.latent_input = latent_input
        if control_latent_channels is None:
            control_latent_channels = self.in_channels
        else:
            control_latent_channels *= self.patch_size * self.patch_size

        self.pos_embed_input = operations.Linear(control_latent_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        if not self.latent_input:
            if self.mistoline:
                self.input_cond_block = MistolineCondDownsamplBlock(dtype=dtype, device=device, operations=operations)
            else:
                self.input_hint_block = nn.Sequential(
                    operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
                )

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control_type: Tensor = None,
        out_mask: Tensor = None
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        expected_img_in = _infer_linear_input_features(getattr(self, "img_in", None), None)
        if not (isinstance(expected_img_in, int) and expected_img_in > 0):
            expected_img_in, _ = _infer_linear_weight_shape(getattr(self, "img_in", None))

        if isinstance(expected_img_in, int) and expected_img_in > 0 and img.shape[-1] != expected_img_in:
            raise InfuseNetFluxCompatibilityError(
                f"InfuseNet received latent token width {img.shape[-1]}, but the loaded checkpoint expects {expected_img_in} for 'img_in'. "
                "This usually means the FLUX model and InfiniteYou InfuseNet checkpoint are incompatible (different latent width / patch-size configuration). "
                "Use a matching InfuseNet checkpoint for the selected FLUX model family."
            )

        # running on sequences img
        img = self.img_in(img)

        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(timesteps, self.timestep_embedding_dim))
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, self.timestep_embedding_dim))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        if self.controlnet_mode_embedder is not None and len(control_type) > 0:
            control_cond = self.controlnet_mode_embedder(torch.tensor(control_type, device=img.device), out_dtype=img.dtype).unsqueeze(0).repeat((txt.shape[0], 1, 1))
            txt = torch.cat([control_cond, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:,:1], txt_ids], dim=1)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        controlnet_double = ()

        for i in range(len(self.double_blocks)):
            img, txt = self.double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
            controlnet_double = controlnet_double + (self.controlnet_blocks[i](img),)

        img = torch.cat((txt, img), 1)

        controlnet_single = ()

        for i in range(len(self.single_blocks)):
            img = self.single_blocks[i](img, vec=vec, pe=pe)
            controlnet_single = controlnet_single + (self.controlnet_single_blocks[i](img[:, txt.shape[1] :, ...]),)

        repeat = math.ceil(self.main_model_double / len(controlnet_double))
        if self.latent_input:
            out_input = ()
            for x in controlnet_double:
                if out_mask is not None:
                    out_input += (x * out_mask,) * repeat
                else:
                    out_input += (x,) * repeat
        else:
            out_input = (controlnet_double * repeat)

        out = {"input": out_input[:self.main_model_double]}
        if len(controlnet_single) > 0:
            repeat = math.ceil(self.main_model_single / len(controlnet_single))
            out_output = ()
            if self.latent_input:
                for x in controlnet_single:
                    if out_mask is not None:
                        out_output += (x * out_mask,) * repeat
                    else:
                        out_output += (x,) * repeat
            else:
                out_output = (controlnet_single * repeat)
            out["output"] = out_output[:self.main_model_single]
        return out

    def _resolve_extra_conditions(self, *, context, y=None, guidance=None, kwargs=None):
        kwargs = kwargs or {}

        # Search all known key names used by FLUX.1 and FLUX.2 conditioning pipelines
        # for the pooled text embedding vector.
        resolved_y = _first_present(
            kwargs,
            "y", "pooled_output", "pooled_outputs", "vector", "vec",
            "clip_pooled", "pooled", "text_embeds",
        )
        if resolved_y is None:
            resolved_y = y
        resolved_y = _normalize_condition_tensor(resolved_y, like_tensor=context, name="y")

        # Search all known key names for the guidance (distillation) embedding.
        resolved_guidance = _first_present(
            kwargs,
            "guidance", "guidance_vec", "guidance_embedding", "guidance_scale",
        )
        if resolved_guidance is None:
            resolved_guidance = guidance
        resolved_guidance = _normalize_condition_tensor(resolved_guidance, like_tensor=context, name="guidance")

        if resolved_y is None:
            # Graceful fallback for FLUX.2 or any pipeline that does not expose a
            # CLIP-style pooled embedding: derive the expected vector dimension from
            # the model's vector_in projection and substitute a zero tensor so that
            # sampling continues without error.
            vec_in_mod = getattr(self, "vector_in", None)
            in_dim = _infer_linear_input_features(vec_in_mod, None)
            if in_dim is not None:
                resolved_y = context.new_zeros(context.shape[0], in_dim)
            else:
                available_keys = ", ".join(sorted(kwargs.keys())) if len(kwargs) > 0 else "none"
                raise InfuseNetFluxCompatibilityError(
                    "InfuseNet-FLUX expected pooled conditioning tensor 'y', but the active ComfyUI FLUX pipeline did not provide one "
                    "and the expected pooled-vector dimension could not be inferred from the loaded checkpoint. "
                    f"Available extra args: {available_keys}. "
                    "Make sure the workflow exposes FLUX-compatible pooled embeddings, or that the InfuseNet checkpoint includes a 'vector_in' layer."
                )

        return resolved_y, resolved_guidance

    def forward(self, x, timesteps, context, y=None, guidance=None, hint=None, **kwargs):
        y, guidance = self._resolve_extra_conditions(context=context, y=y, guidance=guidance, kwargs=kwargs)
        if hint is None:
            raise InfuseNetFluxCompatibilityError(
                "InfuseNet-FLUX expected a control hint tensor but none was provided. "
                "Make sure the Apply InfuseNet node is connected to a control image before sampling."
            )

        patch_size = self.patch_size
        if self.latent_input:
            hint = comfy.ldm.common_dit.pad_to_patch_size(hint, (patch_size, patch_size))
        elif self.mistoline:
            hint = hint * 2.0 - 1.0
            hint = self.input_cond_block(hint)
        else:
            hint = hint * 2.0 - 1.0
            hint = self.input_hint_block(hint)

        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        # ── Expand / normalise guidance to match the real batch size ──────────
        # guidance can arrive as:
        #   • None           – ComfyUI pipeline did not supply it
        #   • 0-dim tensor   – came in as a Python scalar that was converted by
        #                      _normalize_condition_tensor
        #   • (1,) tensor    – single-element, needs tiling for CFG
        #   • (bs,) tensor   – already correct
        if guidance is not None:
            if guidance.dim() == 0:
                guidance = guidance.unsqueeze(0)
            if guidance.shape[0] != bs:
                if guidance.shape[0] == 1:
                    guidance = guidance.expand(bs)
                else:
                    repeats = math.ceil(bs / guidance.shape[0])
                    guidance = guidance.repeat(repeats)[:bs]
        elif self.params.guidance_embed:
            guidance = timesteps.new_zeros(bs)

        # ── Expand y (pooled embedding) to match the real batch size ──────────
        # The zero-fallback in _resolve_extra_conditions is built with
        # context.shape[0] (id-embedding batch = 1) while bs may be 2 under CFG.
        if y is not None and y.shape[0] != bs:
            if y.shape[0] == 1:
                y = y.expand(bs, -1)
            else:
                repeats = math.ceil(bs / y.shape[0])
                y = y.repeat(repeats, 1)[:bs]

        control_mask = kwargs.get("control_mask", None)
        out_mask = None
        if control_mask is not None:
            in_mask = comfy.sampler_helpers.prepare_mask(control_mask, (bs, c, h, w), img.device)
            in_mask = comfy.ldm.common_dit.pad_to_patch_size(in_mask, (patch_size, patch_size))
            in_mask = rearrange(in_mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            # (b, seq_len, _) =>(b, seq_len, pulid.dim)
            in_mask = in_mask[..., 0].unsqueeze(-1).repeat(1, 1, img.shape[-1]).to(dtype=img.dtype)
            img = img * in_mask
            
            out_mask = comfy.sampler_helpers.prepare_mask(control_mask, (bs, 
                                                                      self.hidden_size // (patch_size * patch_size), h, w), img.device)
            out_mask = comfy.ldm.common_dit.pad_to_patch_size(out_mask, (patch_size, patch_size))
            out_mask = rearrange(out_mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        # Expand context (id_embedding) to match the image batch size.  When
        # CFG > 1 the sampler passes a double-batched x (positive + negative
        # concatenated), so context must be tiled to the same batch size or the
        # double blocks will receive mismatched tensors and raise a shape error.
        if context.shape[0] != bs:
            if context.shape[0] == 1:
                context = context.expand(bs, -1, -1)
            elif context.shape[0] < bs:
                repeats = math.ceil(bs / context.shape[0])
                context = context.repeat(repeats, 1, 1)[:bs]
            else:
                context = context[:bs]
        return self.forward_orig(img, img_ids, hint, context, txt_ids, timesteps, y, guidance, control_type=kwargs.get("control_type", []), out_mask=out_mask)

def load_infuse_net_flux(ckpt_path, model_options={}):
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(new_sd, model_options=model_options)
    for k in sd:
        new_sd[k] = sd[k]

    num_union_modes = 0
    union_cnet = "controlnet_mode_embedder.weight"
    if union_cnet in new_sd:
        num_union_modes = new_sd[union_cnet].shape[0]

    pos_embed_input_weight = new_sd.get("pos_embed_input.weight")
    if pos_embed_input_weight is None:
        raise InfuseNetFluxCompatibilityError(
            "The selected InfiniteYou checkpoint is missing 'pos_embed_input.weight'. "
            "This checkpoint does not match the expected FLUX InfuseNet ControlNet format."
        )

    inferred_unet_config = dict(model_config.unet_config)
    base_latent_channels = inferred_unet_config.get("in_channels")
    patch_size = _infer_patch_size(pos_embed_input_weight.shape[1], base_latent_channels, fallback=inferred_unet_config.get("patch_size", 2))
    control_latent_channels = pos_embed_input_weight.shape[1] // (patch_size * patch_size)
    concat_mask = False
    if control_latent_channels == 17:
        concat_mask = True

    inferred_unet_config.setdefault("main_model_double", inferred_unet_config.get("depth"))
    inferred_unet_config.setdefault("main_model_single", inferred_unet_config.get("depth_single_blocks"))
    inferred_unet_config["patch_size"] = patch_size

    control_model = InfuseNetFlux(latent_input=True, num_union_modes=num_union_modes, control_latent_channels=control_latent_channels, operations=operations, device=offload_device, dtype=unet_dtype, **inferred_unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.Flux()
    extra_conds = ['y']
    if getattr(control_model.params, "guidance_embed", False):
        extra_conds.append('guidance')
    control = InfuseNet(control_model, compression_ratio=1, latent_format=latent_format, concat_mask=concat_mask, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control
