"""Continue the released spatial adapter or freeze its merged backbone."""
import _bootstrap  # noqa: F401
import hashlib
import json
from pathlib import Path
import torch
import xvla_bridge as XB

PARENT = 'cce/results/libero_family_spec_v4r1_fallback.json'
MODULES = ('soft_prompt_hub', 'action_encoder', 'action_decoder')


def frozen_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_policy(mode, checkpoint=None, training=False):
    XB.add_xvla_to_path()
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    spec = json.loads(Path(PARENT).read_text())
    base, adapter = spec['server']['model_path'], spec['server']['lora_path']
    model = XVLA.from_pretrained(base, torch_dtype=torch.float32)
    processor = XVLAProcessor.from_pretrained(base)
    source = str(checkpoint) if mode == 'ft_demo' and checkpoint else adapter
    wrapped = PeftModel.from_pretrained(model, source, is_trainable=training and mode == 'ft_demo')
    cfg = wrapped.peft_config['default']
    assert cfg.r == 8 and cfg.lora_alpha == 16 and cfg.bias == 'none'
    expected = load_file(str(Path(source) / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(wrapped)
    assert set(expected) == set(actual), 'Adapter tensor names differ'
    assert all(torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()), 'Adapter load mismatch'
    stats = {'adapter_tensor_count': len(expected), 'adapter_loaded_exactly': True,
             'official_adapter': adapter, 'mode': mode, 'checkpoint': str(checkpoint) if checkpoint else None}
    del actual, expected
    if mode == 'prompt_fit':
        model = wrapped.merge_and_unload(safe_merge=True)
        assert not any('lora_' in n for n, _ in model.named_parameters())
        if checkpoint:
            state = load_file(str(Path(checkpoint) / 'cce_domain_state.safetensors'))
            expected_keys = {n for n in model.state_dict() if any(n.startswith(f'transformer.{m}.') for m in MODULES)}
            assert set(state) == expected_keys
            result = model.load_state_dict(state, strict=False)
            assert not result.unexpected_keys
        model.requires_grad_(False)
        if training:
            for m in MODULES:
                getattr(model.transformer, m).requires_grad_(True)
        raw = model
    elif mode == 'ft_demo':
        if training:
            model = wrapped; raw = wrapped.get_base_model()
        else:
            model = wrapped.merge_and_unload(safe_merge=True); raw = model
    else:
        raise ValueError(mode)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if training:
        if mode == 'ft_demo':
            assert names and all('lora_' in n or 'modules_to_save' in n for n in names)
            assert any('vlm' in n and 'lora_' in n for n in names), 'Official all-linear recipe missing VLM adapters'
        else:
            assert names and all(any(n.startswith(f'transformer.{m}.') for m in MODULES) for n in names)
    stats.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 total_parameters=sum(p.numel() for p in model.parameters()), trainable_names=names if training else [])
    return model, raw, processor, stats
