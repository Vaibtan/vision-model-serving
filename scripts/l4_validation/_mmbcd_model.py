"""Private network-free MMBCD reconstruction for the released checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def build_mmbcd(dino_repo: Path):
    """Construct MMBCD from local architecture code without downloading weights."""

    dino_repo = dino_repo.resolve()
    module_path = dino_repo / "vision_transformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    import torch
    import torch.nn as nn
    from transformers import RobertaConfig, RobertaForSequenceClassification

    module_spec = importlib.util.spec_from_file_location(
        "vision_model_serving_pinned_dino", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load pinned DINO module: {module_path}")
    dino_vit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(dino_vit)

    class MMBCDInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_size = 224
            self.image_encoder = dino_vit.vit_small(patch_size=8, num_classes=0)
            # Both aliases exist in the released state dictionary.
            self.img_fc1 = nn.Linear(384, 256)
            self.img_fc_layer = nn.Sequential(
                nn.BatchNorm1d(384), self.img_fc1, nn.GELU()
            )
            config = RobertaConfig(
                vocab_size=50265,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                position_embedding_type="absolute",
                use_cache=True,
                classifier_dropout=None,
                num_labels=2,
                output_hidden_states=True,
            )
            self.text_encoder = RobertaForSequenceClassification(config)
            self.txt_fc1 = nn.Linear(768, 256)
            self.txt_fc_layer = nn.Sequential(
                nn.BatchNorm1d(768), self.txt_fc1, nn.GELU()
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=1,
                batch_first=True,
                dropout=0.3,
            )
            self.model_fc2 = nn.Linear(256 * 3, 2)

        def forward(self, image_tensor, input_ids, attention_mask):
            images = image_tensor.view(-1, 3, self.img_size, self.img_size)
            image_features = self.image_encoder(images).squeeze(-1).squeeze(-1)
            image_embeddings = self.img_fc_layer(image_features).view(
                image_tensor.shape[0], image_tensor.shape[1], -1
            )
            maxpooled_image_embeddings, _ = torch.max(image_embeddings, dim=1)
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            sentence_embeddings = text_output.hidden_states[-1][:, 0, :]
            text_embeddings = self.txt_fc_layer(sentence_embeddings)
            attention_features, _ = self.attention(
                text_embeddings.unsqueeze(1), image_embeddings, image_embeddings
            )
            fused_embeddings = torch.cat(
                (
                    attention_features.squeeze(1),
                    text_embeddings,
                    maxpooled_image_embeddings,
                ),
                dim=1,
            )
            logits = self.model_fc2(fused_embeddings.squeeze(1))
            return logits, fused_embeddings

    return MMBCDInference()


def verify_alias_values(state_dict):
    import torch

    alias_pairs = [
        ("img_fc1.weight", "img_fc_layer.1.weight"),
        ("img_fc1.bias", "img_fc_layer.1.bias"),
        ("txt_fc1.weight", "txt_fc_layer.1.weight"),
        ("txt_fc1.bias", "txt_fc_layer.1.bias"),
    ]
    for first, second in alias_pairs:
        if not torch.equal(state_dict[first], state_dict[second]):
            raise RuntimeError(f"Checkpoint alias values differ: {first} vs {second}")
