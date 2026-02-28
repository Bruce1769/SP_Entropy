import torch
from sampling.sampling_utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers import DynamicCache


class KVCacheModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1.0, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor, use_debug=False) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._past_key_values = outputs.past_key_values
            logits = outputs.logits
            # norm 最后一个 position 的 logits 并返回
            last_q = norm_logits(logits[:, -1, :], self._temperature, self._top_k, self._top_p)
            return last_q
        else:
            if self._past_key_values is not None and isinstance(self._past_key_values, (list, tuple)):
                self._past_key_values = DynamicCache.from_legacy_cache(self._past_key_values)

            outputs = self._model(input_ids, past_key_values=self._past_key_values, use_cache=True)

            logits = outputs.logits
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)

            # norm 所有新位置的 logits
            probs = torch.empty_like(logits)
            for i in range(logits.shape[1]):
                probs[:, i, :] = norm_logits(logits[:, i, :], self._temperature, self._top_k, self._top_p)

            self._past_key_values = outputs.past_key_values
            return probs

    def rollback(self, end_pos: int):
        """截断 KV Cache 到 end_pos 长度"""
        if self._past_key_values is None:
            return

        if hasattr(self._past_key_values, 'crop'):
            self._past_key_values.crop(end_pos)
        elif hasattr(self._past_key_values, 'key_cache'):
            # DynamicCache: 直接原地截断 list 中的 tensor
            for layer_idx in range(len(self._past_key_values.key_cache)):
                self._past_key_values.key_cache[layer_idx] = self._past_key_values.key_cache[layer_idx][:, :, :end_pos, :]
                self._past_key_values.value_cache[layer_idx] = self._past_key_values.value_cache[layer_idx][:, :, :end_pos, :]
        else:
            # Legacy tuple format fallback
            past_key_values_trimmed = []
            for kv in self._past_key_values:
                k, v = kv
                if isinstance(self._model, BloomForCausalLM):
                    k = k[:, :, :end_pos]
                    v = v[:, :end_pos, :]
                else:
                    k = k[:, :, :end_pos, :]
                    v = v[:, :, :end_pos, :]
                past_key_values_trimmed.append((k, v))
            self._past_key_values = past_key_values_trimmed
