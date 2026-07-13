"""VLM 백엔드 래퍼. 기본은 Qwen2.5-VL (transformers).

다른 모델/API로 바꾸려면 answer(frames, prompt) -> str 인터페이스만 맞추면 된다.
"""
import torch

import config
from prompts import SYSTEM_PROMPT


class QwenVLM:
    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def answer(self, frames, prompt: str) -> str:
        """frames: PIL.Image 리스트 (시간순), prompt: 질문+보기 텍스트."""
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=frames, return_tensors="pt"
        ).to(self.model.device)

        out = self.model.generate(
            **inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False
        )
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


def load_model(name: str):
    return QwenVLM(name)
