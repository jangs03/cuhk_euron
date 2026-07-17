"""VLM 백엔드 래퍼. 기본은 Qwen2.5-VL (transformers).

다른 모델/API로 바꾸려면 answer(frames, prompt) -> str 인터페이스만 맞추면 된다.
로짓 기반 디코딩(option_logprobs/yes_probability)은 지원하는 백엔드만 구현하면 됨.
"""
import torch

import config
from prompts import SYSTEM_PROMPT


def _attn_implementation() -> str:
    """flash-attn이 설치돼 있으면 FlashAttention-2, 아니면 PyTorch SDPA.
    비전 토큰이 긴 입력(프레임 8~16장)에서 FA2가 유의미하게 빠르다."""
    import importlib.util
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


class QwenVLM:
    def __init__(self, model_name: str = config.DEFAULT_MODEL,
                 quant: str | None = None):
        """quant: None(bf16) / '4bit' / '8bit' (bitsandbytes 온더플라이 양자화).

        속도가 목적이면 quant보다 AWQ 체크포인트를 권장:
          --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ  (+ pip install autoawq)
        AWQ는 융합 커널이라 빠르고 VRAM ~1/3, bnb 4bit는 VRAM 절감용(속도는 비슷하거나 느림).
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = dict(
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation=_attn_implementation(),
        )
        if quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if quant == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"[vlm] {model_name} | attn={kwargs['attn_implementation']}"
              f" | quant={quant or ('awq' if 'awq' in model_name.lower() else 'bf16')}")

    def _build_inputs(self, frames, prompt: str, times: list[float] | None = None,
                      assistant_prefix: str = ""):
        """공통 입력 구성. assistant_prefix를 주면 어시스턴트 응답이 그 텍스트로
        시작한다고 가정한 위치의 로짓을 뽑을 수 있다 (예: 'ANSWER:')."""
        # 프레임마다 'Frame i:' 라벨을 끼워 넣어 시간축을 명시 (sequence/emotion에 중요)
        content = []
        for i, img in enumerate(frames, 1):
            label = f"Frame {i}:"
            if times and i <= len(times):
                label = f"Frame {i} (t={times[i - 1]:.1f}s):"
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + assistant_prefix
        return self.processor(
            text=[text], images=frames, return_tensors="pt"
        ).to(self.model.device)

    @torch.inference_mode()
    def answer(self, frames, prompt: str, times: list[float] | None = None) -> str:
        """frames: PIL.Image 리스트 (시간순), prompt: 질문+보기 텍스트,
        times: 프레임별 타임스탬프(초) — 있으면 라벨에 포함 (속도/순서 판단 단서)."""
        inputs = self._build_inputs(frames, prompt, times)
        out = self.model.generate(
            **inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False
        )
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def _first_token_ids(self, word: str) -> list[int]:
        """'A'와 ' A'처럼 변형 표기의 첫 토큰 id 목록 (중복 제거)."""
        tok = self.processor.tokenizer
        ids = []
        for v in (word, " " + word):
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
        return list(dict.fromkeys(ids))

    @torch.inference_mode()
    def option_logprobs(self, frames, prompt: str, words: list[str],
                        times: list[float] | None = None) -> dict[str, float]:
        """'ANSWER:' 다음 첫 토큰의 로그확률로 각 후보 단어를 스코어링.

        자유 생성 대신 모델의 확신을 직접 읽는다 — 생성 노이즈/파싱 실패 제거.
        words 예: ["A","B","C","D"] 또는 ["YES","NO"].
        """
        inputs = self._build_inputs(frames, prompt, times, assistant_prefix="ANSWER:")
        logits = self.model(**inputs).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        scores = {}
        for w in words:
            ids = self._first_token_ids(w)
            scores[w] = float(torch.logsumexp(logprobs[ids], dim=0))
        return scores

    @torch.inference_mode()
    def yes_probability(self, frames, prompt: str,
                        times: list[float] | None = None) -> float:
        """이진 질의의 P(YES) — YES/NO 두 스코어를 정규화한 상대 확률."""
        lp = self.option_logprobs(frames, prompt, ["YES", "NO"], times)
        pair = torch.tensor([lp["YES"], lp["NO"]])
        return float(torch.softmax(pair, dim=0)[0])


def load_model(name: str, quant: str | None = None):
    return QwenVLM(name, quant=quant)
