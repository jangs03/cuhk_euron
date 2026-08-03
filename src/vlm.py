"""VLM 백엔드 래퍼. 기본은 Qwen2.5-VL (transformers).

다른 모델/API로 바꾸려면 answer(frames, prompt) -> str 인터페이스만 맞추면 된다.
로짓 기반 디코딩(option_logprobs/yes_probability)은 지원하는 백엔드만 구현하면 됨.
"""
import math

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


# 표준 chat template + 다중 이미지를 지원해 이 파이프라인에서 그대로 동작하는 모델들.
# (InternVL·Phi-4는 자체 인터페이스라 전용 어댑터 없이는 쓸 수 없어 제외했다)
KNOWN_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",   # 다중 이미지·비디오 특화
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "google/gemma-3-4b-it", "google/gemma-3-12b-it",  # 다른 계열 (라이선스 동의 필요)
]

# 저장소 코드 실행이 필요한 공식 저장소 (--trust-remote-code 없이도 자동 허용)
AUTO_TRUST_PREFIXES = ("OpenGVLab/", "microsoft/Phi-")

# 프로세서 출력에서 '이미지가 실제로 들어갔는지' 판별할 키들
IMAGE_INPUT_KEYS = ("pixel_values", "pixel_values_videos", "image_sizes",
                    "input_image_embeds", "image_patches", "images")


def _not_found(model_name: str):
    raise SystemExit(
        f"\n[vlm] 모델을 찾을 수 없습니다: {model_name}\n"
        f"  HF에 없는 ID이거나 오타입니다. 사용 가능한 예:\n"
        + "".join(f"    - {m}\n" for m in KNOWN_MODELS)
        + "  ※ Qwen3-VL은 4B / 8B / 32B만 있습니다 (7B 없음)\n"
    ) from None


def _load_backbone(model_name: str, kwargs: dict):
    """모델 클래스 자동 선택 — Qwen2.5-VL / Qwen3-VL 등을 --model만으로 교체.

    체크포인트의 model_type을 먼저 읽어 아키텍처가 맞는 클래스로만 로드한다.
    맞지 않는 클래스로 폴백하면 가중치가 0개 로드된 채 조용히 돌아가므로
    (InternVL 체크포인트를 Qwen 클래스에 넣는 사고) 그런 폴백은 하지 않는다."""
    trc = bool(kwargs.get("trust_remote_code", False))
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trc)
    except Exception as e:
        msg = str(e)
        if "is not a local folder" in msg or "Repository Not Found" in msg:
            _not_found(model_name)
        raise
    ckpt_type = (getattr(cfg, "model_type", "") or "").lower()

    model = None
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    except Exception as e:
        print(f"[vlm] 범용 로더 실패({type(e).__name__}: {str(e)[:120]})")
        if ckpt_type.startswith("qwen"):        # 같은 계열일 때만 전용 클래스 폴백
            from transformers import Qwen2_5_VLForConditionalGeneration
            kwargs.pop("trust_remote_code", None)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, **kwargs)
        elif trc:                                # 커스텀 아키텍처는 AutoModel로
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name, **kwargs)
        else:
            raise SystemExit(
                f"\n[vlm] {model_name} (model_type={ckpt_type}) 를 로드할 수 없습니다.\n"
                f"  현재 transformers가 이 아키텍처를 지원하지 않습니다.\n"
                f"  해결: pip install -U transformers 후 재시도, 또는 다른 모델 사용\n"
            ) from None

    # 아키텍처 불일치 검증 — 다른 클래스로 로드되면 가중치가 실리지 않는다
    loaded_type = (getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    if ckpt_type and loaded_type and ckpt_type != loaded_type:
        raise SystemExit(
            f"\n[vlm] 아키텍처 불일치 — 이 모델은 이 파이프라인에서 쓸 수 없습니다.\n"
            f"  체크포인트: {ckpt_type}  →  로드된 클래스: {loaded_type}\n"
            f"  가중치가 로드되지 않아 무작위 출력이 나옵니다 (점수 무의미).\n"
            f"  해결: pip install -U transformers 로 지원 여부 확인, 또는 전용 어댑터 필요\n"
        )
    return model


class QwenVLM:
    def __init__(self, model_name: str = config.DEFAULT_MODEL,
                 quant: str | None = None, trust_remote_code: bool = False):
        """quant: None(bf16) / '4bit' / '8bit' (bitsandbytes 온더플라이 양자화).

        속도가 목적이면 quant보다 AWQ 체크포인트를 권장:
          --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ  (+ pip install autoawq)
        AWQ는 융합 커널이라 빠르고 VRAM ~1/3, bnb 4bit는 VRAM 절감용(속도는 비슷하거나 느림).

        trust_remote_code: 모델 저장소의 코드를 실행합니다 (InternVL 등 일부 모델에 필요).
        신뢰할 수 있는 공식 저장소에만 사용하세요.
        """
        from transformers import AutoProcessor

        # T4(Turing)는 bf16 미지원 → fp16으로 자동 하향. Ampere+(A100/L4)는 bf16.
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        kwargs = dict(
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=_attn_implementation(),
        )
        if quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if quant == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # InternVL·Phi 계열은 저장소 코드가 있어야 로드됨 (공식 저장소만 자동 허용)
        if not trust_remote_code and model_name.startswith(AUTO_TRUST_PREFIXES):
            print(f"[vlm] {model_name}: 공식 저장소이므로 trust_remote_code 자동 활성화")
            trust_remote_code = True
        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        self.model = _load_backbone(model_name, kwargs)
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        self.model_name = model_name
        self._image_check_done = False       # 첫 호출에서 이미지 입력 여부 1회 검사
        print(f"[vlm] {model_name} | {type(self.model).__name__}"
              f" | dtype={dtype} | attn={kwargs['attn_implementation']}"
              f" | quant={quant or ('awq' if 'awq' in model_name.lower() else 'none')}")

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
        inputs = self.processor(text=[text], images=frames, return_tensors="pt")

        # 모델마다 이미지 플레이스홀더 규약이 달라(예: Phi-4의 <|image_N|>) 프레임이
        # 조용히 무시될 수 있다. 그 경우 점수가 조용히 망가지므로 명시적으로 실패시킨다.
        if not self._image_check_done:
            if not any(k in inputs for k in IMAGE_INPUT_KEYS):
                raise SystemExit(
                    f"\n[vlm] {self.model_name}: 프로세서 출력에 이미지 입력이 없습니다.\n"
                    f"  받은 키: {sorted(inputs.keys())}\n"
                    f"  이 모델은 chat template의 이미지 규약이 달라 전용 어댑터가 필요합니다.\n"
                    f"  (이대로 두면 모델이 영상을 못 보고 텍스트만으로 답해 점수가 왜곡됩니다)\n"
                )
            self._image_check_done = True
        return inputs.to(self.model.device)

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
                        times: list[float] | None = None,
                        prior: dict[str, float] | None = None) -> dict[str, float]:
        """'ANSWER:' 다음 첫 토큰의 로그확률로 각 후보 단어를 스코어링.

        자유 생성 대신 모델의 확신을 직접 읽는다 — 생성 노이즈/파싱 실패 제거.
        words 예: ["A","B","C","D"] 또는 ["YES","NO"].

        prior: 보기 글자별 사전 편향 (예: {"A":0.31,"B":0.24,...}).
        주면 log(prior)를 빼서 위치 편향을 제거한다 (PriDe/contextual calibration).
        VLM은 특정 글자를 선호하는 경향이 있어 객관식 정확도를 깎는다 —
        src/check_option_bias.py로 측정 후 --option-prior로 주입.
        """
        inputs = self._build_inputs(frames, prompt, times, assistant_prefix="ANSWER:")
        logits = self.model(**inputs).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        scores = {}
        for w in words:
            ids = self._first_token_ids(w)
            scores[w] = float(torch.logsumexp(logprobs[ids], dim=0))
        if prior:
            # 사용 가능한 보기에 대해서만 prior 재정규화 (HARn single은 3지선다)
            avail = {w: prior[w] for w in words if w in prior and prior[w] > 0}
            if len(avail) == len(words):
                z = sum(avail.values())
                scores = {w: s - math.log(avail[w] / z) for w, s in scores.items()}
        return scores

    @torch.inference_mode()
    def score_candidates(self, frames, prompt: str, candidates: list[str],
                         times: list[float] | None = None,
                         prefix: str = "ANSWER: ") -> dict[str, float]:
        """후보 문자열들이 prefix 뒤에 올 로그확률(토큰 합)을 각각 계산.

        sequence처럼 답이 '여러 글자의 순열'인 경우, 자유 생성 후 파싱하는 대신
        가능한 답을 전부 채점해 최댓값을 고른다 → **파싱 실패가 원천적으로 없고
        항상 유효한 순열이 나온다**. (모델이 지시 형식을 안 지켜도 안전)
        """
        base = self._build_inputs(frames, prompt, times, assistant_prefix=prefix)
        n0 = int(base["input_ids"].shape[1])

        scores = {}
        for cand in candidates:
            inp = self._build_inputs(frames, prompt, times,
                                     assistant_prefix=prefix + cand)
            ids = inp["input_ids"][0]
            if ids.shape[0] <= n0:               # 토큰 경계가 어긋난 경우 방어
                scores[cand] = float("-inf")
                continue
            logprobs = torch.log_softmax(self.model(**inp).logits[0].float(), dim=-1)
            idx = torch.arange(n0, ids.shape[0], device=ids.device)
            scores[cand] = float(logprobs[idx - 1, ids[idx]].sum())
        return scores

    @torch.inference_mode()
    def yes_probability(self, frames, prompt: str,
                        times: list[float] | None = None) -> float:
        """이진 질의의 P(YES) — YES/NO 두 스코어를 정규화한 상대 확률."""
        lp = self.option_logprobs(frames, prompt, ["YES", "NO"], times)
        pair = torch.tensor([lp["YES"], lp["NO"]])
        return float(torch.softmax(pair, dim=0)[0])


def load_model(name: str, quant: str | None = None,
               trust_remote_code: bool = False):
    return QwenVLM(name, quant=quant, trust_remote_code=trust_remote_code)
