"""Shared processor/collator wrappers used by the training workflow."""

from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image


class ThinkingOffProcessor:
    def __init__(self, processor):
        self._processor = processor

    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self._processor.apply_chat_template(messages, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._processor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._processor, name)


def _as_pil_image(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value from dataset: {type(value)}")


class ResponseOnlyVisionDataCollator:
    def __init__(self, model, processor, max_seq_length):
        from unsloth.trainer import UnslothVisionDataCollator

        self.processor = processor
        self.base = UnslothVisionDataCollator(model, processor, max_seq_length=max_seq_length, resize="max")

    def __call__(self, examples):
        batch = self.base(examples)
        for row, example in enumerate(examples):
            prompt_messages = example["messages"][:-1]
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            images = [_as_pil_image(part["image"]) for message in prompt_messages for part in message["content"] if part.get("type") == "image"]
            prompt_batch = self.processor(text=prompt_text, images=images, return_tensors="pt", add_special_tokens=False)
            prefix_length = int(prompt_batch["input_ids"].shape[-1])
            if prefix_length > batch["labels"].shape[1]:
                raise ValueError("The prompt was truncated before the target JSON.")
            batch["labels"][row, :prefix_length] = -100
            end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            end_positions = torch.where(batch["input_ids"][row, prefix_length:] == end_token_id)[0]
            if len(end_positions) == 0:
                raise ValueError("The target JSON closing <|im_end|> token was not found.")
            batch["labels"][row, prefix_length + int(end_positions[0]) + 1 :] = -100
        return batch
