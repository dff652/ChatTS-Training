import pytest

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.template import TEMPLATES


def test_qwen3_vl_plugin_is_registered():
    plugin = get_mm_plugin(
        name="qwen3_vl",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
    )

    assert plugin.__class__.__name__ == "Qwen3VLPlugin"
    assert TEMPLATES["qwen3_vl"].__class__.__name__ == "ReasoningTemplate"
    assert TEMPLATES["qwen3_vl_nothink"].__class__.__name__ == "Template"
    assert TEMPLATES["qwen3_vl_nothink"].mm_plugin.__class__.__name__ == "Qwen3VLPlugin"


def test_qwen3_vl_plugin_rejects_video_training():
    plugin = get_mm_plugin(
        name="qwen3_vl",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
    )

    with pytest.raises(ValueError, match="image-only backport"):
        plugin.process_messages(
            messages=[{"role": "user", "content": "<video>describe"}],
            images=[],
            videos=["video.mp4"],
            audios=[],
            processor=None,
        )
