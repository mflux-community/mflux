__all__ = ["QwenImage21Edit"]


def __getattr__(name: str) -> type:
    if name == "QwenImage21Edit":
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        return QwenImage21Edit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
