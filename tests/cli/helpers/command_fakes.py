import inspect
from typing import ClassVar

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

REAL_SAVE = inspect.signature(GeneratedImage.save)


class FakeImage:
    def __init__(self, seed: int):
        self.seed = seed
        self.saves: list[tuple[str, bool]] = []

    def save(self, *args, **kwargs):
        bound = REAL_SAVE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self.saves.append((str(bound.arguments["path"]), bound.arguments["export_json_metadata"]))


class FakeModel:
    # Stands in for a model class at the one boundary a CLI owns nothing behind: records how it
    # was built and what it was asked to generate, never touches weights or Metal. Every call is
    # bound against the real signatures, captured before any monkeypatching, so a kwarg the real
    # model does not take fails the test.
    real: ClassVar[type]
    init_signature: ClassVar[inspect.Signature]
    generate_signature: ClassVar[inspect.Signature]
    instances: ClassVar[list["FakeModel"]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.init_signature = inspect.signature(cls.real.__init__)
        cls.generate_signature = inspect.signature(cls.real.generate_image)
        cls.instances = []

    def __init__(self, **kwargs):
        bound = type(self).init_signature.bind(self, **kwargs)
        bound.apply_defaults()
        # init_kwargs is what the CLI passed; model_config is what the real model would hold,
        # its own default included when the CLI passes none.
        self.init_kwargs = kwargs
        self.model_config = bound.arguments.get("model_config")
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.generate_calls: list[dict] = []
        self.before_loop_count_at_generate: list[int] = []
        self.images: list[FakeImage] = []
        type(self).instances.append(self)

    def bind_generate(self, **kwargs):
        type(self).generate_signature.bind(self, **kwargs)

    def generate_image(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        self.before_loop_count_at_generate.append(len(self.callbacks.before_loop))
        image = FakeImage(kwargs["seed"])
        self.images.append(image)
        return image


def reset_cli_globals(monkeypatch):
    # parse_args sets GeneratedImage.model_path and can switch off ImageUtil.embed_metadata_enabled
    # for the whole process; reset both so tests cannot leak into each other or the suite.
    monkeypatch.setattr(GeneratedImage, "model_path", None)
    monkeypatch.setattr(ImageUtil, "embed_metadata_enabled", True)
