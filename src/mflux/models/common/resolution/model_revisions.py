class ModelRevisions:
    # The HuggingFace commit each built-in model repo loads from. Without a pin, a cached snapshot is used
    # as it is and a new download takes main, so two installs of the same mflux version can run different
    # weights when a repo changes (Qwen-Image-2.1 replaced its transformer while it was being ported, and its
    # golden reference kept the old weights).
    # To move a pin, change the hash here and regenerate that model's golden references in the same PR.
    # A local path, a mflux save or a repo that is not listed loads as before.
    PINNED: dict[str, str] = {
        "AiArtLab/qwen3-0.6b-4b-adapter": "26a3064438fb32f5d51b482cfb1f7fa642ff0b0d",
        "AiArtLab/zen-image-edit": "3d5717559ff5bd8b431e66032b285a3fa4a1b20c",
        "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union": "8a4702014d4dabb5f896fcba917e2ee0a961465f",
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1": "5155fc56d17821007d6f62ac192c09e0f0e72016",
        "baidu/ERNIE-Image": "5346b31d68c9c23758ba56ef8be5e9dc174c7f99",
        "baidu/ERNIE-Image-Turbo": "bc68c81e2a1730a394d5fc9fae70713dee940140",
        "black-forest-labs/FLUX.1-Depth-dev": "fb5e9b1bae41b8c8adcea4ea2a87b74dd298f07a",
        "black-forest-labs/FLUX.1-dev": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        "black-forest-labs/FLUX.1-Fill-dev": "358293da0354175698b67ec8299acf928313a78a",
        "black-forest-labs/FLUX.1-Kontext-dev": "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
        "black-forest-labs/FLUX.1-Krea-dev": "8162a9c7b05a641be098422bf2fcf335615c2f28",
        "black-forest-labs/FLUX.1-Redux-dev": "c95859fbf7703ca4d6824b4da4407d7cd0434f81",
        "black-forest-labs/FLUX.1-schnell": "741f7c3ce8b383c54771c7003378a50191e9efe9",
        "black-forest-labs/flux-3-image-klein": "8454017f9ebaada30c998fccd9f67a80c7580f68",
        "black-forest-labs/flux-3-image-klein-base": "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4",
        "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "black-forest-labs/FLUX.2-klein-9B": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
        "black-forest-labs/FLUX.2-klein-9b-kv": "a6dfb36eca3a3906eb2fd460795adfb844e5fcce",
        "black-forest-labs/FLUX.2-klein-base-4B": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        "black-forest-labs/FLUX.2-klein-base-9B": "32773329fbe7e81a90ef971740e8ba4b0364ecf3",
        "black-forest-labs/FLUX.2-small-decoder": "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a",
        "Boogu/Boogu-Image-0.1-Turbo": "53ad54522023f64d049f7f38e4d679359ef3fb92",
        "briaai/FIBO": "7c7e00ca081f6cd24bbbd745d13cc214484c9199",
        "briaai/Fibo-2.0-dev": "fb2b781f9e9e5572a6c348a5af7a4d155a2b1ed0",
        "briaai/Fibo-2.0-edit-turbo": "901c02672d969ed21e44ff58f1a0872c7581ddb0",
        "briaai/Fibo-Edit": "1bee4ae9b119d2634ec9f378370f2a1ae49d89cc",
        "briaai/Fibo-Edit-RMBG": "cb134b058d1a6089fc3b9e5fc04a4511337b53ed",
        "briaai/Fibo-lite": "dcd26dfe1051e59b77b88acbacc025613aaa09de",
        "briaai/FIBO-vlm": "bd13da5e0b73c808014d6260963e311c8cdf0d8b",
        "Comfy-Org/Lens": "393618fd7db3e370509f84390b9162b8fd839a25",
        "google/gemma-2-2b-it": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        "ideogram-ai/ideogram-4-fp8": "ee79a7237b519f1402ceacf952f30c8a31ec5073",
        "inclusionAI/Ming-Image-0.1-Design": "208087ada1486931692c1896f38d4cd16ff3df82",
        "InstantX/FLUX.1-dev-Controlnet-Canny": "e7cee4b2afa335bf8f913f2b044764b1a1b01881",
        "jasperai/Flux.1-dev-Controlnet-Upscaler": "866bf7df8056351506e264551a37a3035e571954",
        "krea/Krea-2-Raw": "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603",
        "krea/Krea-2-Turbo": "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
        "lllyasviel/Annotators": "982e7edaec38759d914a963c48c4726685de7d96",
        "mlx-community/gpt-oss-20b-MXFP4-Q8": "773a7da77e569019bb0fd17a554b263738d669a3",
        "numz/SeedVR2_comfyUI": "09ced71023636e9bc8cdf9cdecfb2625d1e691e8",
        "nvidia/PiD": "1b9b0872cef786e527ecc5f12b5ab2e6a02d55a4",
        "OpenMOSS-Team/MOSS-Image-Base": "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        "OpenMOSS-Team/MOSS-Image-SFT": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
        "Patil/Krea-2-depth-controlnet": "21889413aa7282a6e78bd510247cceccad034b24",
        "Qwen/Qwen-Image-2.1": "d26bb61231c349cf6b7896fa83353113880e1ba3",
        "Qwen/Qwen-Image-2.1-Turbo": "fc13ba1986fcb905854dc8de2f52eee395f49b5b",
        "Qwen/Qwen-Image-2512": "25468b98e3276ca6700de15c6628e51b7de54a26",
        "Qwen/Qwen-Image-Edit-2509": "d3968ef930e841f4c73640fb8afa3b306a78167e",
        "Qwen/Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "speridlabs/iris-3b": "adc05b909fb05d4276d26aae5a52b36406b2ef26",
        "Tongyi-MAI/Z-Image": "04cc4abb7c5069926f75c9bfde9ef43d49423021",
        "Tongyi-MAI/Z-Image-Turbo": "f332072aa78be7aecdf3ee76d5c247082da564a6",
        "xiaozaa/catvton-flux-beta": "4088ab884c686e07396d1eb12cf10bf0facb65e0",
    }

    @staticmethod
    def get(repo_id: str | None) -> str | None:
        return ModelRevisions.PINNED.get(repo_id) if repo_id else None
