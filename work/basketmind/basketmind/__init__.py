"""BasketMind: can a small pretrained language model learn what sells together?"""

import os
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # GPT-2 comes from the local Hugging Face cache
warnings.filterwarnings("ignore", message=r"(?s).*loss_type=None.*")


def _hide_broken_torchaudio() -> None:
    """transformers imports torchaudio when it is installed; a build for another torch crashes.

    We never use audio. If torchaudio fails to load, tell transformers it is absent. In a
    consistent environment (e.g. the lfretail workbench) this does nothing.
    """
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        return
    except OSError:
        import transformers.utils as tu
        import transformers.utils.import_utils as iu
        iu.is_torchaudio_available = lambda: False
        tu.is_torchaudio_available = iu.is_torchaudio_available


_hide_broken_torchaudio()
__version__ = "0.1.0"
