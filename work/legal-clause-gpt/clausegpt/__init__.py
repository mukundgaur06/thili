"""Legal-clause GPT: fine-tune GPT-2 to tag LEDGAR contract clauses with their type."""


def _hide_broken_torchaudio() -> None:
    """transformers imports torchaudio whenever it is installed, even for a text model.

    If the installed torchaudio was built for a different torch, that import crashes with
    an OSError. We never use audio, so in that case tell transformers torchaudio is absent.
    In a consistent environment (e.g. the lfllm workbench) this does nothing.
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
