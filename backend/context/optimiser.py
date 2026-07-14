from llmlingua import PromptCompressor
from llm import utility_llm
import logging

logger = logging.getLogger(__name__)


class Optimiser:

    def __init__(self) -> None:
        self._compressor = None

    def _get_compressor(self) -> PromptCompressor:
        if self._compressor is None:
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )

        return self._compressor

    def compress(self, texts: list[str], target_tokens: int) -> str:
        try:
            result = self._get_compressor().compress_prompt(
                context=texts, target_token=target_tokens
            )
            return str(result["compressed_prompt"])
        except Exception as e:
            logger.warning(
                msg=f"WARN: Exception Occured. Failed to compress prompt using compressor. Reason: {str(e)}"
            )
            prompt = f"Summarise below in ~{target_tokens} tokens:\n\n" + "\n\n".join(
                texts
            )

            return utility_llm.invoke(prompt).content
