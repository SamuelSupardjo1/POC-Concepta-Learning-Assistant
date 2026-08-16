import requests
import unicodedata


class OllamaLLM:
    """
    Ollama LLM wrapper for the Intelligent Learning Assistant.

    Uses a local Ollama server and generates an answer from
    the provided grounded prompt.

    The generation configuration is intentionally conservative
    because the assistant must produce concise, lesson-grounded
    answers rather than creative responses.
    """

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.0,
        num_predict: int = 180,
    ) -> None:

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict

    def _sanitize_answer(self, answer: str) -> str:
        """
        Sanitize LLM output to prevent encoding errors on Windows.

        Replaces characters that cannot be encoded in cp1252
        (e.g. unicode checkmarks, arrows, math symbols) with
        their closest ASCII equivalent or removes them.

        Uses NFKD normalization first to decompose characters
        (e.g. accented letters become base + diacritic), then
        strips anything not representable in ASCII.
        """

        if not answer:
            return answer

        # Decompose unicode (e.g. é -> e + combining accent)
        normalized = unicodedata.normalize("NFKD", answer)

        # Keep only the ASCII-encodable characters
        ascii_bytes = normalized.encode("ascii", errors="ignore")
        sanitized = ascii_bytes.decode("ascii")

        # If stripping removed everything (e.g. full CJK answer),
        # fall back to cp1252-safe replacement so we at least get
        # a printable string rather than an empty result.
        if sanitized.strip():
            return sanitized.strip()

        return answer.encode("cp1252", errors="replace").decode("cp1252")

    def generate(self, prompt: str) -> str:

        if not prompt or not prompt.strip():
            return ""

        print()
        print("=== LLM GENERATION ===")
        print(f"Model: {self.model}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Temperature: {self.temperature}")
        print(f"Max output tokens: {self.num_predict}")
        print("Sending request to Ollama...")

        try:

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,

                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=self.timeout,
            )

            print(
                f"Ollama HTTP status: "
                f"{response.status_code}"
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get(
                "response",
                "",
            )

            print("LLM generation completed.")

            print(
                f"Answer length: "
                f"{len(answer)} characters"
            )

            return self._sanitize_answer(answer.strip())

        except requests.exceptions.Timeout:

            print(
                "ERROR: Ollama request timed out."
            )

            return ""

        except requests.exceptions.ConnectionError:

            print(
                "ERROR: Cannot connect to Ollama."
            )

            print(
                "Make sure Ollama is running:"
            )

            print(
                "ollama serve"
            )

            return ""

        except requests.exceptions.RequestException as error:

            print(
                f"ERROR: Ollama request failed: "
                f"{error}"
            )

            return ""

        except Exception as error:

            print(
                f"ERROR: Unexpected LLM error: "
                f"{error}"
            )

            return ""