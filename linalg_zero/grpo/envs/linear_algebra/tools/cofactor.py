from typing import Any

from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.grpo.envs.tool import Tool
from linalg_zero.shared.lib import matrix_cofactor


class Cofactor(Tool):
    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: Any) -> str:
        try:
            return str(matrix_cofactor(**kwargs))
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def get_info() -> dict[str, Any]:
        return get_json_schema(matrix_cofactor)
