import argparse
import gc
import json
import os
import threading
from typing import Any

import gradio as gr
import spaces  # Import spaces early to enable ZeroGPU support
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from linalg_zero.distillation.components.models import DefaultConfig
from linalg_zero.distillation.data import FunctionInvocationInfo, ThoughtSchema
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.lib import get_lib, get_tools
from linalg_zero.shared.system_prompts import (ANSWER_CLOSE, ANSWER_OPEN,
                                               THINK_CLOSE, THINK_OPEN,
                                               TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
                                               TOOL_RESPONSE_CLOSE,
                                               TOOL_RESPONSE_OPEN,
                                               get_math_system_prompt)
from torch.utils._pytree import tree_map
from transformers import (AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer, pipeline)

# Global event to signal cancellation from the UI thread to the generation thread
cancel_event = threading.Event()

access_token = os.environ["HF_TOKEN"]
DEMO_DIR = os.path.dirname(__file__)
ASSISTANT_AVATAR_PATH = os.path.join(DEMO_DIR, "assets", "linalgzero-avatar.svg")

# Optional: Disable GPU visibility if you wish to force CPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = " "

# ------------------------------
# Allowed model definitions
# ------------------------------
MODELS = {
    "Qwen3-1.7B": {
        "repo_id": "Qwen/Qwen3-1.7B",
        "description": "Dense causal language model with 1.7B parameters.",
        "params_b": 1.7,
    },
    "atomwalk12/LinAlgZero-GRPO": {
        "repo_id": "atomwalk12/LinAlgZero-GRPO-merged",
        "description": "LinAlgZero GRPO fine-tuned model.",
        "params_b": 3.0,
    },
    "atomwalk12/LinalgZero-SFT": {
        "repo_id": "atomwalk12/LinalgZero-SFT",
        "description": "LinAlgZero SFT fine-tuned model.",
        "params_b": 3.0,
    },
}
# Global cache for pipelines to avoid re-loading.
PIPELINES = {}
MODEL_SNAPSHOT_PATHS: dict[str, str] = {}
SYSTEM_PROMPT = get_math_system_prompt(include_examples=False)
TOOL_LIBRARY = get_lib()
TOOL_SCHEMAS = get_tools()
PARSER = XMLParser()
MODEL_MESSAGE_CONFIG = DefaultConfig()
TRACE_TITLES = {"💭 Thought", "🛠 Tool Call", "📦 Tool Response"}
KEEP_SPECIAL_TOKENS = {
    ANSWER_OPEN,
    ANSWER_CLOSE,
    THINK_OPEN,
    THINK_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN,
    TOOL_RESPONSE_CLOSE,
}
SIMPLE_EXAMPLE_QUERIES = [
    "What is the rank of matrix A = [[2, 3], [2, -4]]?",
    "Step 1: find the transpose of A = [[1, 2], [3, 4]]. Step 2: find the trace of the result.",
    "Find the determinant of [[3, 1], [2, 5]].",
    "Find the Frobenius norm of [[3, -2], [-1, 5]].",
    "Find the cofactor matrix of [[5, 2], [1, 3]].",
]


def _parse_launch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable a public Gradio share link.",
    )
    args, _ = parser.parse_known_args()
    return args


def _extract_example_query(example: dict[str, Any]) -> str | None:
    query = example.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()

    messages = example.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None

    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _truncate_example_label(query: str, max_chars: int = 96) -> str:
    one_line_query = " ".join(query.split())
    if len(one_line_query) <= max_chars:
        return one_line_query
    return one_line_query[: max_chars - 3] + "..."


DATASET_EXAMPLE_REPO = "atomwalk12/linalgzero-grpo"
DATASET_EXAMPLE_SPLIT = "test"


def _load_dataset_example_queries(limit: int = 10) -> list[str]:
    try:
        dataset = load_dataset(
            DATASET_EXAMPLE_REPO,
            split=DATASET_EXAMPLE_SPLIT,
            streaming=True,
        )
    except Exception:
        return []

    queries: list[str] = []
    for example in dataset:
        query = _extract_example_query(example)
        if not query:
            continue
        queries.append(query)
        if len(queries) >= limit:
            break
    return queries


DATASET_EXAMPLE_QUERIES = _load_dataset_example_queries()
DATASET_EXAMPLE_CHOICES = [(_truncate_example_label(query), query) for query in DATASET_EXAMPLE_QUERIES]
SIMPLE_EXAMPLE_ROWS = [[query] for query in SIMPLE_EXAMPLE_QUERIES]
if DATASET_EXAMPLE_QUERIES:
    DATASET_EXAMPLE_INFO = f"Loaded from {DATASET_EXAMPLE_REPO} ({DATASET_EXAMPLE_SPLIT})"
else:
    DATASET_EXAMPLE_INFO = "Dataset examples unavailable"

EVAL_TEMPERATURE = 0.0
UI_TOP_K_DEFAULT = 40
UI_TOP_P_DEFAULT = 1.0
EVAL_REPETITION_PENALTY = 1.0
DEFAULT_MAX_TOKENS = 1024
ZERO_GPU_DURATION_SECONDS = 120

def prepare_model_artifacts(model_name: str, *, local_files_only: bool = False) -> str:
    cached_path = MODEL_SNAPSHOT_PATHS.get(model_name)
    if cached_path is not None and os.path.exists(cached_path):
        return cached_path

    repo = MODELS[model_name]["repo_id"]
    model_path = snapshot_download(
        repo_id=repo,
        token=access_token,
        local_files_only=local_files_only,
    )
    MODEL_SNAPSHOT_PATHS[model_name] = model_path
    return model_path


def load_pipeline(model_name):
    """
    Load and cache a transformers pipeline for text generation.
    Tries bfloat16, falls back to float16 or float32 if unsupported.
    """
    global PIPELINES
    if model_name in PIPELINES:
        return PIPELINES[model_name]
    model_path = prepare_model_artifacts(model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=access_token,
        local_files_only=True,
    )
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            pipe = pipeline(
                task="text-generation",
                model=model_path,
                tokenizer=tokenizer,
                trust_remote_code=True,
                dtype=dtype, # Use `dtype` instead of deprecated `torch_dtype`
                device_map="auto",
                use_cache=True,      # Enable past-key-value caching
                token=access_token,
                local_files_only=True,
            )
            PIPELINES[model_name] = pipe
            return pipe
        except Exception:
            continue
    # Final fallback
    pipe = pipeline(
        task="text-generation",
        model=model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        use_cache=True,
        local_files_only=True,
    )
    PIPELINES[model_name] = pipe
    return pipe

def _extract_visible_title(message: dict[str, Any]) -> str | None:
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if isinstance(title, str):
            return title
    return None


def _build_llm_messages(chat_history: list[dict[str, Any]] | None, user_msg: str) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history or []:
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if role == "assistant" and _extract_visible_title(msg) in TRACE_TITLES:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg})
    return messages


def _render_prompt(messages: list[dict[str, Any]], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": TOOL_SCHEMAS,
        }
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    prompt = messages[0]["content"].strip() + "\n"
    for msg in messages[1:]:
        if msg["role"] == "user":
            prompt += "User: " + str(msg["content"]).strip() + "\n"
        elif msg["role"] == "assistant":
            prompt += "Assistant: " + str(msg.get("content", "")).strip() + "\n"
            for tool_call in msg.get("tool_calls", []):
                function_info = tool_call.get("function", {})
                prompt += (
                    f"{TOOL_CALL_OPEN}"
                    + json.dumps({
                        "name": function_info.get("name"),
                        "arguments": function_info.get("arguments"),
                    })
                    + f"{TOOL_CALL_CLOSE}\n"
                )
        elif msg["role"] == "tool":
            prompt += f"User: {TOOL_RESPONSE_OPEN}{msg['content']}{TOOL_RESPONSE_CLOSE}\n"
    if not prompt.strip().endswith("Assistant:"):
        prompt += "Assistant: "
    return prompt


class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_token_sequences: list[list[int]]) -> None:
        super().__init__()
        self.stop_token_sequences = [seq for seq in stop_token_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if cancel_event.is_set():
            return True

        current_tokens = input_ids[0].tolist()
        for seq in self.stop_token_sequences:
            if len(current_tokens) >= len(seq) and current_tokens[-len(seq):] == seq:
                return True
        return False


def _get_stop_token_sequences(tokenizer: AutoTokenizer) -> list[list[int]]:
    stop_strings = [TOOL_CALL_CLOSE, ANSWER_CLOSE, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE]
    return [tokenizer.encode(stop_text, add_special_tokens=False) for stop_text in stop_strings]


def _decode_generated_tokens(tokenizer: AutoTokenizer, generated_tokens: torch.Tensor) -> str:
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    cleaned_output = raw_output
    for special_token in tokenizer.all_special_tokens:
        if special_token not in KEEP_SPECIAL_TOKENS:
            cleaned_output = cleaned_output.replace(special_token, "")
    return cleaned_output


def _clean_decoded_text(tokenizer: AutoTokenizer, text: str) -> str:
    cleaned_text = text
    for special_token in tokenizer.all_special_tokens:
        if special_token not in KEEP_SPECIAL_TOKENS:
            cleaned_text = cleaned_text.replace(special_token, "")
    return cleaned_text


def _extract_partial_tag_content(message: str, tag: str) -> str | None:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start_idx = message.rfind(open_tag)
    if start_idx == -1:
        return None
    content = message[start_idx + len(open_tag):]
    end_idx = content.find(close_tag)
    if end_idx != -1:
        content = content[:end_idx]
    return content.strip()


def _extract_live_sections(message: str) -> dict[str, str | None]:
    return {
        "thought": _extract_partial_tag_content(message, "think"),
        "tool_call": _extract_partial_tag_content(message, "tool_call"),
        "answer": _extract_partial_tag_content(message, "answer"),
    }


def _set_live_message(
    history: list[dict[str, Any]],
    live_indices: dict[str, int],
    key: str,
    content: str | None,
    *,
    title: str | None = None,
    code_block_lang: str | None = None,
) -> None:
    if not content:
        return

    rendered_content = content
    if code_block_lang is not None:
        rendered_content = f"```{code_block_lang}\n{content}\n```"

    idx = live_indices.get(key)
    if idx is None:
        message: dict[str, Any] = {"role": "assistant", "content": rendered_content}
        if title is not None:
            message["metadata"] = {"title": title}
        history.append(message)
        live_indices[key] = len(history) - 1
        return

    history[idx]["content"] = rendered_content


def _build_debug_info() -> str:
    tool_lines = "\n".join(f"- `{tool_name}`" for tool_name in sorted(TOOL_LIBRARY.keys()))
    return f"### Available Tools\n{tool_lines}\n\n### System Prompt\n```text\n{SYSTEM_PROMPT}\n```"


def _stream_assistant_turn(
    pipe: Any,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
):
    prompt = _render_prompt(messages, pipe.tokenizer)
    inputs = pipe.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=bool(getattr(pipe.tokenizer, "pad_token_id", None)),
    )
    device = getattr(pipe, "device", None) or pipe.model.device
    inputs = tree_map(lambda x: x.to(device) if hasattr(x, "to") else x, inputs)

    streamer = TextIteratorStreamer(
        pipe.tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )

    generation_kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": int(max_tokens),
        "repetition_penalty": float(repeat_penalty),
        "use_cache": True,
        "pad_token_id": getattr(pipe.tokenizer, "pad_token_id", pipe.tokenizer.eos_token_id),
        "stopping_criteria": StoppingCriteriaList([StopOnSequences(_get_stop_token_sequences(pipe.tokenizer))]),
        "streamer": streamer,
    }
    eos_token_id = getattr(pipe.tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if temperature > 0:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
        })
    else:
        generation_kwargs["do_sample"] = False

    error_holder: dict[str, Exception] = {}

    def _run_generation() -> None:
        try:
            with torch.inference_mode():
                pipe.model.generate(**generation_kwargs)
        except Exception as exc:
            error_holder["error"] = exc
        finally:
            streamer.end()

    generation_thread = threading.Thread(target=_run_generation, daemon=True)
    generation_thread.start()

    raw_output = ""
    yield raw_output, prompt
    for chunk in streamer:
        raw_output = _clean_decoded_text(pipe.tokenizer, raw_output + chunk)
        yield raw_output, prompt

    generation_thread.join()
    if "error" in error_holder:
        raise error_holder["error"]


def _parse_assistant_output(raw_output: str, context: list[dict[str, Any]]) -> tuple[ThoughtSchema | None, dict[str, Any]]:
    analysis = PARSER.analyze_message_in_context(
        context,
        message=raw_output,
        tool_names=list(TOOL_LIBRARY.keys()),
    )
    if not bool(analysis["is_valid_think_then_tool_or_answer"]):
        return None, analysis
    if analysis["has_answer"] and not bool(analysis["answer_policy_valid"]):
        return None, analysis

    tool_call: FunctionInvocationInfo | None = None
    tool_info = analysis["tool"]
    if analysis["has_tool_call"]:
        if not bool(tool_info["json_valid"]):
            return None, analysis
        tool_call = FunctionInvocationInfo(
            name=str(tool_info["name"]),
            arguments=dict(tool_info["arguments"]),
        )

    return ThoughtSchema(
        thought=analysis["thought"] or "",
        tool_call=tool_call,
        final_answer=analysis["answer"],
        completed=analysis["answer"] is not None,
    ), analysis


def _execute_tool_call(message: ThoughtSchema) -> dict[str, str]:
    if message.tool_call is None:
        raise ValueError("Tool call is required")

    name = message.tool_call.name
    arguments = message.tool_call.arguments
    try:
        if name not in TOOL_LIBRARY:
            return {
                "function_name": name,
                "execution_result": f"ERROR: Function '{name}' not found in library",
            }
        result = TOOL_LIBRARY[name](**arguments)
        return {"function_name": name, "execution_result": str(result)}
    except Exception as exc:
        return {
            "function_name": name,
            "execution_result": f"ERROR: {type(exc).__name__}: {exc}",
        }


# Keep a single GPU allocation for the whole multi-turn solve, with a fixed
# budget that is simpler and more predictable than a heuristic estimate.
@spaces.GPU(duration=ZERO_GPU_DURATION_SECONDS)
def chat_response(
    user_msg,
    chat_history,
    show_tool_trace,
    enable_streaming,
    max_tool_turns,
    model_name,
    max_tokens,
    temperature,
    top_k,
    top_p,
    repeat_penalty,
):
    """
    Generates responses by iteratively calling linear algebra tools until a final answer is produced.
    """
    cancel_event.clear()

    history = list(chat_history or [])
    history.append({"role": "user", "content": user_msg})
    debug_info = _build_debug_info()

    try:
        pipe = load_pipeline(model_name)
        context = _build_llm_messages(chat_history, user_msg)

        yield history, debug_info

        for step_idx in range(int(max_tool_turns)):
            raw_output = ""
            live_indices: dict[str, int] = {}

            turn_stream = _stream_assistant_turn(
                pipe=pipe,
                messages=context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )

            if enable_streaming:
                for partial_output, _current_prompt in turn_stream:
                    raw_output = partial_output

                    live_sections = _extract_live_sections(raw_output)
                    if show_tool_trace:
                        _set_live_message(
                            history,
                            live_indices,
                            "thought",
                            live_sections["thought"],
                            title="💭 Thought",
                        )
                        _set_live_message(
                            history,
                            live_indices,
                            "tool_call",
                            live_sections["tool_call"],
                            title="🛠 Tool Call",
                            code_block_lang="json",
                        )
                    _set_live_message(
                        history,
                        live_indices,
                        "answer",
                        live_sections["answer"],
                    )

                    yield history, debug_info
            else:
                for partial_output, _current_prompt in turn_stream:
                    raw_output = partial_output

            if cancel_event.is_set():
                history.append({"role": "assistant", "content": "[Generation Canceled]"})
                yield history, debug_info
                break

            parsed, analysis = _parse_assistant_output(raw_output, context)
            if parsed is None:
                history.append({
                    "role": "assistant",
                    "content": "I couldn't produce a valid tool call or final answer for this problem.",
                })
                yield history, debug_info
                break

            assistant_message = MODEL_MESSAGE_CONFIG.format_assistant_message(parsed)
            if assistant_message is None:
                history.append({"role": "assistant", "content": "The model returned an empty action."})
                yield history, debug_info
                break

            context.append(assistant_message)

            if show_tool_trace and parsed.thought:
                _set_live_message(
                    history,
                    live_indices,
                    "thought",
                    parsed.thought,
                    title="💭 Thought",
                )

            if parsed.tool_call is not None:
                if show_tool_trace:
                    _set_live_message(
                        history,
                        live_indices,
                        "tool_call",
                        json.dumps(
                            {
                                "name": parsed.tool_call.name,
                                "arguments": parsed.tool_call.arguments,
                            },
                            indent=2,
                        ),
                        title="🛠 Tool Call",
                        code_block_lang="json",
                    )

                tool_result = _execute_tool_call(parsed)
                context.append(MODEL_MESSAGE_CONFIG.create_tool_message(context, tool_result))

                if show_tool_trace:
                    history.append({
                        "role": "assistant",
                        "content": tool_result["execution_result"],
                        "metadata": {"title": "📦 Tool Response"},
                    })

                yield history, debug_info
                continue

            if parsed.final_answer is not None:
                answer_idx = live_indices.get("answer")
                if answer_idx is not None:
                    history[answer_idx]["content"] = parsed.final_answer
                else:
                    history.append({"role": "assistant", "content": parsed.final_answer})
                yield history, debug_info
                break
        else:
            history.append({
                "role": "assistant",
                "content": "I hit the tool-turn limit before finishing the problem.",
            })
            yield history, debug_info
    except GeneratorExit:
        print("Chat response cancelled.")
        return
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {e}"})
        yield history, debug_info
    finally:
        gc.collect()


demo_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
)

demo_css = """
    .chatbot { border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    button.primary { font-weight: 600; }
    .gradio-accordion { margin-bottom: 12px; }
"""

with gr.Blocks(title="LLM Inference with ZeroGPU") as demo:
    # Header
    gr.Markdown("""
    # 🧠 LinAlgZero Demo
    ### Multi-turn linear algebra solving with tool calling
    """)

    with gr.Row():
        # Left Panel - Configuration
        with gr.Column(scale=3):
            # Core Settings (Always Visible)
            with gr.Group():
                gr.Markdown("### ⚙️ Core Settings")
                model_dd = gr.Dropdown(
                    label="🤖 Model",
                    choices=list(MODELS.keys()),
                    value="atomwalk12/LinAlgZero-GRPO",
                    info="Select the language model to use"
                )
                trace_chk = gr.Checkbox(
                    label="🔍 Show Tool Trace",
                    value=True,
                    info="Show thoughts, tool calls, and tool responses in the chat"
                )
                stream_chk = gr.Checkbox(
                    label="⚡ Enable Live Streaming",
                    value=True,
                    info="Stream partial model output while each tool-calling turn is being generated"
                )
                max_tool_turns = gr.Slider(
                    1, 12, value=6, step=1,
                    label="Max Tool Turns",
                    info="Maximum number of tool-calling turns before stopping"
                )

            example_dropdown = gr.Dropdown(
                label="📚 Dataset Example",
                choices=DATASET_EXAMPLE_CHOICES,
                value=None,
                info=DATASET_EXAMPLE_INFO,
                allow_custom_value=False,
            )

            # Advanced Settings (Collapsible)
            with gr.Accordion("🎛️ Advanced Generation Parameters", open=True):
                max_tok = gr.Slider(
                    64, 16384, value=DEFAULT_MAX_TOKENS, step=32,
                    label="Max Tokens",
                    info="Maximum length of generated response"
                )
                temp = gr.Slider(
                    0.0, 2.0, value=EVAL_TEMPERATURE, step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                with gr.Row():
                    k = gr.Slider(
                        1, 100, value=UI_TOP_K_DEFAULT, step=1,
                        label="Top-K",
                        info="Number of top tokens to consider",
                        interactive=False,
                    )
                    p = gr.Slider(
                        0.1, 1.0, value=UI_TOP_P_DEFAULT, step=0.05,
                        label="Top-P",
                        info="Nucleus sampling threshold",
                        interactive=False,
                    )
                rp = gr.Slider(
                    1.0, 2.0, value=EVAL_REPETITION_PENALTY, step=0.1,
                    label="Repetition Penalty",
                    info="Penalize repeated tokens"
                )

            # Actions
            with gr.Column():
                reset_params_btn = gr.Button("↺ Restore Default Params", variant="secondary")
                clr = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

        # Right Panel - Chat Interface
        with gr.Column(scale=7):
            chat = gr.Chatbot(
                height=1200,
                label="💬 Conversation",
                buttons=["copy"],
                avatar_images=(None, ASSISTANT_AVATAR_PATH),
                layout="bubble"
            )

            # Input Area
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="💭 Type your message here... (Press Enter to send)",
                    scale=9,
                    container=False,
                    show_label=False,
                    lines=1,
                    max_lines=5
                )
                with gr.Column(scale=1, min_width=120):
                    submit_btn = gr.Button("📤 Send", variant="primary", size="lg", interactive=False)
                    cancel_btn = gr.Button("⏹️ Stop", variant="stop", visible=False, size="lg")

            # Example Prompts
            gr.Examples(
                examples=SIMPLE_EXAMPLE_ROWS,
                inputs=txt,
                label="💡 Example Prompts"
            )

            # Debug/Status Info (Collapsible)
            with gr.Accordion("🔍 Debug Info", open=False):
                dbg = gr.Markdown(value=_build_debug_info())

    # --- Event Listeners ---

    # Group all inputs for cleaner event handling
    chat_inputs = [txt, chat, trace_chk, stream_chk, max_tool_turns, model_dd, max_tok, temp, k, p, rp]
    # Group all UI components that can be updated.
    ui_components = [chat, dbg, txt, submit_btn, cancel_btn]

    def submit_and_manage_ui(user_msg, chat_history, *args):
        """
        Orchestrator function that manages UI state and calls the backend chat function.
        It uses a try...finally block to ensure the UI is always reset.
        """
        if not user_msg.strip():
            # If the message is empty, do nothing.
            # We yield an empty dict to avoid any state changes.
            yield {}
            return

        # 1. Update UI to "generating" state.
        #    Crucially, we do NOT update the `chat` component here, as the backend
        #    will provide the correctly formatted history in the first response chunk.
        yield {
            txt: gr.update(value="", interactive=False),
            submit_btn: gr.update(interactive=False),
            cancel_btn: gr.update(visible=True),
        }

        cancelled = False
        try:
            model_name = args[3]
            prepare_model_artifacts(model_name)

            # 2. Call the backend and stream updates
            backend_args = [user_msg, chat_history] + list(args)
            for response_chunk in chat_response(*backend_args):
                yield {
                    chat: response_chunk[0],
                    dbg: response_chunk[1],
                }
        except GeneratorExit:
            # Mark as cancelled and re-raise to prevent "generator ignored GeneratorExit"
            cancelled = True
            print("Generation cancelled by user.")
            raise
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            # If an error happens, add it to the chat history to inform the user.
            error_history = (chat_history or []) + [
                {'role': 'user', 'content': user_msg},
                {'role': 'assistant', 'content': f"**An error occurred:** {str(e)}"}
            ]
            yield {chat: error_history}
        finally:
            # Only reset UI if not cancelled (to avoid "generator ignored GeneratorExit")
            if not cancelled:
                print("Resetting UI state.")
                yield {
                    txt: gr.update(interactive=True),
                    submit_btn: gr.update(interactive=False),
                    cancel_btn: gr.update(visible=False),
                }

    def set_cancel_flag():
        """Called by the cancel button, sets the global event."""
        cancel_event.set()
        print("Cancellation signal sent.")

    def reset_ui_after_cancel():
        """Reset UI components after cancellation."""
        cancel_event.clear()  # Clear the flag for next generation
        print("UI reset after cancellation.")
        return {
            txt: gr.update(interactive=True),
            submit_btn: gr.update(interactive=False),
            cancel_btn: gr.update(visible=False),
        }

    def clear_chat_state():
        """Clear the chat and restore default UI state."""
        cancel_event.clear()
        return {
            chat: [],
            dbg: _build_debug_info(),
            txt: "",
            submit_btn: gr.update(interactive=False),
            cancel_btn: gr.update(visible=False),
        }

    def set_example_text(example_query: str | None) -> str:
        return example_query or ""

    def apply_example_selection(evt: gr.SelectData):
        if evt is None or not evt.selected:
            return gr.skip(), gr.skip()
        example_text = set_example_text(evt.value)
        return gr.update(value=example_text), toggle_submit_button(example_text)

    def toggle_submit_button(text: str | None):
        return gr.update(interactive=bool(text and text.strip()))

    def toggle_sampling_controls(temperature: float):
        sampling_enabled = float(temperature) > 0.0
        return [
            gr.update(interactive=sampling_enabled),
            gr.update(interactive=sampling_enabled),
        ]

    def restore_generation_defaults():
        return (
            gr.update(value=DEFAULT_MAX_TOKENS),
            gr.update(value=EVAL_TEMPERATURE),
            gr.update(value=UI_TOP_K_DEFAULT, interactive=False),
            gr.update(value=UI_TOP_P_DEFAULT, interactive=False),
            gr.update(value=EVAL_REPETITION_PENALTY),
        )

    # Event for submitting text via Enter key or Submit button
    submit_event = txt.submit(
        fn=submit_and_manage_ui,
        inputs=chat_inputs,
        outputs=ui_components,
    )
    submit_click_event = submit_btn.click(
        fn=submit_and_manage_ui,
        inputs=chat_inputs,
        outputs=ui_components,
    )

    # Event for the "Cancel" button.
    # It sets the cancel flag, cancels the submit event, then resets the UI.
    cancel_btn.click(
        fn=set_cancel_flag,
        cancels=[submit_event, submit_click_event]
    ).then(
        fn=reset_ui_after_cancel,
        outputs=ui_components
    )

    txt.input(fn=toggle_submit_button, inputs=txt, outputs=submit_btn)
    txt.change(fn=toggle_submit_button, inputs=txt, outputs=submit_btn)
    example_dropdown.select(
        fn=apply_example_selection,
        outputs=[txt, submit_btn],
    )
    temp.change(fn=toggle_sampling_controls, inputs=temp, outputs=[k, p])
    reset_params_btn.click(
        fn=restore_generation_defaults,
        outputs=[max_tok, temp, k, p, rp],
    )

    # Clear chat action
    clr.click(
        fn=set_cancel_flag,
        cancels=[submit_event, submit_click_event]
    ).then(
        fn=clear_chat_state,
        outputs=ui_components
    )

if __name__ == "__main__":
    launch_args = _parse_launch_args()
    demo.launch(
        theme=demo_theme,
        css=demo_css,
        share=launch_args.share,
        ssr_mode=False,
    )
