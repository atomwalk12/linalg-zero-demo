"""
Check that answer messages correctly reuse tool responses and don't duplicate thinking.

This script validates two key properties of the distilled dataset:
1. Answer Reuse: Final answers MUST reference the immediately preceding tool response,
   not be manually recalculated
2. Unique Thinking: Final answer messages should NOT duplicate the thinking process
   from the tool-calling message (indicates distillation issues)

Examples flagged:
- Answer doesn't match tool response value
- Previous message is NOT a tool response
- Identical <think> blocks in tool-calling and final answer messages
"""

import copy
import json
import re
import shutil
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DownloadMode, load_dataset, load_from_disk
from transformers import AutoTokenizer

from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers

output_buffer = StringIO()

local_dataset_path = "atomwalk12/linalgzero-distilled-local"
if not Path(local_dataset_path).exists():
    dataset = load_dataset("atomwalk12/linalgzero-distilled", "default", split="train")
    dataset.save_to_disk(local_dataset_path)


def parse_messages(messages_field: Any) -> list[dict]:
    """
    Parse messages from dataset field (handles both JSON string and dict formats).

    Args:
        messages_field: Messages field from dataset (can be string or list)

    Returns:
        List of message dictionaries
    """
    return json.loads(messages_field) if isinstance(messages_field, str) else messages_field


def get_message_content(message: dict) -> str:
    """
    Extract and clean content from a message dictionary.

    Args:
        message: Message dictionary with 'content' field

    Returns:
        Stripped message content string
    """
    return (message.get("content") or "").strip()


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Load a tokenizer by name.

    Args:
        tokenizer_name: HuggingFace model name for the tokenizer

    Returns:
        Loaded AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def count_tokens_in_message(content: str, tokenizer: AutoTokenizer) -> int:
    """Count tokens in a message content using the provided tokenizer."""
    return len(tokenizer.encode(content))


def get_max_assistant_tokens(messages: list[dict], tokenizer: AutoTokenizer) -> int:
    """
    Get the maximum token count among all assistant messages.

    Args:
        messages: List of message dictionaries
        tokenizer: Tokenizer to use for counting tokens

    Returns:
        Maximum token count found in assistant messages, or 0 if no assistant messages
    """
    assistant_token_counts = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = get_message_content(msg)
            token_count = count_tokens_in_message(content, tokenizer)
            assistant_token_counts.append(token_count)

    return max(assistant_token_counts) if assistant_token_counts else 0


def print_to_both(*args, **kwargs):
    """Print to both console and buffer."""
    print(*args, **kwargs)
    if output_buffer:
        print(*args, **kwargs, file=output_buffer)


def extract_answer_value(content: str, parser: XMLParser) -> str | None:
    """Extract the value inside <answer> tags."""
    answers = parser.extract_tag_contents(content, "answer")
    # Return the last answer if multiple exist
    return answers[-1] if answers else None


def check_tool_call_is_skipped_due_to_simple_op(
    messages: list[dict], parser: XMLParser, minimal_dataset: bool = False
) -> tuple[bool, str]:
    """
    Check if answer messages correctly reference the immediately preceding tool response.

    Returns:
        (has_issue, reason): bool indicating if there's an issue, and string with reason
    """
    answer_msg = messages[-1]
    first_to_last_msg = messages[-2]
    if first_to_last_msg["role"] != "tool":
        return False, "No tool response before final answer"

    answer = parse_string(parser.extract_last_answer(answer_msg["content"]))
    tool_response = parse_string(first_to_last_msg["content"])

    if verify_answers(tool_response, answer):
        return False, "Answer matches tool response"
    else:
        if first_to_last_msg["name"] == "determinant":
            if answer in [1, 2, 3] and not minimal_dataset:
                return False, ""
            else:
                return (
                    True,
                    "Interesting case: using determinant as the last tool call, but final answer is not a possible rank",
                )
        else:
            return (
                True,
                f"Answer does not match tool response: {answer} (final answer) != {tool_response} (final tool response -- {first_to_last_msg['name']})\n",
            )


def check_repeated_tool_calls(messages: list[dict], parser: XMLParser) -> tuple[bool, list[dict]]:
    """
    Check if the same tool function is called multiple times in a conversation.

    Returns:
        (has_repeated_calls, repeated_calls_info):
            - bool indicating if there are repeated calls
            - List of dicts with tool_name, positions, and count
    """
    tool_calls = {}  # {tool_name: [position_indices]}

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls") and len(msg["tool_calls"]) > 0:
            msg_calls = msg["tool_calls"]
            assert len(msg_calls) == 1, "Expected only one tool call per assistant message"
            tool_name = msg_calls[0]["function"]["name"]
            if tool_name not in tool_calls:
                tool_calls[tool_name] = []
            tool_calls[tool_name].append(i)

    # Find tools called more than once
    repeated = []
    for tool_name, positions in tool_calls.items():
        if len(positions) > 1:
            repeated.append({"tool_name": tool_name, "positions": positions, "count": len(positions)})

    return (len(repeated) > 0, repeated)


def check_all_think_duplicates(messages: list[dict], parser: XMLParser) -> tuple[list[dict], list[dict]]:
    """
    Find ALL duplicate <think> blocks across all message pairs (including non-adjacent).
    Keep only the first occurrence and replace subsequent duplicates with placeholder.

    Returns:
        (duplicates, modified_messages):
            - List of duplicate info dicts with positions and adjacency info
            - Modified messages list with replacements applied
    """
    duplicates = []
    modified_messages = [msg.copy() for msg in messages]  # Deep copy to avoid mutating original

    # Extract all <think> blocks from assistant messages
    thinks = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            think_content = parser.extract_tag_contents(msg["content"], "think")
            if think_content and think_content[0].strip():  # Non-empty
                thinks.append((i, think_content[0].strip()))

    # Track which think blocks have been seen (by content)
    seen_thinks = {}  # content -> first position
    positions_to_replace = []  # positions to replace with placeholder

    # Compare all pairs (including non-adjacent)
    for i in range(len(thinks)):
        for j in range(i + 1, len(thinks)):
            pos_i, content_i = thinks[i]
            pos_j, content_j = thinks[j]

            if content_i == content_j:
                duplicates.append({
                    "positions": (pos_i, pos_j),
                    "adjacent": (pos_j - pos_i == 2),  # assistant → tool → assistant
                    "content_preview": content_i[:100],
                })

                # Mark the later occurrence for replacement
                if content_i not in seen_thinks:
                    seen_thinks[content_i] = pos_i

                # Only replace if this is not the first occurrence
                if pos_j not in [seen_thinks[c] for c in seen_thinks]:
                    positions_to_replace.append(pos_j)

    # Replace duplicate think blocks with placeholder
    for pos in positions_to_replace:
        old_content = modified_messages[pos]["content"]
        # Replace the <think>...</think> content with placeholder
        new_content = re.sub(
            r"<think>.*?</think>", "<think>Finalise based on tool result</think>", old_content, flags=re.DOTALL
        )
        modified_messages[pos]["content"] = new_content

    return duplicates, modified_messages


def check_exact_sequence(
    messages: list[dict], parser: XMLParser, stepwise_ground_truths: list[dict]
) -> tuple[bool, list[dict]]:
    """
    Check if the sequence of messages matches the stepwise ground truths.

    Returns:
        (has_issue, issues_info): bool indicating if there are mismatches,
                                   and list of dicts with detailed error information
    """
    issues = []

    for step_idx, ground_truth in enumerate(stepwise_ground_truths):
        msg_idx = 2 + (step_idx * 2)  # Tool call messages are at indices 2, 4, 6, ...

        if msg_idx >= len(messages):
            issues.append({
                "step": step_idx,
                "error": "Missing tool call",
                "expected_tools": list(ground_truth.keys()),
                "actual_tool": None,
                "message_index": msg_idx,
            })
            continue

        # Check if it's actually a tool call message
        if messages[msg_idx]["tool_calls"] is None:
            issues.append({
                "step": step_idx,
                "error": "Expected tool call but message has no tool_calls",
                "expected_tools": list(ground_truth.keys()),
                "actual_tool": None,
                "message_index": msg_idx,
            })
            continue

        tool_name = messages[msg_idx]["tool_calls"][0]["function"]["name"]

        # Check if tool name matches any expected tool in ground truth
        if tool_name not in ground_truth:
            issues.append({
                "step": step_idx,
                "error": "Tool name mismatch",
                "expected_tools": list(ground_truth.keys()),
                "actual_tool": tool_name,
                "message_index": msg_idx,
            })
            continue

        # Check if tool response matches ground truth
        tool_response_idx = msg_idx + 1
        if tool_response_idx >= len(messages):
            issues.append({
                "step": step_idx,
                "error": "Missing tool response",
                "tool_name": tool_name,
                "expected_value": ground_truth[tool_name],
                "actual_value": None,
                "message_index": tool_response_idx,
            })
            continue

        tool_answer = parse_string(messages[tool_response_idx]["content"])
        expected_answer = ground_truth[tool_name]

        if not verify_answers(expected_answer, tool_answer):
            issues.append({
                "step": step_idx,
                "error": "Tool response value mismatch",
                "tool_name": tool_name,
                "expected_value": expected_answer,
                "actual_value": tool_answer,
                "message_index": tool_response_idx,
            })

    return (len(issues) > 0, issues)


def matches_exact_message_count(messages: list[dict], ground_truths: list[dict]) -> tuple[bool, str]:
    """
    Check if the number of messages matches the expected message count.
    """
    # Expected message count: 2 (user + system) + 2 * len(ground_truths) (tool calls and responses) + 1 (final answer)
    expected_message_count = len(ground_truths) * 2 + 3
    return len(messages) == expected_message_count, f"Expected {expected_message_count} messages, got {len(messages)}"


def print_messages_above_token_threshold(
    tokenizer: AutoTokenizer,
    ds: Dataset,
    token_threshold: int,
):
    """
    Print all messages with token counts above the specified threshold.
    """

    def collect_token_counts(ds, tokenizer):
        token_counts_by_role = {"user": [], "assistant": [], "tool": []}
        all_token_counts = []
        messages_with_tokens = []  # Store (token_count, sample_idx, msg_idx, role, content)

        for i in range(len(ds)):
            messages = parse_messages(ds[i]["messages"])
            for msg_idx, msg in enumerate(messages):
                content = get_message_content(msg)
                role = msg.get("role", "unknown")

                # Skip system messages
                if role == "system":
                    continue

                # Count tokens using shared utility
                token_count = count_tokens_in_message(content, tokenizer)
                all_token_counts.append(token_count)
                messages_with_tokens.append((token_count, i, msg_idx, role, content))

                if role in token_counts_by_role:
                    token_counts_by_role[role].append(token_count)

        return all_token_counts, token_counts_by_role, messages_with_tokens

    _, _, messages_with_tokens = collect_token_counts(ds, tokenizer)

    # Print messages above token threshold
    print_to_both(f"\n{'=' * 80}")
    print_to_both(f"MESSAGES WITH MORE THAN {token_threshold} TOKENS (system messages excluded)")
    print_to_both(f"{'=' * 80}\n")

    # Filter messages above threshold and sort by token count (descending)
    # Note: system messages already excluded during collection
    high_token_messages = [msg for msg in messages_with_tokens if msg[0] > token_threshold]
    sorted_messages = sorted(high_token_messages, key=lambda x: x[0], reverse=True)

    print_to_both(f"Found {len(sorted_messages)} messages above {token_threshold} tokens\n")

    for rank, (token_count, sample_idx, msg_idx, role, content) in enumerate(sorted_messages, 1):
        print_to_both(f"\n{'─' * 80}")
        print_to_both(
            f"Rank #{rank} | Tokens: {token_count} | Sample: {sample_idx} | Message: {msg_idx} | Role: {role}"
        )
        print_to_both(f"{'─' * 80}")
        print_to_both(content)

    print_to_both(f"\n{'=' * 80}")
    print_to_both(f"SUMMARY: {len(sorted_messages)} messages found with more than {token_threshold} tokens")
    print_to_both(f"{'=' * 80}")


def analyze_dataset(  # noqa: C901
    dataset_name: str,
    config: str,
    split: str,
    _load_from_disk: bool = True,
    verbose: bool = True,
    minimal_dataset: bool = False,
    assistant_token_threshold: int | None = None,
    tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct",
    push_to_hub: bool = False,
):
    """Analyze dataset for answer reuse and duplicated thinking issues."""
    print_to_both(f"Loading dataset: {dataset_name}/{config} ({split})")
    if _load_from_disk:
        ds = load_from_disk(dataset_name)
    else:
        ds = load_dataset(dataset_name, config, split=split, download_mode=DownloadMode.FORCE_REDOWNLOAD)

    print_to_both(f"Dataset size: {len(ds)}")
    print_to_both("Checking for answer reuse and duplicated thinking issues...")
    print_to_both()

    parser = XMLParser()

    # Load tokenizer if token threshold is specified
    tokenizer = None
    if assistant_token_threshold is not None:
        print_to_both(f"Loading tokenizer for token counting: {tokenizer_name}")
        tokenizer = load_tokenizer(tokenizer_name)

    # Track issues
    reuse_issues = []
    all_think_duplicates = []
    exact_sequence_issues = []
    repeated_tool_calls = []
    message_count_issues = []
    token_threshold_issues = []

    for idx in range(len(ds)):
        messages = parse_messages(ds[idx]["messages"])
        stepwise_ground_truths = json.loads(ds[idx]["stepwise_ground_truths"])

        # # Check 1: Answer must reference previous tool response
        has_reuse_issue, reuse_reason = check_tool_call_is_skipped_due_to_simple_op(
            messages, parser, minimal_dataset=minimal_dataset
        )
        if has_reuse_issue:
            reuse_issues.append((idx, reuse_reason))

        # Check 2: Repeated tool calls
        has_repeated, repeated_info = check_repeated_tool_calls(messages, parser)
        if has_repeated:
            repeated_tool_calls.append((idx, repeated_info))

        # Check 3: Check exact sequence based on stepwise ground truths
        has_exact_sequence, exact_sequence_info = check_exact_sequence(messages, parser, stepwise_ground_truths)
        if has_exact_sequence and minimal_dataset:
            exact_sequence_issues.append((idx, exact_sequence_info))

        # Check 4: Message count validation
        has_count_issue, count_reason = matches_exact_message_count(messages, stepwise_ground_truths)
        if not has_count_issue:
            message_count_issues.append((idx, count_reason))

        # Check 5: Assistant token threshold
        if assistant_token_threshold is not None and tokenizer is not None:
            max_tokens = get_max_assistant_tokens(messages, tokenizer)
            if max_tokens > assistant_token_threshold:
                token_threshold_issues.append((idx, max_tokens))

        # Check 6: ALL duplicate thinking across conversation (including non-adjacent)
        duplicates, modified_messages = check_all_think_duplicates(messages, parser)
        if duplicates:
            # Print before/after for verification
            print_to_both(f"\n{'=' * 80}")
            print_to_both(f"Example {idx}: Found {len(duplicates)} duplicate(s)")
            print_to_both(f"{'=' * 80}")

            for dup in duplicates:
                pos_i, pos_j = dup["positions"]
                print_to_both(
                    f"\nDuplicate between positions {pos_i} and {pos_j} (adjacent: {dup['adjacent']}, last: {pos_j == len(messages) - 1})"
                )
                is_answer = pos_j == len(messages) - 1

                if verbose:
                    analyse_indices("analysis_verbose", msgs=messages)
                else:
                    # Show the second occurrence (which will be replaced)
                    print_to_both(f"\n--- BEFORE (Message {pos_j}) ---")
                    print_to_both(messages[pos_j]["content"])  # First 500 chars

                    print_to_both(f"\n--- AFTER (Message {pos_j}) ---")
                    print_to_both(modified_messages[pos_j]["content"])  # First 500 chars

                all_think_duplicates.append((idx, dup, is_answer))

    # Report results
    print_to_both("=" * 80)
    print_to_both("ANALYSIS RESULTS")
    print_to_both("=" * 80)

    print_to_both(
        f"\n1. Answer-Tool Response Mismatch Issues: {len(reuse_issues)} ({len(reuse_issues) / len(ds) * 100:.2f}%)"
    )

    if reuse_issues:
        for idx, reason in reuse_issues:
            print_to_both(f"  - Example {idx}: {reason}")

    print_to_both(
        f"\n2. Repeated Tool Calls: {len(repeated_tool_calls)} ({len(repeated_tool_calls) / len(ds) * 100:.2f}%)"
    )
    if repeated_tool_calls:
        for idx, repeated_info in repeated_tool_calls:
            for tool_info in repeated_info:
                print_to_both(
                    f"  - Example {idx}: '{tool_info['tool_name']}' called {tool_info['count']} times at positions {tool_info['positions']}"
                )

    print_to_both(
        f"\n3. Exact Ground Truth Sequence Mismatches: {len(exact_sequence_issues)} ({len(exact_sequence_issues) / len(ds) * 100:.2f}%)"
    )
    if exact_sequence_issues:
        for idx, issues_list in exact_sequence_issues:
            for issue in issues_list:
                print_to_both(
                    f"  - Example {idx} step {issue['step']}: {issue['error']} "
                    f"(expected: {issue.get('expected_tools') or issue.get('expected_value')}, "
                    f"got: {issue.get('actual_tool') or issue.get('actual_value')})"
                )

    print_to_both(
        f"\n4. Message Count Mismatches: {len(message_count_issues)} ({len(message_count_issues) / len(ds) * 100:.2f}%)"
    )
    if message_count_issues:
        for idx, reason in message_count_issues:
            print_to_both(f"  - Example {idx}: {reason}")

    print_to_both(
        f"\n5. Assistant Token Threshold Exceeded: {len(token_threshold_issues)} ({len(token_threshold_issues) / len(ds) * 100:.2f}%)"
    )
    if token_threshold_issues:
        if assistant_token_threshold is not None:
            print_to_both(f"  Threshold: {assistant_token_threshold} tokens")
        for idx, max_tokens in token_threshold_issues:
            print_to_both(f"  - Example {idx}: max {max_tokens} tokens")
        if verbose:
            print_messages_above_token_threshold(tokenizer, ds, assistant_token_threshold)

    # Comprehensive duplicate check
    adjacent_dups = [d for d in all_think_duplicates if d[1]["adjacent"]]
    non_adjacent_dups = [d for d in all_think_duplicates if not d[1]["adjacent"]]

    print_to_both("\n6. ALL Think Block Duplicates (comprehensive check):")
    print_to_both(f"  Total: {len(all_think_duplicates)}")
    print_to_both(f"  Adjacent (i → tool → j): {len(adjacent_dups)}")
    print_to_both(f"  Non-adjacent: {len(non_adjacent_dups)}")

    if adjacent_dups:
        print_to_both("\n  Adjacent duplicates:")
        for idx, dup, is_answer in adjacent_dups:
            print_to_both(
                f"    Example {idx}, msgs {dup['positions']}, is answer: {is_answer}: {dup['content_preview']}..."
            )

    if non_adjacent_dups:
        print_to_both("\n  Non-adjacent duplicates:")
        for idx, dup, is_answer in non_adjacent_dups:
            print_to_both(
                f"    Example {idx}, msgs {dup['positions']}, is answer: {is_answer}: {dup['content_preview']}..."
            )

    print_to_both(f"Number of adjacent duplicates with final answer: {len([d for d in adjacent_dups if d[2]])}")

    # Combined issues
    all_issues = set(
        [idx for idx, _ in reuse_issues]
        + [idx for idx, _, _ in all_think_duplicates]
        + [idx for idx, _ in repeated_tool_calls]
        + [idx for idx, _ in exact_sequence_issues]
        + [idx for idx, _ in message_count_issues]
        + [idx for idx, _ in token_threshold_issues]
    )
    print_to_both(
        f"\n7. Total Unique Examples with Issues: {len(all_issues)} ({len(all_issues) / len(ds) * 100:.2f}%)"
    )

    print_to_both("\n" + "=" * 80)

    cleaned_ds = remove_issues(all_issues, ds)

    # Save or push cleaned dataset
    print_to_both("\n" + "=" * 80)
    print_to_both("Cleaned dataset ready.")
    print_to_both(f"Total examples removed: {len(all_issues)}")
    if push_to_hub:
        print_to_both(f"Pushing dataset to https://huggingface.co/datasets/{dataset_name}-clean")
        print_to_both(f"Dataset size: {len(cleaned_ds)}")
        cleaned_ds.push_to_hub(f"{dataset_name}-clean")
    print_to_both("=" * 80)

    return reuse_issues, all_think_duplicates, all_issues


def remove_issues(all_issues, ds):
    if len(all_issues) > 0:
        # Create a mask for filtering
        keep_mask = [i not in all_issues for i in range(len(ds))]

        # Filter dataset
        cleaned_ds = ds.select([i for i, keep in enumerate(keep_mask) if keep])

        print_to_both(f"\nInitial dataset size: {len(ds)}")
        print_to_both(f"\nCleaned dataset size: {len(cleaned_ds)}")
        print_to_both(f"Removed: {len(ds) - len(cleaned_ds)} examples")

        return cleaned_ds
    else:
        print_to_both("✅ No problematic examples found! Dataset is clean.")
        return ds


def analyse_indices(  # noqa: C901
    analysis_type: str, indices: list[int] | None = None, ds: Dataset | None = None, msgs: list[dict] | None = None
) -> Dataset | None:
    def print_msgs(messages: list[dict]):
        for i, msg in enumerate(messages):
            print_to_both(f"\n--- Message {i} ({msg['role']}) ---")

            if msg["role"] == "user":
                print_to_both(msg["content"])
            elif msg["role"] == "assistant":
                # Print content (truncate if too long)
                content = msg["content"]
                print_to_both(content)

                # Print tool calls if present
                if msg.get("tool_calls"):
                    print_to_both("\nTOOL CALLS:")
                    for tc in msg["tool_calls"]:
                        print_to_both(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
            elif msg["role"] == "tool":
                print_to_both(f"RESULT: {msg['content']}")

        print_to_both("\n")

    if indices is not None and ds is not None:
        print_to_both("=" * 80)
        print_to_both(f"Analysing indices: {indices} for analysis type: {analysis_type}")
        print_to_both("=" * 80)

        for idx in indices:
            print_to_both("=" * 80)
            print_to_both(f"EXAMPLE {idx}")
            print_to_both("=" * 80)

            messages = json.loads(ds[idx]["messages"]) if isinstance(ds[idx]["messages"], str) else ds[idx]["messages"]
            print_msgs(messages)

        return ds
    elif msgs is not None:
        print_msgs(msgs)
        return None
    else:
        raise ValueError("Either indices and ds or msgs must be provided")


def clean_dataset(dataset, settings):
    print_to_both("=" * 80)
    print_to_both("Cleaning dataset...")
    print_to_both("=" * 80)

    def remove_messages(example, idx):
        if idx in settings and "to_remove" in settings[idx]:
            if settings[idx]["initial_msg_count"] != len(json.loads(example["messages"])):
                # If msg count doesn't match, skip cleaning as it has already been cleaned
                current_count = len(json.loads(example["messages"]))
                assert settings[idx]["expected_final_msg_count"] == current_count, (
                    f"Index {idx}: Expected final message count does not match got message count"
                )
                print_to_both(
                    f"INFO: Did not modify index {idx} as it has already been cleaned. "
                    f"Config initial message count: {settings[idx]['initial_msg_count']}. "
                    f"Current message count matches expected message count: {current_count} == {settings[idx]['expected_final_msg_count']}."
                )
                return example

            to_remove = settings[idx]["to_remove"]
            msgs = copy.deepcopy(json.loads(example["messages"]))
            clean_msgs = [msg for i, msg in enumerate(msgs) if i not in to_remove]

            # Validation
            assert len(clean_msgs) == len(msgs) - len(to_remove), (
                f"Index {idx}: Number of messages to remove does not match"
            )

            remove_reason = settings[idx]["remove_reason"]
            print_to_both(f"""NOTE: Message {idx}.""")
            print_to_both(f"""Removed messages with indices: {to_remove}""")
            print_to_both(f"Remove reason: {remove_reason}")

            updated_example = example.copy()
            updated_example["messages"] = json.dumps(clean_msgs)
            return updated_example

        return example

    def update_messages(example, idx):
        if idx in settings and "to_replace" in settings[idx]:
            if settings[idx]["initial_msg_count"] != len(json.loads(example["messages"])):
                # If msg count doesn't match, skip cleaning as it has already been cleaned
                current_count = len(json.loads(example["messages"]))
                assert settings[idx]["expected_final_msg_count"] == current_count, (
                    f"Index {idx}: Expected final message count does not match got message count"
                )
                return example

            to_replace = settings[idx]["to_replace"]

            # Parse JSON once at the beginning
            messages = json.loads(example["messages"])

            # Process all indices for this example
            for index, replacement in to_replace.items():
                msg_idx = int(index)
                current_content = messages[msg_idx]["content"]

                # Parse and replace specific tags
                for tag, new_content in replacement["content"].items():
                    # Use regex to find and replace the specific tag
                    tag_pattern = rf"<{re.escape(tag)}>\s*.*?\s*</{re.escape(tag)}>"
                    current_content = re.sub(tag_pattern, new_content, current_content, flags=re.DOTALL)

                messages[msg_idx]["content"] = current_content

            # Serialize JSON once at the end
            example["messages"] = json.dumps(messages)

        return example

    # We update first, then remove redundant messages. This ensures the indices
    # remain consistent in the yaml config file.
    dataset = dataset.map(update_messages, with_indices=True)
    dataset = dataset.map(remove_messages, with_indices=True)
    return dataset


def check_integrity(dataset: Dataset, indices: list[int], cleaning_config: dict):
    for idx in indices:
        messages = json.loads(dataset[idx]["messages"])
        assert all(messages[i]["role"] != messages[i - 1]["role"] for i in range(1, len(messages))), (
            "Messages are not in the correct order"
        )

        assert len(messages) == cleaning_config[idx]["expected_final_msg_count"], (
            f"Index {idx}: Number of final messages does not match expected number"
        )


def load_cleaning_config(config_path: Path) -> dict:
    """Load cleaning configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert string keys back to tuples for remove_reason
    for operations in config.values():
        if "remove_reason" in operations:
            operations["remove_reason"] = {
                tuple(map(int, k.split(","))): v for k, v in operations["remove_reason"].items()
            }

    return config


def check_should_print_info(dataset: Dataset, cleaning_config: dict) -> tuple[list[int], list[int]]:
    """Check if should print info for the dataset."""
    all_indices = list(cleaning_config)
    dirty_indices = []
    for idx in all_indices:
        # If the current message count is equal to the config initial message count,
        # this means that it has not yet been processed
        if cleaning_config[idx]["initial_msg_count"] == len(json.loads(dataset[idx]["messages"])):
            dirty_indices.append(idx)
    print_to_both(f"Processing {len(dirty_indices)} indices for cleaning...")
    print_to_both(f"Will print info only for indices: {dirty_indices}")
    print_to_both(f"All indices: {all_indices}")
    return all_indices, dirty_indices


def process_dataset_cleaning(dataset, config_path: Path):
    """Apply cleaning operations from configuration file."""
    cleaning_config = load_cleaning_config(config_path)

    all_indices, dirty_indices = check_should_print_info(dataset, cleaning_config)
    dataset = analyse_indices("original", indices=all_indices, ds=dataset)
    cleaned_dataset = clean_dataset(dataset, cleaning_config)

    # Notice that here we print only the dirty indices that have been cleaned
    cleaned_dataset = analyse_indices("cleaned", indices=dirty_indices, ds=cleaned_dataset)

    # Check integrity against ALL indices
    check_integrity(cleaned_dataset, indices=all_indices, cleaning_config=cleaning_config)

    return cleaned_dataset, cleaning_config


def save_dataset_to_disk(dataset: Dataset, path: str):
    """Save the dataset to a directory."""
    # Save to temporary location
    temp_path = f"{path}_temp"
    dataset.save_to_disk(temp_path)

    # Remove old dataset and rename temp
    shutil.rmtree(path)
    shutil.move(temp_path, path)


def inspect(indices: list[int], ds: Dataset):
    analyse_indices("original", indices=indices, ds=ds)


if __name__ == "__main__":
    # TODO: check for duplicate function calls and fix them
    inspect_indices = False
    analyse = False
    local_dataset = False
    commit = False

    if inspect_indices:
        dataset_path = local_dataset_path if local_dataset else "atomwalk12/linalgzero-distilled"
        dataset = load_dataset(dataset_path, "default", split="train")
        inspect(indices=[177, 285], ds=dataset)
        sys.exit()

    # Set to None to disable token threshold check, or set to a number (e.g., 1000) to enable
    assistant_token_threshold = None
    dataset_path = local_dataset_path if local_dataset else "atomwalk12/linalgzero-distilled"
    if analyse:
        output_file = "analyse_indices.txt"
        dataset = load_from_disk(dataset_path)
        cleaned_dataset, config = process_dataset_cleaning(dataset, Path("linalg_zero/config/cleaning_config.yaml"))
        assert cleaned_dataset is not None, "Cleaned dataset is None"

        if commit:
            save_dataset_to_disk(cleaned_dataset, path=local_dataset_path)
        else:
            print_to_both("Not committing dataset to disk.")
    else:
        output_file = "analyse_dataset.txt"
        reuse_issues, all_think_duplicates, all_issues = analyze_dataset(
            dataset_path,
            "default",
            "train",
            _load_from_disk=local_dataset,
            verbose=True,
            minimal_dataset=True,
            assistant_token_threshold=800,
            push_to_hub=False,
        )

    print_to_both(f"Wrote data to file: {output_file}")

    # Write buffer to file
    with open(output_file, "w") as f:
        f.write(output_buffer.getvalue())
