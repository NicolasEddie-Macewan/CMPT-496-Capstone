"""
@file BR_agent.py
@brief Defines the BRAgent, a LangGraph-based agent for validating and citing business rules extracted from source code.
@details Implements a condenser-retriever-validator-writer workflow that takes business rules from G1/G2,
condenses duplicates, validates each rule against vector-retrieved code context, and writes the results to JSON.
"""

from agent.states.BR_agent_state import BRGraphState
from agent.structured_output.BR_output import (
    CondensedRule, ValidatedRule, DiscardedRule,
    CondenserOutput, ValidatorOutput, Explanation
)
from agent.structured_output.file_summary_output import BusinessRule
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import sys
import json
import asyncio
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
from collections import deque, defaultdict

MAX_CONCURRENCY = 10


class BRAgent:
    """
    @brief LangGraph-based agent for validating business rules against codebase evidence.

    @details
    The BRAgent constructs and executes a LangGraph workflow that:
    - Condenses duplicate/similar business rules from G1/G2 output.
    - Iteratively retrieves relevant code snippets and file summaries per rule.
    - Validates each rule against retrieved context, producing evidence citations or discarding unsupported rules.
    - Writes validated and discarded rules to JSON output files.
    """

    def __init__(self, model=None):
        """
        @brief Initializes the BRAgent with a specified language model.
        @param model An optional language model to use. If not provided, defaults to gemini-3-flash-preview.
        """
        if model is None:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                api_key=api_key)
        else:
            self.llm = model
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        """
        @brief Constructs the StateGraph that defines the BRAgent workflow.
        @return A compiled StateGraph object.

        @details
        Graph structure:
            condenser → retriever → validator → conditional → writer → END

        Conditional routing from validator:
            - If current_rule is set (need_more_context, or next rule popped) → retriever
            - If current_rule is None (all rules processed) → writer
        """
        builder = StateGraph(BRGraphState)

        # Set nodes
        builder.add_node("condenser", self.condenser_node)
        builder.add_node("retriever", self.retriever_node)
        builder.add_node("validator", self.validator_node)
        builder.add_node("writer", self.writer_node)

        # Set edges
        builder.set_entry_point("condenser")
        builder.add_conditional_edges(
            "condenser",
            lambda state: "retriever" if state.get("current_rule") else "writer"
        )
        builder.add_edge("retriever", "validator")
        builder.add_conditional_edges(
            "validator",
            lambda state: "retriever" if state.get("current_rule") else "writer"
        )
        builder.add_edge("writer", END)

        return builder.compile()

    def run(self, input_rules: dict[str, list[BusinessRule]], codebase_name: str):
        """
        @brief Executes the BRAgent workflow.
        @param input_rules Dictionary of business rules from G1/G2. Keys are file or directory paths,
               values are lists of BusinessRule objects.
        @param codebase_name Name of the target codebase, used to look up the correct ChromaDB collections.
        @return Final state of the graph after execution.
        """
        script_dir = Path(__file__).parent.resolve()
        db_dir = (script_dir.parent / "vectorStores").resolve()

        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=str(db_dir))

        code_collection = client.get_collection(
            name=f"{codebase_name}_code_db",
            embedding_function=embedding_fn
        )

        summary_collection = client.get_collection(
            name=f"{codebase_name}_summary_db",
            embedding_function=embedding_fn
        )

        initial_state = {
            "input_rules": input_rules,
            "rules_queue": deque(),
            "current_rule": None,
            "validated_rules": [],
            "discarded_rules": [],
            "code_context": [],
            "summary_context": [],
            "codebase_k": 15,
            "file_summary_k": 5,
            "code_collection": code_collection,
            "summary_collection": summary_collection,
            "codebase_name": codebase_name,
            "output_directory": "./agent/BR_agent_output",
        }

        self._loop = asyncio.new_event_loop()
        try:
            return self.graph.invoke(initial_state)
        finally:
            self._loop.close()
            self._loop = None

    def condenser_node(self, state: BRGraphState) -> BRGraphState:
        """
        @brief Condenses duplicate or near-duplicate business rules from G1/G2 output.

        @details
        Groups input rules by directory (derived from file path keys, made relative to codebase root).
        For each directory group with more than one rule, prompts the LLM (with CondenserOutput
        structured output) to identify and merge duplicates or near-duplicates.
        Single-rule groups are passed through without an LLM call.
        LLM calls are batched concurrently across directory groups for speed.
        IDs are assigned sequentially after all results are collected, in sorted directory order.
        Each CondensedRule carries all source file paths from its directory group.
        Runs exactly once at the start of the graph.

        @param state Current workflow state containing input_rules and codebase_name.
        @return Updated state with rules_queue and current_rule populated.
        """
        input_rules = state["input_rules"]
        codebase_name = state["codebase_name"]

        # Handle empty input
        if not input_rules:
            return {
                "rules_queue": deque(),
                "current_rule": None,
            }

        # Group rules by relative directory
        # Keys in input_rules are file paths; derive directory relative to codebase root
        dir_groups: dict[str, dict] = defaultdict(lambda: {"rules": [], "file_paths": set()})

        for file_path, rules in input_rules.items():
            abs_dir = os.path.dirname(file_path)

            # Attempt to make the directory relative to the codebase root
            # The codebase root name appears in the path; find it and compute relative
            try:
                path_obj = Path(abs_dir)
                # Walk up to find the codebase root directory
                parts = path_obj.parts
                codebase_idx = None
                for i, part in enumerate(parts):
                    if part == codebase_name:
                        codebase_idx = i
                        break

                if codebase_idx is not None:
                    rel_dir = Path(*parts[codebase_idx:]).as_posix()
                else:
                    rel_dir = Path(abs_dir).as_posix()
            except (ValueError, TypeError):
                rel_dir = abs_dir

            dir_groups[rel_dir]["file_paths"].add(file_path)
            dir_groups[rel_dir]["rules"].extend(rules)

        # Filter out groups with no rules
        dir_groups = {k: v for k, v in dir_groups.items() if v["rules"]}

        # Separate single-rule groups (no LLM call needed) from multi-rule groups
        single_rule_groups = {}
        multi_rule_groups = {}
        for dir_name, group in dir_groups.items():
            if len(group["rules"]) == 1:
                single_rule_groups[dir_name] = group
            else:
                multi_rule_groups[dir_name] = group

        # Batch async LLM calls for multi-rule groups
        structured_llm = self.llm.with_structured_output(CondenserOutput)
        sorted_multi_dirs = sorted(multi_rule_groups.keys())

        async def run_batch():
            sem = asyncio.Semaphore(MAX_CONCURRENCY)
            async def guarded(directory: str):
                async with sem:
                    return await _condense_group(
                        structured_llm,
                        directory,
                        multi_rule_groups[directory]["rules"]
                    )
            return await asyncio.gather(
                *(guarded(d) for d in sorted_multi_dirs)
            )

        if sorted_multi_dirs:
            results = self._loop.run_until_complete(run_batch())
        else:
            results = []

        # Build condensed rule results keyed by directory (preserving sorted order)
        # results[i] corresponds to sorted_multi_dirs[i]
        condensed_by_dir: dict[str, list[str]] = {}
        for dir_name, (returned_dir, condensed_strings, err) in zip(sorted_multi_dirs, results):
            if err is not None:
                # On error, pass through original rules uncondensed
                print(f"Condensation error for {dir_name}: {err}")
                condensed_strings = [r.rule for r in multi_rule_groups[dir_name]["rules"]]
            condensed_by_dir[dir_name] = condensed_strings

        # Assign sequential IDs across all groups in sorted directory order
        all_condensed: list[CondensedRule] = []
        rule_id = 1

        for dir_name in sorted(dir_groups.keys()):
            group = dir_groups[dir_name]
            file_paths = sorted(group["file_paths"])

            if dir_name in single_rule_groups:
                # Single rule — pass through without LLM
                rule_text = group["rules"][0].rule
                all_condensed.append(CondensedRule(
                    id=rule_id,
                    rule=rule_text,
                    source_directory=dir_name,
                    source_file_paths=file_paths,
                ))
                rule_id += 1
            else:
                # Multi-rule group — use LLM-condensed results
                for rule_text in condensed_by_dir[dir_name]:
                    all_condensed.append(CondensedRule(
                        id=rule_id,
                        rule=rule_text,
                        source_directory=dir_name,
                        source_file_paths=file_paths,
                    ))
                    rule_id += 1

        # Populate queue and set first rule
        if all_condensed:
            rules_queue = deque(all_condensed[1:])
            current_rule = all_condensed[0]
        else:
            rules_queue = deque()
            current_rule = None

        print(f"Condensed {sum(len(g['rules']) for g in dir_groups.values())} input rules "
              f"into {len(all_condensed)} condensed rules across {len(dir_groups)} directory groups.")

        return {
            "rules_queue": rules_queue,
            "current_rule": current_rule,
        }

    def retriever_node(self, state: BRGraphState) -> BRGraphState:
        """
        @brief Retrieves relevant code snippets and file summaries for the current business rule.

        @details
        Queries two ChromaDB vector collections (code and summary) using the current rule's
        text and source directory as the query. Results are prioritized in three tiers:
            1. Results from the rule's source files (highest priority)
            2. Results from the rule's source directory
            3. All other results (fallback)

        Retrieved context accumulates across retrieval iterations for the same rule.
        Duplicates are filtered out using a set of previously retrieved items.
        Retrieval depth is controlled by codebase_k and file_summary_k, which may be
        increased by the validator on "need_more_context" decisions.

        @param state Current workflow state containing current_rule and retrieval parameters.
        @return Updated state with code_context and summary_context populated/extended.
        @raises ValueError If current_rule is not set.
        """
        current_rule = state.get("current_rule")
        if not current_rule:
            raise ValueError("No current rule to retrieve context for.")

        code_collection = state["code_collection"]
        summary_collection = state["summary_collection"]

        source_directory = current_rule.source_directory
        source_file_paths = current_rule.source_file_paths

        # Build query text: rule text is the primary signal, directory is secondary context
        query_text = f"{current_rule.rule} {source_directory}"

        code_k = state["codebase_k"]
        summary_k = state["file_summary_k"]

        # Query both vector stores
        code_results = code_collection.query(
            query_texts=[query_text],
            n_results=code_k
        )

        summary_results = summary_collection.query(
            query_texts=[query_text],
            n_results=summary_k
        )

        # Track existing context for deduplication
        existing_code_context = set(state.get("code_context", []))
        existing_summary_context = set(state.get("summary_context", []))

        # Process code results with three-tier prioritization
        code_docs = code_results.get("documents", [[]])[0]
        code_metas = code_results.get("metadatas", [[]])[0]

        source_file_code = []
        directory_code = []
        fallback_code = []

        for doc, meta in zip(code_docs, code_metas):
            file_path = self._normalize_path(meta.get("file", ""))
            formatted = self._format_code_result(doc, meta, source_directory)

            if self._is_from_source_file(file_path, source_file_paths):
                source_file_code.append(formatted)
            elif self._is_in_directory(file_path, source_directory):
                directory_code.append(formatted)
            else:
                fallback_code.append(formatted)

        # Process summary results with three-tier prioritization
        summary_docs = summary_results.get("documents", [[]])[0]
        summary_metas = summary_results.get("metadatas", [[]])[0]

        source_file_summary = []
        directory_summary = []
        fallback_summary = []

        for doc, meta in zip(summary_docs, summary_metas):
            summary_path = self._normalize_path(meta.get("path", ""))
            formatted = self._format_summary_result(doc, meta, source_directory)

            if self._is_from_source_file(summary_path, source_file_paths):
                source_file_summary.append(formatted)
            elif self._is_in_directory(summary_path, source_directory):
                directory_summary.append(formatted)
            else:
                fallback_summary.append(formatted)

        # Append new results to context (prioritized order, no duplicates)
        updated_code_context = list(state.get("code_context", []))
        updated_summary_context = list(state.get("summary_context", []))

        for item in source_file_code + directory_code + fallback_code:
            if item not in existing_code_context:
                updated_code_context.append(item)

        for item in source_file_summary + directory_summary + fallback_summary:
            if item not in existing_summary_context:
                updated_summary_context.append(item)

        return {
            "code_context": updated_code_context,
            "summary_context": updated_summary_context,
        }

    def validator_node(self, state: BRGraphState) -> BRGraphState:
        """
        @brief Assesses context sufficiency and validates the current business rule in a single LLM call.

        @details
        Implementation considerations:
        - This node combines the responsibilities of context analysis and rule validation,
          reducing the number of LLM calls per rule from two to one.
        - Uses ValidatorOutput structured output with a Literal["need_more_context", "valid", "discard"]
          decision field to force the LLM into one of three distinct outcomes.
        - Prompt engineering is critical: the model must understand that "need_more_context" is a
          legitimate and distinct outcome from "discard". The prompt should clearly differentiate:
            * "need_more_context": the retrieved context is insufficient to make a judgement, and
              more retrieval may help.
            * "valid": the rule is supported by evidence in the retrieved context.
            * "discard": the rule is not supported and additional retrieval is unlikely to help.

        On "need_more_context":
        - Increase codebase_k and file_summary_k (bounded by maximum caps).
          Suggested maximums: codebase_k max = 30, file_summary_k max = 10.
        - If k values have reached the cap, force a decision (valid or discard) — do not allow
          infinite retrieval loops.
        - Keep current_rule unchanged so the conditional edge routes back to the retriever.

        On "valid":
        - Create a ValidatedRule from the current CondensedRule and the LLM's Explanation.
        - Return it as a single-element list (the Annotated[list, add] reducer will append it).
        - Reset retrieval state: clear code_context, summary_context, reset codebase_k and
          file_summary_k to defaults.
        - Pop the next rule from rules_queue as current_rule, or set current_rule to None if
          the queue is empty (triggering the writer via the conditional edge).

        On "discard":
        - Create a DiscardedRule from the current CondensedRule and the LLM's discard_reason.
        - Return it as a single-element list (the Annotated[list, add] reducer will append it).
        - Reset retrieval state and advance to next rule, same as "valid".

        @param state Current workflow state containing current_rule, contexts, and retrieval params.
        @return Updated state reflecting the decision outcome.
        """
        pass

    def writer_node(self, state: BRGraphState) -> BRGraphState:
        """
        @brief Writes validated and discarded rules to JSON output files.

        @details
        Implementation considerations:
        - Serializes state["validated_rules"] to a JSON file containing all rules that
          passed validation along with their Explanation evidence.
        - Serializes state["discarded_rules"] to a separate JSON file for transparency
          and debugging, containing all rejected rules with reasons.
        - Output directory structure: {output_directory}/{codebase_name}/
          with files like validated_rules.json and discarded_rules.json.
        - Creates output directories if they don't exist.
        - Runs exactly once at the end of the graph.

        @param state Current workflow state containing validated_rules and discarded_rules.
        @return Empty dict (terminal node).
        """
        pass

    # Helper methods

    def _normalize_path(self, path_value: str) -> str:
        """
        @brief Normalizes a filesystem path to POSIX format.
        @param path_value The path to normalize.
        @return The normalized POSIX-style path, or an empty string if the input is empty.
        """
        if not path_value:
            return ""
        return Path(path_value).as_posix()

    def _is_in_directory(self, candidate_path: str, target_rel_dir: str) -> bool:
        """
        @brief Checks whether a file path belongs to a specified directory.
        @param candidate_path The file path being evaluated.
        @param target_rel_dir The relative directory to check membership against.
        @return True if the path belongs to the directory or one of its subdirectories.
        """
        candidate_path = self._normalize_path(candidate_path)

        if target_rel_dir == ".":
            return True

        parent_dir = Path(candidate_path).parent.as_posix()
        return parent_dir == target_rel_dir or parent_dir.startswith(target_rel_dir + "/")

    def _is_from_source_file(self, candidate_path: str, source_file_paths: list[str]) -> bool:
        """
        @brief Checks whether a retrieved result's file path matches any of the rule's source files.
        @param candidate_path The file path from the retrieved result's metadata.
        @param source_file_paths List of source file paths from the CondensedRule.
        @return True if the candidate matches any source file path.
        """
        candidate_normalized = self._normalize_path(candidate_path)
        for source_path in source_file_paths:
            source_normalized = self._normalize_path(source_path)
            # Check both exact match and suffix match (handles absolute vs relative paths)
            if candidate_normalized == source_normalized:
                return True
            if candidate_normalized.endswith(source_normalized) or source_normalized.endswith(candidate_normalized):
                return True
        return False

    def _format_code_result(self, doc: str, meta: dict, source_directory: str) -> str:
        """
        @brief Formats a retrieved code chunk and its metadata for inclusion in context.
        @param doc The retrieved code snippet content.
        @param meta Metadata associated with the snippet.
        @param source_directory The source directory of the current rule.
        @return A formatted string representing the code context entry.
        """
        file_path = self._normalize_path(str(meta.get("file", "unknown")))

        return (
            f"[CODE CHUNK]\n"
            f"Directory: {source_directory}\n"
            f"File: {file_path}\n"
            f"Container: {meta.get('container', 'unknown')}\n"
            f"Name: {meta.get('name', 'unknown')}\n"
            f"Type: {meta.get('type', 'unknown')}\n"
            f"Namespace: {meta.get('namespace', 'unknown')}\n"
            f"Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}\n"
            f"Content:\n{doc}"
        )

    def _format_summary_result(self, doc: str, meta: dict, source_directory: str) -> str:
        """
        @brief Formats a retrieved summary entry and its metadata for context.
        @param doc The retrieved summary text.
        @param meta Metadata associated with the summary node.
        @param source_directory The source directory of the current rule.
        @return A formatted string representing the summary context entry.
        """
        summary_path = self._normalize_path(str(meta.get("path", "unknown")))

        return (
            f"[SUMMARY NODE]\n"
            f"Directory: {source_directory}\n"
            f"Path: {summary_path}\n"
            f"Node Type: {meta.get('type', 'unknown')}\n"
            f"Name: {meta.get('name', 'unknown')}\n"
            f"Parent: {meta.get('parent', 'N/A')}\n"
            f"Content:\n{doc}"
        )


async def _condense_group(structured_llm, directory: str, rules: list) -> tuple[str, list[str], Exception | None]:
    """
    @brief Async helper that prompts the LLM to condense a single directory group of business rules.
    @param structured_llm LLM configured with CondenserOutput structured output.
    @param directory The relative directory name for this group.
    @param rules List of BusinessRule objects to condense.
    @return Tuple of (directory, list of condensed rule strings, error or None).
    """
    try:
        rule_list = "\n".join(f"{i+1}. {r.rule}" for i, r in enumerate(rules))

        system_message = """You are a Senior Software Architect. Your task is to condense a list of business rules by merging rules that are semantically similar or redundant."""

        prompt = f"""Directory: {directory}

        Business rules to condense:
        {rule_list}

        MERGING GUIDELINES:
        - Merge rules that express the same constraint or policy in different words.
        - Merge rules that are specific instances of a more general pattern. When several rules each describe a similar aspect of the codebase's behaviour but for different cases, combine them into one general rule that captures the shared intent.
        - When merging, produce a single clear statement that preserves the meaning of all merged rules. Do not lose important specifics unless they are redundant.
        - Do NOT merge rules that govern different aspects of the system, even if they sound superficially similar.
        - Do NOT invent new rules that are not supported by the originals.
        - Do NOT discard a rule unless it is fully covered by another rule in the list.
        - Rules that are already unique and distinct should be kept as-is.

        POSITIVE EXAMPLE — rules that SHOULD be merged:
        Input:
        1. A number can be converted into its written French representation.
        2. A number can be converted into its written Arabic representation.
        3. A number can be converted into its written Spanish representation.
        Output:
        1. A number can be converted into written representations in various languages.

        NEGATIVE EXAMPLE — rules that should NOT be merged:
        Input:
        1. Order total must be non-negative.
        2. An order must contain at least one item to be processed.
        These both relate to order validation, but they enforce different constraints (value range vs. item count). They must remain separate.

        Return the condensed list of business rules."""

        messages = [("system", system_message), ("user", prompt)]
        output = await structured_llm.ainvoke(messages)
        return directory, output.condensed_rules, None
    except Exception as e:
        return directory, [], e


if __name__ == "__main__":
    """
    @brief Script entry point for running BRAgent.
    @details Loads business rules from a JSON file and runs the validation pipeline.
    """
    if len(sys.argv) != 3:
        print("Usage: python -m agent.BR_agent <codebase_name> <rules_json_path>")
        sys.exit(1)

    codebase_name = sys.argv[1]
    rules_path = sys.argv[2]

    with open(rules_path, "r", encoding="utf-8") as f:
        raw_rules = json.load(f)

    # Convert raw JSON dicts back to BusinessRule objects
    input_rules = {
        path: [BusinessRule(**rule) for rule in rules]
        for path, rules in raw_rules.items()
    }

    agent = BRAgent()
    agent.run(input_rules, codebase_name)
    print("BRAgent has completed its task!")
