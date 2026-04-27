#!/usr/bin/env python3
"""
RL Reward Scorer - Retroactively compute RL reward scores from existing test results.

Compares two result files side by side and highlights cases where selective
consistency hurts most (high reward delta).

Usage:
    PYTHONPATH=/zhaoshu/subgraph:/zhaoshu/subgraph/src python3 scripts/prompt_tuning/tools/rl_reward_scorer.py \
      --baseline reports/.../results.json \
      --selective reports/.../results.json \
      --top-delta 20

If plug_v10 import fails, computes F1/compliance/behavior/lazy rewards directly
from trajectory data without depending on the RL environment.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List

# ---------------------------------------------------------------------------
# Attempt to import plug_v10 reward functions
# ---------------------------------------------------------------------------
PLUG_V10_AVAILABLE = False
try:
    # Add repo root to path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    src_path = os.path.join(repo_root, 'src')
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

    from plug_v10 import (
        calculate_f1,
        normalize,
        parse_boxed_answers,
    )
    PLUG_V10_AVAILABLE = True
    print("[INFO] plug_v10 imported successfully - full reward computation available.",
          file=sys.stderr)
except Exception as e:
    print(f"[WARN] plug_v10 import failed ({e}). "
          f"Computing F1/compliance/behavior/lazy from trajectory data directly.",
          file=sys.stderr)

    # Provide fallback implementations
    def normalize(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch.isalnum() or ch in " .-_")
        return " ".join(s.split())

    def calculate_f1(pred: List[str], gt: List[str]) -> float:
        if not pred or not gt:
            return 0.0
        p_set = {normalize(x) for x in pred}
        g_set = {normalize(x) for x in gt}
        if not p_set or not g_set:
            return 0.0
        common = p_set & g_set
        if not common:
            return 0.0
        precision = len(common) / len(p_set)
        recall = len(common) / len(g_set)
        return 2 * precision * recall / (precision + recall)

    def parse_boxed_answers(text: str) -> List[str]:
        text_without_reason = re.sub(
            r'<reason>.*?</reason>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return [a.strip() for a in
                re.findall(r"\\boxed\{([^}]*)\}", text_without_reason)
                if a.strip()]


# ===========================================================================
# 1. Data loading
# ===========================================================================

def load_results(path: str) -> List[Dict]:
    """Load results.json and return list of case dicts."""
    with open(path) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} cases from {path}", file=sys.stderr)
    return data


# ===========================================================================
# 2. Reconstruct rollout_info from trajectory data
# ===========================================================================

def extract_entities_from_response_text(text: str) -> List[str]:
    """Extract entity lists from action backend response_text.

    Looks for:
    - 'CVT-Expanded Entities (N):\\n  [entity1, entity2, ...]'
    - 'Leaf Entities (N):\\n  [entity1, entity2, ...]'
    - 'Target Entities:\\n  [entity1, entity2, ...]'
    """
    entities = []
    # Pattern: "CVT-Expanded Entities (N):" or "Leaf Entities (N):" followed by list
    for pattern in [
        r'CVT-Expanded Entities\s*\(\d+\):\s*\n\s*\[([^\]]+)\]',
        r'Leaf Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]',
        r'Target Entities\s*:?\s*\n\s*\[([^\]]+)\]',
    ]:
        match = re.search(pattern, text)
        if match:
            list_str = match.group(1)
            for item in re.findall(r"'([^']*)'|\"([^\"]*)\"", list_str):
                val = item[0] or item[1]
                if val and val.strip():
                    entities.append(val.strip())
            if not entities:
                # Fallback: comma-separated
                for part in list_str.split(','):
                    cleaned = part.strip().strip("'\"")
                    if cleaned:
                        entities.append(cleaned)
            break
    return entities


def extract_plan_relations_from_queries(queries: List[Dict]) -> List[str]:
    """Extract planned relations from plan queries."""
    relations = []
    for q in queries:
        if q.get('tool_name') == 'plan':
            args = q.get('arguments', {})
            for field in ['related', 'maybe_related']:
                vals = args.get(field, [])
                if isinstance(vals, list):
                    relations.extend(v for v in vals if v)
            # Also constraint_relations
            cr = args.get('constraint_relations', [])
            if isinstance(cr, list):
                relations.extend(v for v in cr if v)
    return relations


def extract_action_signature(query: Dict) -> str:
    """Generate a signature for an action query."""
    args = query.get('arguments', {})
    anchor = args.get('anchor', '')
    if isinstance(anchor, list):
        anchor = '|'.join(sorted(anchor))
    path = args.get('path', [])
    path_sig = ''
    if isinstance(path, list):
        path_sig = '|'.join(
            f"{p.get('relation', '')}:{p.get('direction', 'out')}"
            for p in path if isinstance(p, dict)
        )
    return f"{anchor}|{path_sig}"


def reconstruct_rollout_info(case: Dict) -> Dict:
    """Reconstruct rollout_info dict from a single case's trajectory.

    Returns dict with format:
    {
        "turn_0": {
            "queries": [...],
            "tool_results": [{"status": "SUCCESS/ERROR", "tool_name": ..., ...}],
            "parsed": {...},
            "behavior_violations": [...],
            "completion": "...",
            ...
        },
        "turn_1": {...},
        ...
    }
    """
    trajectory = case.get('trajectory', [])
    rollout_info = {}

    for turn_data in trajectory:
        turn_idx = turn_data.get('turn', 0) - 1  # 0-indexed
        if turn_idx < 0:
            turn_idx = 0
        key = f"turn_{turn_idx}"

        raw_response = turn_data.get('raw_response', '')
        parsed_output = turn_data.get('parsed_output', {})
        executed_queries = turn_data.get('executed_queries', [])
        backend_results = turn_data.get('backend_results', [])
        frontend_errors = turn_data.get('frontend_errors', [])
        state_snapshot = turn_data.get('state_snapshot', {})

        # Map backend results to tool_results format expected by reward functions
        tool_results = []
        for br in backend_results:
            status = br.get('status', '')
            if status == 'SUCCESS':
                mapped_status = 'KG_SUCCESS'
            elif status == 'ERROR':
                mapped_status = 'KG_ERROR'
            else:
                mapped_status = status

            tool_name = br.get('tool_name', '')
            response_text = br.get('response_text', '')

            # Extract entities from response_text
            found_entities = extract_entities_from_response_text(response_text)

            result = {
                'status': mapped_status,
                'tool_name': tool_name,
                'response_text': response_text,
                'found_end_entities': found_entities,
            }
            tool_results.append(result)

        # Add frontend errors as tool results too
        for fe in frontend_errors:
            if isinstance(fe, dict):
                tool_results.append({
                    'status': 'PARSE_ERROR',
                    'tool_name': 'frontend',
                    'response_text': fe.get('message', str(fe)),
                    'found_end_entities': [],
                })

        # Extract plan relations for plan quality reward
        plan_relations = extract_plan_relations_from_queries(executed_queries)
        action_hints = []
        # Extract action hints from plan backend results
        for j, br in enumerate(backend_results):
            if br.get('tool_name') == 'plan':
                rt = br.get('response_text', '')
                # Parse action space from plan response
                for action_match in re.finditer(
                        r'Action:\s*action\(anchor="([^"]+)",\s*path=\[([^\]]+)\]\)',
                        rt):
                    anchor = action_match.group(1)
                    path_str = action_match.group(2)
                    steps = []
                    for step_match in re.finditer(
                            r'\{"relation":\s*"([^"]+)",\s*"direction":\s*"([^"]+)"\}',
                            path_str):
                        steps.append({
                            'relation': step_match.group(1),
                            'direction': step_match.group(2),
                        })
                    if steps:
                        sig = f"{anchor}|{'|'.join(s['relation']+':'+s['direction'] for s in steps)}"
                        action_hints.append(sig)

        # Compute action results for plan/action rewards
        action_results = {}
        gt = case.get('ground_truth', [])
        gt_norm = {normalize(g) for g in gt} if gt else set()
        for j, eq in enumerate(executed_queries):
            if eq.get('tool_name') == 'action':
                sig = extract_action_signature(eq)
                # Get entities from corresponding backend result
                entities = []
                if j < len(tool_results):
                    entities = tool_results[j].get('found_end_entities', [])
                entities_norm = {normalize(e) for e in entities}
                has_gt = bool(gt_norm & entities_norm) if gt_norm else False
                core_rels = []
                path = eq.get('arguments', {}).get('path', [])
                if isinstance(path, list):
                    core_rels = [p.get('relation', '') for p in path
                                 if isinstance(p, dict) and p.get('relation')]
                action_results[sig] = {
                    'entities': entities,
                    'has_gt': has_gt,
                    'core_relations': core_rels,
                }

        # Detect behavior violations
        behavior_violations = []
        turn_count_so_far = turn_idx + 1
        boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', raw_response)
        if turn_count_so_far <= 1 and boxed_matches:
            behavior_violations.append("l1_early_guess")
        if not executed_queries:
            behavior_violations.append("no_tool_calls")
        if not boxed_matches:
            behavior_violations.append("no_answer")
        elif boxed_matches and not boxed_matches[-1].strip():
            behavior_violations.append("empty_answer")

        # Check for insufficient answer
        pred = parse_boxed_answers(raw_response)
        if pred and any('insufficient' in str(p).lower() for p in pred):
            behavior_violations.append("answered_insufficient")

        turn_info = {
            'queries': executed_queries,
            'tool_results': tool_results,
            'parsed': parsed_output,
            'completion': raw_response,
            'behavior_violations': behavior_violations,
            'plan_relations': plan_relations,
            'generator_relations': plan_relations,
            'constraint_relations': [],
            'action_hints': action_hints,
            'action_results': action_results,
            'feedback': turn_data.get('feedback', ''),
            'kg_state_snapshot': {
                'plan': bool(plan_relations),
                'any_match_executed': any(
                    q.get('tool_name') == 'action' for q in executed_queries),
                'retrieved_candidates': state_snapshot.get(
                    'retrieved_candidates', []),
            },
            'verified_constraints_count': 0,
            'invalid_rejection_count': 0,
        }

        rollout_info[key] = turn_info

    return rollout_info


# ===========================================================================
# 3. Reward computation (standalone, no plug_v10 dependency)
# ===========================================================================

def compute_f1_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute F1 reward from predicted vs ground_truth."""
    pred = case.get('predicted', [])
    gt = case.get('ground_truth', [])
    f1 = calculate_f1(pred, gt) if pred and gt else 0.0
    return {
        'score': f1,
        'predicted': pred,
        'ground_truth': gt,
    }


def compute_compliance_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute tag structure compliance reward.

    Checks each turn's raw_response for required tags and penalizes issues.
    Mimics ComplianceReward from plug_v10.
    """
    turn_scores = []
    severe_error = False
    details = []

    for key in sorted(rollout_info.keys()):
        if not key.startswith('turn_'):
            continue
        info = rollout_info[key]
        text = info.get('completion', '')
        text_lower = text.lower()
        queries = info.get('queries', [])
        tool_results = info.get('tool_results', [])

        score = 0.0

        # Positive: has <reason> with meaningful content
        reason_match = re.search(r'<reason>(.*?)</reason>', text,
                                 re.DOTALL | re.IGNORECASE)
        if reason_match:
            reason_content = reason_match.group(1).strip()
            if len(reason_content) >= 30:
                # Repetition check (simplified)
                lines = [l.strip() for l in reason_content.splitlines()
                         if len(l.strip()) >= 15]
                if lines:
                    line_counts = Counter(lines)
                    is_repetitive = (any(c >= 3 for c in line_counts.values()) or
                                     (len(lines) / max(1, len(line_counts)) > 2.0))
                    if not is_repetitive:
                        score += 0.5
                    else:
                        score -= 0.5
                        details.append(f"{key}: repetitive_reason")
                else:
                    score += 0.5
            details.append(f"{key}: +has_reason")
        else:
            details.append(f"{key}: -no_reason")

        # Check for <brief_summary>
        if re.search(r'<brief_summary>.*?</brief_summary>', text,
                     re.DOTALL | re.IGNORECASE):
            score += 0.2
            details.append(f"{key}: +has_summary")

        # Truncated <reason> check
        if '<reason>' in text_lower and '</reason>' not in text_lower:
            severe_error = True
            details.append(f"{key}: TRUNCATED_REASON")

        # Premature answer + tool call
        has_boxed = '\\boxed{' in text
        has_tool_call = '<query>' in text_lower or '<act>' in text_lower
        if has_boxed and has_tool_call:
            score -= 0.4
            details.append(f"{key}: premature_answer")

        # Tool errors
        for tr in tool_results:
            status = tr.get('status', '')
            if status in ('PARSE_ERROR', 'SYNTAX_ERROR'):
                severe_error = True
                score -= 2.0
            elif status == 'KG_RELATION_ERROR':
                score -= 0.08

        # Compliance bonus
        if score >= 0.5:
            score += 0.3

        turn_scores.append(score)

    if severe_error:
        final_score = -2.0
    elif turn_scores:
        final_score = sum(turn_scores) / len(turn_scores)
    else:
        final_score = 0.0

    return {
        'score': max(-2.0, final_score),
        'details': details,
    }


def compute_behavior_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute behavior reward (anti-patterns).

    Mimics BehaviorReward from plug_v10.
    """
    VIOLATIONS = {
        'no_answer': -1.0,
        'empty_answer': -0.8,
        'gibberish': -0.8,
        'no_tool_calls': -0.6,
        'answer_hallucination': -0.4,
        'l1_early_guess': -1.0,
        'l2_plan_no_exec': -0.6,
        'l3_fake_insufficient': -0.3,
        'duplicate_calls': -0.2,
    }
    COMPLIANCE_BONUS = 0.5

    triggered = set()
    all_executed_sigs = set()
    evidence_entities = set()
    turn_count = 0
    has_plan = False
    has_execution = False
    retrieved_candidates = set()
    has_duplicate = False

    for key in sorted(rollout_info.keys()):
        if not key.startswith('turn_'):
            continue
        info = rollout_info[key]
        turn_count += 1

        queries = info.get('queries', [])
        for q in queries:
            tool_name = q.get('tool_name', '')
            args = q.get('arguments', {})
            if tool_name == 'action':
                sig = extract_action_signature(q)
                if sig in all_executed_sigs:
                    has_duplicate = True
                all_executed_sigs.add(sig)
            elif tool_name == 'plan':
                all_executed_sigs.add(
                    f"plan::{json.dumps(args, sort_keys=True, default=str)}")
            else:
                sig = f"{tool_name}::{json.dumps(args, sort_keys=True, default=str)}"
                if sig in all_executed_sigs:
                    has_duplicate = True
                all_executed_sigs.add(sig)

        # Entity evidence
        for tr in info.get('tool_results', []):
            evidence_entities.update(tr.get('found_end_entities', []))

        # Plan / execution tracking
        snapshot = info.get('kg_state_snapshot', {})
        if snapshot.get('plan'):
            has_plan = True
        if snapshot.get('any_match_executed'):
            has_execution = True

        cands = snapshot.get('retrieved_candidates', [])
        if isinstance(cands, (list, set)):
            retrieved_candidates.update(cands)

        # Feedback-based violations
        feedback = info.get('feedback', '')
        if 'Phase 1 Violation' in feedback:
            triggered.add('phase1_skip')
        if 'Invalid Domain' in feedback:
            triggered.add('domain_hallucination')
        if '[PRE-EXECUTION VALIDATION FAILED]' in feedback:
            triggered.add('relation_hallucination')

    if has_duplicate:
        triggered.add('duplicate_calls')

    # Get final answer
    last_text = ''
    for key in sorted(rollout_info.keys(), reverse=True):
        if key.startswith('turn_'):
            last_text = rollout_info[key].get('completion', '')
            break
    pred = parse_boxed_answers(last_text)
    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', last_text)

    if not boxed_matches:
        triggered.add('no_answer')
    else:
        ans = boxed_matches[-1].strip()
        if not ans:
            triggered.add('empty_answer')
        elif not re.search(r'[a-zA-Z0-9]', ans) and len(ans) < 2:
            triggered.add('gibberish')

    if len(all_executed_sigs) == 0:
        triggered.add('no_tool_calls')

    # Hallucination check
    if pred and evidence_entities:
        for p in pred:
            p_norm = normalize(p)
            if p_norm != "information insufficient":
                found = any(p_norm in normalize(str(e)) for e in evidence_entities)
                if not found:
                    triggered.add('answer_hallucination')
                    break

    # Lazy patterns
    if turn_count <= 1 and boxed_matches:
        triggered.add('l1_early_guess')
    if has_plan and not has_execution:
        triggered.add('l2_plan_no_exec')
    is_insufficient = any('insufficient' in str(p).lower() for p in pred)
    if is_insufficient and retrieved_candidates:
        triggered.add('l3_fake_insufficient')

    if not triggered:
        score = COMPLIANCE_BONUS
    else:
        score = sum(VIOLATIONS.get(v, 0) for v in triggered)

    return {
        'score': score,
        'violations': sorted(triggered),
        'turn_count': turn_count,
        'total_queries': len(all_executed_sigs),
    }


def compute_lazy_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute lazy reward.

    Mimics LazyReward from plug_v10.
    """
    GOOD_BONUS = 0.5
    penalty = 0.0

    turn_count = sum(1 for k in rollout_info if k.startswith('turn_'))
    last_text = ''
    for key in sorted(rollout_info.keys(), reverse=True):
        if key.startswith('turn_'):
            last_text = rollout_info[key].get('completion', '')
            break

    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', last_text)
    pred = parse_boxed_answers(last_text)

    # Early answer without exploration
    if turn_count <= 1 and boxed_matches:
        penalty -= 0.5

    # Insufficient answer progressive penalty
    max_invalid = 0
    for k, v in rollout_info.items():
        if k.startswith('turn_'):
            count = v.get('invalid_rejection_count', 0)
            max_invalid = max(max_invalid, count)
    if max_invalid > 0:
        penalty -= 0.1 * max_invalid * (max_invalid + 1) / 2

    score = GOOD_BONUS if penalty == 0.0 else penalty

    return {
        'score': score,
        'turn_count': turn_count,
        'early_answer': turn_count <= 1 and bool(boxed_matches),
        'insufficient_answer': bool(pred and any(
            'insufficient' in str(p).lower() for p in pred)),
    }


def compute_env_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute environment reward (tool execution success rate)."""
    successes = 0
    failures = 0
    for key in sorted(rollout_info.keys()):
        if not key.startswith('turn_'):
            continue
        for tr in rollout_info[key].get('tool_results', []):
            status = tr.get('status', '')
            if 'SUCCESS' in status:
                successes += 1
            elif 'ERROR' in status or status == 'PARSE_ERROR':
                failures += 1
            else:
                # Unknown status, count as attempt
                successes += 1  # Neutral

    total = successes + failures
    score = successes / total if total > 0 else 0.0

    return {
        'score': score,
        'tool_successes': successes,
        'tool_failures': failures,
        'total_tools': total,
    }


def compute_plan_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute plan quality reward.

    F1 of planned generators vs ground-truth action results.
    """
    LAZY_NO_PLAN = -0.8
    LAZY_EMPTY = -0.5
    LAZY_NO_ACTIONS = -0.8

    gt = case.get('ground_truth', [])
    gt_norm = {normalize(g) for g in gt} if gt else set()

    has_plan = False
    generator_relations = []
    action_hints = []
    action_results = {}

    for key in sorted(rollout_info.keys()):
        if not key.startswith('turn_'):
            continue
        info = rollout_info[key]
        if info.get('generator_relations') or info.get('plan_relations'):
            has_plan = True
        generator_relations.extend(info.get('generator_relations', []))
        action_hints.extend(info.get('action_hints', []))
        action_results.update(info.get('action_results', {}))

    if not has_plan:
        return {'score': LAZY_NO_PLAN, 'reason': 'no_plan'}

    total_generators = len(generator_relations)
    if total_generators == 0:
        return {'score': LAZY_EMPTY, 'reason': 'empty_generators'}

    total_actions = len(action_hints)
    if total_actions == 0:
        return {'score': LAZY_NO_ACTIONS, 'reason': 'no_actions'}

    # Count valid actions (return GT entities)
    valid_actions = 0
    gt_found = set()
    for sig, result in action_results.items():
        if result.get('has_gt', False):
            valid_actions += 1
            entities_norm = {normalize(e) for e in result.get('entities', [])}
            gt_found.update(gt_norm & entities_norm)

    # Count useful generators
    useful_generators = 0
    generator_set = set(generator_relations)
    for sig, result in action_results.items():
        if result.get('has_gt', False):
            action_rels = result.get('core_relations', [])
            if action_rels:
                if generator_set & set(action_rels):
                    useful_generators += 1
            else:
                for rel in generator_relations:
                    if rel in sig:
                        useful_generators += 1
                        break

    precision = useful_generators / total_generators if total_generators > 0 else 0
    recall = valid_actions / total_actions if total_actions > 0 else 0
    if precision + recall > 0:
        f1_gen = 2 * precision * recall / (precision + recall)
    else:
        f1_gen = 0

    coverage = len(gt_found) / len(gt_norm) if gt_norm else 0
    score = f1_gen * coverage

    return {
        'score': score,
        'f1_generator': f1_gen,
        'coverage': coverage,
        'useful_generators': useful_generators,
        'total_generators': total_generators,
        'valid_actions': valid_actions,
        'total_actions': total_actions,
    }


def compute_action_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute action quality reward."""
    LAZY_NO_EXECUTION = -0.8

    gt = case.get('ground_truth', [])
    gt_norm = {normalize(g) for g in gt} if gt else set()

    has_action_hints = False
    total_valid_actions = 0
    executed_actions = []
    match_results = {}

    for key in sorted(rollout_info.keys()):
        if not key.startswith('turn_'):
            continue
        info = rollout_info[key]

        if info.get('action_hints'):
            has_action_hints = True

        for sig, result in info.get('action_results', {}).items():
            if result.get('has_gt', False):
                total_valid_actions += 1

        for q in info.get('queries', []):
            if q.get('tool_name') == 'action':
                sig = extract_action_signature(q)
                executed_actions.append(sig)

        match_results.update(info.get('action_results', {}))

    total_executed = len(executed_actions)
    if has_action_hints and total_executed == 0:
        return {'score': LAZY_NO_EXECUTION, 'reason': 'no_execution'}

    if not has_action_hints:
        return {'score': 0.0, 'reason': 'no_hints'}

    # Calculate metrics
    valid_executed = 0
    gt_found = set()
    for sig in executed_actions:
        if sig in match_results:
            result = match_results[sig]
            entities_norm = {normalize(e)
                             for e in result.get('entities', [])}
            found = gt_norm & entities_norm
            if found:
                valid_executed += 1
                gt_found.update(found)

    precision = valid_executed / total_executed if total_executed > 0 else 0
    recall = valid_executed / total_valid_actions if total_valid_actions > 0 else 0
    if precision + recall > 0:
        f1_action = 2 * precision * recall / (precision + recall)
    else:
        f1_action = 0

    coverage = len(gt_found) / len(gt_norm) if gt_norm else 0
    score = f1_action * coverage

    return {
        'score': score,
        'f1_action': f1_action,
        'coverage': coverage,
        'valid_executed': valid_executed,
        'total_executed': total_executed,
        'total_valid_actions': total_valid_actions,
    }


def compute_reasoning_reward(case: Dict, rollout_info: Dict) -> Dict:
    """Compute reasoning compression reward.

    r_reason = r_exec * F1(Y) * (1 + alpha * Compression)
    """
    HALLUCINATION_PENALTY = -1.0
    ALPHA = 0.3

    gt = case.get('ground_truth', [])

    # Collect U: union of all action tool results
    U = set()
    for key in sorted(rollout_info.keys()):
        if not key.startswith('turn_'):
            continue
        for tr in rollout_info[key].get('tool_results', []):
            if tr.get('tool_name') == 'action':
                U.update(tr.get('found_end_entities', []))

    # Get Y: final boxed answers
    last_text = ''
    for key in sorted(rollout_info.keys(), reverse=True):
        if key.startswith('turn_'):
            last_text = rollout_info[key].get('completion', '')
            break
    Y = set(parse_boxed_answers(last_text))

    if not Y or not U:
        return {'score': 0.0, 'reason': 'no_answer_or_pool'}

    # Fuzzy subset check
    Y_norm = {normalize(y) for y in Y}
    U_norm = {normalize(u) for u in U}
    y_in_u = sum(1 for y in Y_norm
                 if any(y in u or u in y for u in U_norm))
    if y_in_u < len(Y_norm) * 0.5:
        return {'score': HALLUCINATION_PENALTY, 'reason': 'hallucination'}

    # r_exec: execution quality proxy
    f1_pool = calculate_f1(list(U), gt) if gt else 0.0
    r_exec = f1_pool

    # F1(Y)
    f1_final = calculate_f1(list(Y), gt) if gt else 0.0

    # Compression
    compress = (len(U) - len(Y)) / (len(U) + 1e-9)
    compress = max(0.0, compress)

    score = r_exec * f1_final * (1 + ALPHA * compress)

    # Count reason blocks
    reason_blocks = re.findall(r'<reason>(.*?)</reason>', last_text,
                               re.DOTALL | re.IGNORECASE)

    return {
        'score': score,
        'r_exec': r_exec,
        'f1_final': f1_final,
        'compression': compress,
        'pool_size': len(U),
        'answer_size': len(Y),
        'reason_blocks': len(reason_blocks),
    }


def compute_all_rewards(case: Dict) -> Dict:
    """Compute all 8 reward scores for a single case."""
    rollout_info = reconstruct_rollout_info(case)

    rewards = {
        'f1': compute_f1_reward(case, rollout_info),
        'compliance': compute_compliance_reward(case, rollout_info),
        'behavior': compute_behavior_reward(case, rollout_info),
        'lazy': compute_lazy_reward(case, rollout_info),
        'env': compute_env_reward(case, rollout_info),
        'plan': compute_plan_reward(case, rollout_info),
        'action': compute_action_reward(case, rollout_info),
        'reasoning': compute_reasoning_reward(case, rollout_info),
    }

    return rewards


# ===========================================================================
# 4. Comparison and reporting
# ===========================================================================

REWARD_NAMES = ['f1', 'compliance', 'behavior', 'lazy', 'env',
                'plan', 'action', 'reasoning']
REWARD_SHORT = {
    'f1': 'F1',
    'compliance': 'Comply',
    'behavior': 'Behav',
    'lazy': 'Lazy',
    'env': 'Env',
    'plan': 'Plan',
    'action': 'Action',
    'reasoning': 'Reason',
}


def build_case_map(results: List[Dict]) -> Dict[str, Dict]:
    """Build case_id -> result dict mapping."""
    return {r['case_id']: r for r in results}


def compute_reward_table(
    baseline_results: List[Dict],
    selective_results: List[Dict],
) -> List[Dict]:
    """Compute reward comparison for all matching cases.

    Returns list of dicts with:
    - case_id
    - baseline_scores: {reward_name: score}
    - selective_scores: {reward_name: score}
    - deltas: {reward_name: delta}
    - total_delta
    - baseline_rewards: full reward dicts
    - selective_rewards: full reward dicts
    """
    baseline_map = build_case_map(baseline_results)
    selective_map = build_case_map(selective_results)

    common_ids = sorted(set(baseline_map.keys()) & set(selective_map.keys()))

    table = []
    for cid in common_ids:
        b_case = baseline_map[cid]
        s_case = selective_map[cid]

        b_rewards = compute_all_rewards(b_case)
        s_rewards = compute_all_rewards(s_case)

        b_scores = {name: b_rewards[name]['score'] for name in REWARD_NAMES}
        s_scores = {name: s_rewards[name]['score'] for name in REWARD_NAMES}
        deltas = {name: s_scores[name] - b_scores[name]
                  for name in REWARD_NAMES}

        total_b = sum(b_scores.values())
        total_s = sum(s_scores.values())

        table.append({
            'case_id': cid,
            'question': b_case.get('question', '')[:60],
            'baseline_scores': b_scores,
            'selective_scores': s_scores,
            'deltas': deltas,
            'total_baseline': total_b,
            'total_selective': total_s,
            'total_delta': total_s - total_b,
            'baseline_rewards': b_rewards,
            'selective_rewards': s_rewards,
        })

    # Sort by total_delta ascending (worst degradation first)
    table.sort(key=lambda x: x['total_delta'])

    return table


def format_score(val: float) -> str:
    """Format a score value for display."""
    if val == 0.0:
        return "0.00"
    if val > 0:
        return f"+{val:.2f}"
    return f"{val:.2f}"


def format_delta(val: float) -> str:
    """Format a delta value for display with color indicator."""
    if val > 0.01:
        return f"+{val:.2f}"
    elif val < -0.01:
        return f"{val:.2f}"
    else:
        return " ~0.00"


def generate_markdown_report(
    table: List[Dict],
    baseline_path: str,
    selective_path: str,
    top_delta: int,
) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# RL Reward Score Analysis")
    lines.append("")
    lines.append(f"- **Baseline**: `{os.path.basename(baseline_path)}`")
    lines.append(f"- **Selective**: `{os.path.basename(selective_path)}`")
    lines.append(f"- **Common cases**: {len(table)}")
    lines.append(f"- **Top delta shown**: {top_delta}")
    lines.append("")

    # Summary statistics
    if table:
        b_totals = [t['total_baseline'] for t in table]
        s_totals = [t['total_selective'] for t in table]
        d_totals = [t['total_delta'] for t in table]

        avg_b = sum(b_totals) / len(b_totals)
        avg_s = sum(s_totals) / len(s_totals)
        avg_d = sum(d_totals) / len(d_totals)

        worse = sum(1 for d in d_totals if d < -0.1)
        better = sum(1 for d in d_totals if d > 0.1)
        same = len(d_totals) - worse - better

        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Baseline | Selective | Delta |")
        lines.append("|--------|----------|-----------|-------|")
        lines.append(f"| Avg Total Reward | {avg_b:.2f} | {avg_s:.2f} | "
                     f"{format_delta(avg_d)} |")
        lines.append("")
        lines.append(f"- Cases where selective is **worse** (delta < -0.1): {worse}")
        lines.append(f"- Cases where selective is **better** (delta > +0.1): {better}")
        lines.append(f"- Cases roughly **same**: {same}")
        lines.append("")

        # Per-reward averages
        lines.append("## Per-Reward Averages")
        lines.append("")
        header = "| Reward | Baseline | Selective | Delta |"
        sep = "|--------|----------|-----------|-------|"
        lines.append(header)
        lines.append(sep)
        for name in REWARD_NAMES:
            b_vals = [t['baseline_scores'][name] for t in table]
            s_vals = [t['selective_scores'][name] for t in table]
            b_avg = sum(b_vals) / len(b_vals)
            s_avg = sum(s_vals) / len(s_vals)
            d_avg = s_avg - b_avg
            lines.append(
                f"| {REWARD_SHORT[name]:8s} | {b_avg:+.2f}    | {s_avg:+.2f}    | "
                f"{format_delta(d_avg):8s} |")
        lines.append("")

    # Top-delta cases table
    lines.append(f"## Top {top_delta} Highest-Delta Cases (Selective Hurts Most)")
    lines.append("")

    # Header
    col_names = [REWARD_SHORT[n] for n in REWARD_NAMES]
    header = "| Case | " + " | ".join(f"{c:>6s}" for c in col_names) + " | Total |"
    sep = "|------|" + "|".join(["-------" for _ in col_names]) + "|-------|"
    lines.append(header)
    lines.append(sep)

    shown = table[:top_delta]
    for entry in shown:
        cid = entry['case_id'].replace('WebQTest-', 'WQT-')
        b_parts = [f"{entry['baseline_scores'][n]:+.2f}" for n in REWARD_NAMES]
        s_parts = [f"{entry['selective_scores'][n]:+.2f}" for n in REWARD_NAMES]
        d_parts = [f"{entry['deltas'][n]:+.2f}" for n in REWARD_NAMES]

        # Baseline row
        lines.append(f"| {cid} B | " + " | ".join(f"{p:>6s}" for p in b_parts) +
                     f" | {entry['total_baseline']:+.2f} |")
        # Selective row
        lines.append(f"| {cid} S | " + " | ".join(f"{p:>6s}" for p in s_parts) +
                     f" | {entry['total_selective']:+.2f} |")
        # Delta row
        lines.append(f"| {'':>4s} D | " + " | ".join(f"{p:>6s}" for p in d_parts) +
                     f" | {entry['total_delta']:+.2f} |")
        lines.append(sep)

    lines.append("")

    # Detailed analysis for top-5 worst cases
    lines.append("## Detailed Analysis: Top 5 Worst Degradation")
    lines.append("")

    for entry in table[:5]:
        cid = entry['case_id']
        lines.append(f"### {cid}")
        lines.append("")
        lines.append(f"**Question**: {entry['question']}")
        lines.append("")
        lines.append("| Reward | Baseline | Selective | Delta |")
        lines.append("|--------|----------|-----------|-------|")
        for name in REWARD_NAMES:
            b = entry['baseline_scores'][name]
            s = entry['selective_scores'][name]
            d = entry['deltas'][name]
            lines.append(
                f"| {REWARD_SHORT[name]:8s} | {b:+.2f}    | {s:+.2f}    | "
                f"{format_delta(d):8s} |")
        lines.append(
            f"| **TOTAL** | **{entry['total_baseline']:+.2f}** | "
            f"**{entry['total_selective']:+.2f}** | **{format_delta(entry['total_delta'])}** |")
        lines.append("")

        # Behavior violations comparison
        b_violations = entry['baseline_rewards']['behavior'].get('violations', [])
        s_violations = entry['selective_rewards']['behavior'].get('violations', [])
        if b_violations != s_violations:
            lines.append(f"- Baseline violations: {b_violations or 'none'}")
            lines.append(f"- Selective violations: {s_violations or 'none'}")
            lines.append("")

        # Plan quality comparison
        b_plan = entry['baseline_rewards']['plan']
        s_plan = entry['selective_rewards']['plan']
        if abs(b_plan['score'] - s_plan['score']) > 0.05:
            lines.append(f"- Baseline plan: {b_plan}")
            lines.append(f"- Selective plan: {s_plan}")
            lines.append("")

    # Cases where selective improves most
    lines.append(f"## Top {top_delta} Cases Where Selective Improves")
    lines.append("")
    improved = sorted(table, key=lambda x: x['total_delta'], reverse=True)
    lines.append(header)
    lines.append(sep)
    for entry in improved[:top_delta]:
        cid = entry['case_id'].replace('WebQTest-', 'WQT-')
        s_parts = [f"{entry['selective_scores'][n]:+.2f}" for n in REWARD_NAMES]
        b_parts = [f"{entry['baseline_scores'][n]:+.2f}" for n in REWARD_NAMES]
        d_parts = [f"{entry['deltas'][n]:+.2f}" for n in REWARD_NAMES]

        lines.append(f"| {cid} S | " + " | ".join(f"{p:>6s}" for p in s_parts) +
                     f" | {entry['total_selective']:+.2f} |")
        lines.append(f"| {cid} B | " + " | ".join(f"{p:>6s}" for p in b_parts) +
                     f" | {entry['total_baseline']:+.2f} |")
        lines.append(f"| {'':>4s} D | " + " | ".join(f"{p:>6s}" for p in d_parts) +
                     f" | {entry['total_delta']:+.2f} |")
        lines.append(sep)
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Retroactively compute RL reward scores from test results')
    parser.add_argument('--baseline', required=True,
                        help='Path to baseline results.json')
    parser.add_argument('--selective', required=True,
                        help='Path to selective consistency results.json')
    parser.add_argument('--top-delta', type=int, default=20,
                        help='Number of top-delta cases to highlight')
    args = parser.parse_args()

    print(f"[INFO] Loading baseline: {args.baseline}", file=sys.stderr)
    baseline = load_results(args.baseline)
    print(f"[INFO] Loading selective: {args.selective}", file=sys.stderr)
    selective = load_results(args.selective)

    print("[INFO] Computing rewards for all cases...", file=sys.stderr)
    table = compute_reward_table(baseline, selective)
    print(f"[INFO] Computed rewards for {len(table)} common cases.",
          file=sys.stderr)

    # Generate report
    report = generate_markdown_report(
        table, args.baseline, args.selective, args.top_delta)

    # Print to stdout
    print(report)

    # Save alongside baseline results
    baseline_dir = os.path.dirname(os.path.abspath(args.baseline))
    out_path = os.path.join(baseline_dir, 'rl_reward_analysis.md')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n[INFO] Report saved to {out_path}", file=sys.stderr)

    # Also save as JSON for programmatic access
    json_out = os.path.join(baseline_dir, 'rl_reward_scores.json')
    json_data = []
    for entry in table:
        json_data.append({
            'case_id': entry['case_id'],
            'question': entry['question'],
            'baseline_scores': entry['baseline_scores'],
            'selective_scores': entry['selective_scores'],
            'deltas': entry['deltas'],
            'total_baseline': entry['total_baseline'],
            'total_selective': entry['total_selective'],
            'total_delta': entry['total_delta'],
        })
    with open(json_out, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[INFO] JSON scores saved to {json_out}", file=sys.stderr)


if __name__ == '__main__':
    main()
