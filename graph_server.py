# file: graph_server_final.py
# V13.0 - Annotated YAML Topology & Common Property Extraction

import os
import sys
import re
import difflib
import pickle
import json
import logging
import argparse
import asyncio
import yaml
from collections import defaultdict, Counter, deque
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Set, Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

try:
    import graph_tool.all as gt
except ImportError:
    logging.warning("Graph-tool not found. Graph operations will fail.")
    pass

# ==============================================================================
# 1. Config
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdaptiveConfig:
    MAX_TOPOLOGY_NODES = 50       # Topology 树中展示多少个分支
    MAX_DETAIL_NODES = 50         # Details 中展示多少个节点的属性
    MAX_INLINE_LIST = 5           # 属性值列表截断长度
    COMMON_PROP_THRESHOLD = 0.9   # 属性值重复率超过多少视为公共属性

IGNORED_RELATIONS = {
    'type.object.type', 'type.object.permission', 'common.topic.article',
    'type.object.name', 'common.topic.image', 'freebase.valuenotation.is_reviewed',
    'freebase.valuenotation.has_no_value'
}

IGNORED_ENTITY_STRINGS = {
    "District represented (if position is district-related)",
    "Office, position, or title",
    "Jurisdiction of office",
    "Governmental body (if position is part of one)",
    "Basic title",
    "Officeholder",
    "Legislative sessions",
    "Appointed By (if Position is Appointed)"
}

# ==============================================================================
# 2. Models
# ==============================================================================

class StructuredData(BaseModel):
    """结构化数据，用于压缩和重建 - 仅 match_pattern 使用"""
    # 起始实体
    start_entity: str = ""
    # 执行的路径定义
    path: List[Dict] = Field(default_factory=list)
    # 叶子节点（目标实体）- 永不压缩
    leaf_entities: List[str] = Field(default_factory=list)
    # CVT 节点及其属性 {cvt_id: {prop: [values]}}
    cvt_nodes: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    # 路径三元组 [(subject, relation, object), ...]
    path_triples: List[List[str]] = Field(default_factory=list)
    # 逻辑模式（抽象路径）
    logic_pattern: str = ""
    
    # [NEW] 分支结构 - 按 path_length 排序（短优先）
    # 格式: [{'path_length': int, 'path_triples': [[s,r,o],...], 'leaf_entity': str, 'branch_signature': str}]
    branches: List[Dict] = Field(default_factory=list)
    
    # [NEW] 叶子实体到最短路径长度的映射（用于去重时保留最短路径）
    leaf_to_shortest_path: Dict[str, int] = Field(default_factory=dict)
    
    # [NEW] Full data tree for multi-hop topology reconstruction
    data_tree: Dict = Field(default_factory=dict)
    
    # [NEW] Native path leaves (non-CVT)
    native_leaves: List[str] = Field(default_factory=list)
    
    # [NEW] CVT-expanded leaves
    cvt_expanded_leaves: List[str] = Field(default_factory=list)
    
    # [NEW] CVT-to-leaf map for CVT-ending paths
    cvt_leaf_map: Dict[str, List[str]] = Field(default_factory=dict)


class FinalResponse(BaseModel):
    success: bool
    status: str
    response_text: str  
    found_end_entities: List[str] = Field(default_factory=list)
    action_hints: List[Dict[str, Any]] = Field(default_factory=list) # Moved and kept the more specific type hint
    # 新增：结构化数据（用于压缩重建）
    structured_data: Optional[StructuredData] = None
    # [NEW] Valid action hints from find_logical_path - list of action dicts: {'start_entity': str, 'steps': List[Dict]}
    # The comment for action_hints is kept here, as it was originally associated with this position.


class ExploreSchemaRequest(BaseModel):
    sample_id: str
    pattern: str

class PathRequest(BaseModel):
    sample_id: str
    start_entity: str
    contains_relation: str
    max_hops: int = 3
    limit: int = 50 

class NeighborsRequest(BaseModel):
    sample_id: str
    entity: str
    direction: str = "out"
    limit: int = 10

class RelationTuple(BaseModel):
    relation: str
    direction: str = Field(..., pattern="^(out|in)$")

class MatchRequestV2(BaseModel):
    sample_id: str
    anchor: str  # Unify to single string anchor
    path: List[RelationTuple]
    # Constraint query mode: uses format_constraint_results for filtering
    is_constraint_query: bool = False
    constraint_relation: str = None  # The constraint being queried (for display)
    # [NEW] Select feature: attribute/entity verification for leaf nodes
    select_relations: List[str] = []  # Full-qualified relations to fetch (e.g., "people.person.gender")
    select_entities: List[str] = []   # Verified entities to check connectivity (2-hop)
    
    # [V13 FIX] Support Aliased Constraints (Model uses constraint_*, backend uses select_*)
    constraint_relations: List[str] = []
    constraint_entities: List[str] = []
    
    @validator('anchor', pre=True)
    def normalize_anchor(cls, v):
        """Accept both string and list[string], normalize to string.
        
        Models often mistakenly pass anchor as ["Entity"] instead of "Entity".
        This validator handles both formats gracefully.
        """
        if isinstance(v, list):
            if len(v) >= 1:
                return v[0]  # Take first element
            return ""  # Empty list edge case
        return v

class BatchConstraintRequest(BaseModel):
    """Request for batch constraint query with automatic path discovery."""
    sample_id: str
    candidates: List[str]           # List of candidate entities (e.g., 20 candidates)
    constraint_relation: str        # Constraint relation (e.g., "people.person.gender")
    max_hops: int = 2              # Max hops for path discovery (default 2 to avoid explosion)
    limit_per_entity: int = 3      # Max paths per entity (handle multi-path issue)

class EntitiesRequest(BaseModel):
    sample_id: str
    entity_substring: str
    limit: int = 10

class SemanticRetrieveRequest(BaseModel):
    """Request for GTE-based semantic retrieval against full subgraph."""
    sample_id: str
    queries: List[str] = Field(..., description="List of text queries (question/sub-questions)")
    top_k: int = 10
    gte_url: str = "http://localhost:8003"

class SemanticRetrieveResponse(BaseModel):
    success: bool
    entities_per_query: List[List[Dict]] = Field(default_factory=list)  # [{candidate, score}] per query
    relations_per_query: List[List[Dict]] = Field(default_factory=list)  # [{candidate, score}] per query
    total_entities: int = 0
    total_relations: int = 0

# ==============================================================================
# 3. Helpers
# ==============================================================================

def _is_cvt_id(entity_name: str) -> bool:
    if not entity_name: return False
    return bool(re.match(r'^[mg]\.[a-zA-Z0-9_]+$', str(entity_name)))

def _build_match_pattern_code(anchor: str, raw_triples: List[List[str]], 
                               select_relations: List[str] = None, 
                               select_entities: List[str] = None) -> str:
    """Build match_pattern code with optional select parameters."""
    steps = []
    curr = anchor
    if raw_triples and raw_triples[0][0].lower() == anchor.lower():
        curr = raw_triples[0][0]

    for s, p, o in raw_triples:
        if s == curr: direction, curr = "out", o
        else: direction, curr = "in", s
        steps.append(f'{{"relation": "{p}", "direction": "{direction}"}}')
    
    path_str = ", ".join(steps)
    
    # Build optional select parameters
    select_parts = []
    if select_relations:
        rel_str = ', '.join(f'"{r}"' for r in select_relations)
        select_parts.append(f'select_relations=[{rel_str}]')
    if select_entities:
        ent_str = ', '.join(f'"{e}"' for e in select_entities)
        select_parts.append(f'select_entities=[{ent_str}]')
    
    select_str = ', '.join(select_parts)
    # [V18] Use "action" tool name (frontend maps to match_pattern endpoint)
    if select_str:
        return f'action(anchor="{anchor}", path=[{path_str}], {select_str})'
    else:
        return f'action(anchor="{anchor}", path=[{path_str}])'

def _simplify_relation_name(relation: str) -> str:
    return relation.split('.')[-1]

def format_schema_hierarchical(pattern: str, relations: List[str]) -> str:
    """
    Format schema relations as a grouped list of full paths.
    Using full paths prevents 'domain loss' errors by the model.
    Grouping by domain maintains readability.
    """
    if not relations: return "status: no_relations_found"
    
    # Group by domain
    groups = {}
    for rel in relations:
        domain = rel.split('.')[0]
        if domain not in groups:
            groups[domain] = []
        groups[domain].append(rel)
        
    # Build YAML-like output
    lines = []
    for domain in sorted(groups.keys()):
        lines.append(f"{domain}:")
        for rel in sorted(groups[domain]):
            lines.append(f"  - {rel}")
            
    return "\n".join(lines)

def format_neighbors(entity: str, neighbors: Dict[str, List[Tuple]]) -> str:
    if not neighbors: return "Status: No neighbors found."
    lines = [f"Neighbors for ({entity}):"]
    for direction in ['out', 'in']:
        if direction not in neighbors: continue
        triples = neighbors[direction]
        if not triples: continue
        lines.append(f"  {direction.upper()} Edges:")
        grouped = defaultdict(list)
        for s, p, o in triples:
            target = o if direction == 'out' else s
            grouped[p].append(target)
        for rel, targets in grouped.items():
            val_str = "[" + ", ".join(targets[:10]) + "]" 
            if len(targets) > 10: val_str += f" (+{len(targets)-10} more)"
            lines.append(f"    {rel}: {val_str}")
    return "\n".join(lines)


def format_logical_paths(start_entity: str, paths: List[Dict],
                         select_relations: List[str] = None,
                         select_entities: List[str] = None) -> str:
    """
    Optimized Logical Path Formatter (v23 - Mixed-Hop RDF Standard)
    
    Features:
    1. STRICT RDF Syntax: (Subject: X, Predicate: Y, Object: Z)
    2. Variable Chaining: Uses {Intermediate_N} to link multi-hop steps correctly.
    3. Direction Agnostic: The logic holds true regardless of IN/OUT combinations.
    4. Select Parameters: Includes select_relations/select_entities in Action hints.
    
    Example Output (Mixed Path):
      Step 1: (Subject: {Intermediate_1}, Predicate: starring, Object: (Keanu))  [IN]
      Step 2: (Subject: {Intermediate_1}, Predicate: director, Object: {Candidate}) [OUT]
    """
    if not paths: return "Status: No logical paths found."

    grouped = defaultdict(list)
    pattern_samples = {}

    for p in paths:
        triples = p.get('triples', [])
        if not triples: continue
        
        logic_steps = []
        # Robust start entity casing lookup
        if triples and triples[0][0].lower() == start_entity.lower(): 
            curr = triples[0][0]
        else:
            curr = start_entity
            
        for s, rel, o in triples:
            if s == curr: 
                direction, curr = "OUT", o
            else: 
                direction, curr = "IN", s
            logic_steps.append((rel, direction))
            
        signature = tuple(logic_steps)
        grouped[signature].append(p)
        if signature not in pattern_samples: 
            pattern_samples[signature] = triples

    lines = [f"Logical Paths for ({start_entity}):"]
    lines.append("(Format: Sequence of RDF Triples. Verify if the Subject-Predicate-Object logic is valid.)")
    
    sorted_patterns = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

    for idx, (sig, matches) in enumerate(sorted_patterns):
        count = len(matches)
        raw_triples = pattern_samples[sig]
        
        lines.append("-" * 10)
        lines.append(f"PATH {idx+1} (matches {count} instances):")
        
        # Explicitly track the "Current Node" variable name in the chain
        curr_node_var = f"({start_entity})" 
        
        for i, (rel, direction) in enumerate(sig):
            # 1. Define the Next Node variable name
            if i == len(sig) - 1: 
                # The final node is always the Candidate
                next_node_var = "{Candidate}"      
            else:
                # Intermediate nodes get a unique ID
                next_node_var = f"{{Intermediate_{i+1}}}" 
            
            # Simplify relation for readability (remove domain prefix if too long)
            simple_rel = rel.split('.')[-1]
            
            # 2. Assign Subject/Object based on Direction (The Core Logic)
            if direction == "OUT":
                # Forward: Current -> Next
                # Logic: Current is Subject
                subj = curr_node_var
                obj  = next_node_var
            else:
                # Reverse: Next -> Current
                # Logic: Next is Subject (e.g., The Movie is the subject of 'starring')
                subj = next_node_var
                obj  = curr_node_var
            
            # 3. Print the Fact
            lines.append(f"  Step {i+1}: (Subject: {subj}, Predicate: {simple_rel}, Object: {obj})")
            
            # 4. Advance the chain: The 'Next' of this step becomes 'Current' of the next
            curr_node_var = next_node_var
            
        # Action Hint (with select params)
        hint = _build_match_pattern_code(start_entity, raw_triples, select_relations, select_entities)
        lines.append(f"  [Action]: {hint}")
        
        # [V14] Analogical Example as Natural Language sentences
        # Format: "A's [relation] is B; B's [relation] is C; ..."
        # This helps model understand directional semantics (e.g., "Brewers's [previously_known_as] is Pilots")
        if raw_triples:
            nl_parts = []
            for s, rel, o in raw_triples:
                s_name = s if isinstance(s, str) else str(s)
                o_name = o if isinstance(o, str) else str(o)
                # Use simplified relation name for readability
                simple_rel = rel.split('.')[-1]
                nl_parts.append(f"{s_name}'s [{simple_rel}] is {o_name}")
            if nl_parts:
                nl_sentence = "; ".join(nl_parts)
                lines.append(f"  [Analogical Example (Read as sentence)]:")
                lines.append(f"    \"{nl_sentence}.\"")
                lines.append(f"  ⚠️ This sentence shows the SEMANTIC DIRECTION. Verify if it matches the question's intent.")
        
    return "\n".join(lines)


def format_batch_match_results(all_results: Dict[str, Dict], path: List, matcher) -> str:
    """Format results from batch entity execution using entity pairs.
    
    Shows Start(entity), End(entity) pairs to save tokens.
    Model infers relation from Logic Pattern.
    
    Compression strategy:
    - Focus on compressing BRANCHES, not leaf entities
    - Single/few branches: show ALL entities (completeness > token saving)
    - Many branches (>5): apply branch-level compression, but still show all entities per branch
    """
    if not all_results:
        return "Status: No matches found."
    
    # Determine compression limits based on branch count
    # Core principle: compress branches, not entities within branches
    # But keep a defensive hard limit per branch
    num_branches = len(all_results)
    
    # Defensive limit: max 60 entities per branch regardless of branch count
    per_branch_limit = 60
    
    # Build logic pattern description (same for all entities) - RDF format
    logic_steps = []
    for i, step in enumerate(path):
        next_ph = "{End}" if i == len(path) - 1 else f"{{Node_{i+1}}}"
        if i == 0:
            prev_ph = "{Start}"
        else:
            prev_ph = f"{{Node_{i}}}"
        
        # Simplify relation for readability
        simple_rel = step.relation.split('.')[-1]
        
        if step.direction == "out":
            # Forward: Current -> Next => Current is Subject
            logic_steps.append(f"Step {i+1}: (Subject: {prev_ph}, Predicate: {simple_rel}, Object: {next_ph})")
        else:
            # Reverse: Next -> Current => Next is Subject
            logic_steps.append(f"Step {i+1}: (Subject: {next_ph}, Predicate: {simple_rel}, Object: {prev_ph})")
    
    lines = [
        f"[Batch Execution Results]",
        f"Entities Processed: {len(all_results)}",
        "-" * 50,
        "[Logic Pattern]:",
        "(Format: Sequence of RDF Triples. Subject-Predicate-Object.)"
    ]
    lines.extend(f"  {lt}" for lt in logic_steps)
    lines.append("-" * 10)
    
    # For each start entity, show entity pairs
    lines.append("[Entity Pairs]:")
    all_ends = set()
    
    for start_entity, data in all_results.items():
        ends = data['ends']
        all_ends.update(ends)
        
        if not ends:
            lines.append(f"  Start({start_entity}): (No matches)")
            continue
        
        # Show as: Start(X), End(Y1, Y2, ...)
        if len(ends) <= per_branch_limit:
            ends_str = ", ".join(ends)
        else:
            ends_str = ", ".join(ends[:per_branch_limit]) + f"... (+{len(ends)-per_branch_limit})"
        
        lines.append(f"  Start({start_entity}), End({ends_str})")
    
    # Summary - always show complete list (no truncation)
    lines.append("")
    lines.append("-" * 10)
    lines.append(f"Leaf Entities ({len(all_ends)}):")
    sorted_ends = sorted(list(all_ends))
    lines.append(f"  {sorted_ends}")
    
    return "\n".join(lines)


def format_constraint_results(all_results: Dict[str, Dict], path: List, matcher, 
                               constraint_relation: str = None) -> str:
    """Format constraint query results with CVT expansion.
    
    Designed for constraint exploration where we need to see:
    1. Logic pattern (the path taken) - skipped if path=None (auto-discovered)
    2. Each candidate's endpoint values
    3. CVT node properties at endpoints (for filtering)
    
    Args:
        all_results: {entity: {'paths': [...], 'ends': [...]}}
        path: The path definition executed (None for auto-discovered paths)
        matcher: GraphPatternMatcher for CVT expansion
        constraint_relation: The constraint relation being queried (e.g., 'people.person.gender')
    """
    if not all_results:
        return "Status: No matches found."
    
    lines = [
        f"[Constraint Query Results]",
        f"Constraint: {constraint_relation or 'N/A'}",
        f"Candidates Queried: {len(all_results)}",
        "-" * 50,
    ]
    
    # Build logic pattern description (skip if path is None - auto-discovered)
    if path:
        logic_steps = []
        for i, step in enumerate(path):
            next_ph = "{End}" if i == len(path) - 1 else f"{{Node_{i+1}}}"
            prev_ph = "{Start}" if i == 0 else f"{{Node_{i}}}"
            
            if step.direction == "out":
                logic_steps.append(f"({prev_ph}, {step.relation}, {next_ph})")
            else:
                logic_steps.append(f"({next_ph}, {step.relation}, {prev_ph})")
        
        lines.append("[Logic Pattern]:")
        lines.extend(f"  {lt}" for lt in logic_steps)
        lines.append("-" * 10)
    
    # For each candidate, show endpoint and CVT properties
    lines.append("[Candidate Attributes]:")
    
    all_ends = set()
    end_to_candidates = {}  # Group candidates by their end value
    
    for candidate, data in all_results.items():
        ends = data.get('ends', [])
        paths_data = data.get('paths', [])
        
        if not ends:
            lines.append(f"  {candidate}: (No match)")
            continue
        
        all_ends.update(ends)
        
        # Collect CVT nodes from paths (second-to-last node if exists)
        cvt_nodes = set()
        for p in paths_data:
            triples = p.get('triples', [])
            if len(triples) >= 2:
                # Get the node before {End}
                last_triple = triples[-1]
                second_last = triples[-2] if len(triples) >= 2 else None
                if second_last:
                    # Check if it's a CVT
                    potential_cvt = second_last[2] if second_last[0] == candidate else second_last[0]
                    if matcher.is_cvt(potential_cvt):
                        cvt_nodes.add(potential_cvt)
        
        # Format output for this candidate
        if len(ends) == 1 and not cvt_nodes:
            # Simple case: direct attribute
            end_val = ends[0]
            lines.append(f"  {candidate}: {end_val}")
            end_to_candidates.setdefault(end_val, []).append(candidate)
        else:
            # Complex case: show ends with CVT properties
            lines.append(f"  {candidate}:")
            for end_val in sorted(ends)[:5]:
                lines.append(f"    - {end_val}")
                end_to_candidates.setdefault(end_val, []).append(candidate)
            
            # Expand CVT nodes if present
            if cvt_nodes:
                for cvt_id in list(cvt_nodes)[:3]:  # Limit CVT expansion
                    rels = matcher.get_relations(cvt_id, "out")
                    props = []
                    for _, p, v in rels.get('out', [])[:5]:
                        if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."):
                            continue
                        prop_name = _simplify_relation_name(p)
                        props.append(f"{prop_name}: {v}")
                    if props:
                        lines.append(f"      [CVT {cvt_id}]: {', '.join(props[:3])}")
            
            if len(ends) > 5:
                lines.append(f"    ... (+{len(ends)-5} more)")
    
    # Summary: group by attribute value (useful for filtering)
    lines.append("")
    lines.append("-" * 10)
    lines.append("[Filter Summary]:")
    for end_val, candidates in sorted(end_to_candidates.items()):
        count = len(candidates)
        if count <= 3:
            cand_str = ", ".join(candidates)
        else:
            cand_str = ", ".join(candidates[:3]) + f"... (+{count-3})"
        lines.append(f"  {end_val} ({count}): {cand_str}")
    
    return "\n".join(lines)


def format_match_results(start_entity: str, paths: List[Dict], matcher, filtered_ends: set = None) -> Tuple[str, Dict]:
    """Optimized single-pass processing for match results formatting.
    
    Args:
        filtered_ends: Optional set of entities to include in output. If provided,
                       only these entities will appear in leaf entity lists.
    
    Returns:
        Tuple[str, Dict]: (formatted_text, structured_data_dict)
        structured_data_dict contains: leaf_entities, cvt_nodes, path_triples, logic_pattern
    """
    if not paths: 
        return "Status: No matches found.", {
            'leaf_entities': [],
            'cvt_nodes': {},
            'path_triples': [],
            'logic_pattern': '',
        }

    # --- A. Analyze Logic (from first path) ---
    first_path = paths[0]['triples']
    logic_triples = []
    # Collect all path triples for structured data
    all_path_triples = []

    
    curr_real = start_entity
    if first_path and first_path[0][0].lower() == start_entity.lower(): 
        curr_real = first_path[0][0]

    node_roles = {0: start_entity} 
    
    for i, (s, rel, o) in enumerate(first_path):
        if s == curr_real: 
            next_real, d = o, "OUT"
        else: 
            next_real, d = s, "IN"
        
        next_ph = "{End}" if i == len(first_path) - 1 else f"{{Node_{i+1}}}"
        node_roles[i+1] = next_ph
        
        # Use RDF triple format: (Subject: X, Predicate: Y, Object: Z)
        simple_rel = rel.split('.')[-1]  # Simplify relation for readability
        if d == "OUT":
            # Forward: Current -> Next => Current is Subject
            subj = f"({node_roles[i]})" if i == 0 else node_roles[i]
            obj = next_ph
            logic_triples.append(f"Step {i+1}: (Subject: {subj}, Predicate: {simple_rel}, Object: {obj})")
        else:
            # Reverse: Next -> Current => Next is Subject
            subj = next_ph
            obj = f"({node_roles[i]})" if i == 0 else node_roles[i]
            logic_triples.append(f"Step {i+1}: (Subject: {subj}, Predicate: {simple_rel}, Object: {obj})")
        
        curr_real = next_real

    # --- B. SINGLE PASS: Build tree, collect CVT nodes, endpoints, AND branches ---
    data_tree = {}
    # [FIX] Track native path leaves separately from CVT-expanded leaves
    found_ends = set()  # Native path endpoints (non-CVT)
    cvt_expanded_ends = set()  # Entities extracted from CVT properties
    cvt_leaf_map = {}  # {cvt_id: [leaf_entities]} for CVT-ending paths
    cvt_by_role = defaultdict(set)  # CVT nodes grouped by path role
    
    # [NEW] Branches list for structured data compression
    branches_raw = []  # Collect all branches first, sort later
    leaf_to_shortest_path = {}  # Track shortest path length for each leaf entity
    
    for p in paths:
        triples = p.get('triples', [])
        if not triples:
            continue
        
        # Collect triples for structured data
        for s, rel, o in triples:
            all_path_triples.append([s, rel, o])
            
        curr_tree = data_tree
        c_ent = start_entity
        if triples[0][0].lower() == start_entity.lower():
            c_ent = triples[0][0]
        
        last_node = None
        # [NEW] Build branch signature (relation sequence with directions)
        branch_signature_parts = []
        branch_triples = []
        
        for i, (s, rel, o) in enumerate(triples):
            if s == c_ent:
                next_ent, c_ent = o, o
                direction = "out"
            else:
                next_ent, c_ent = s, s
                direction = "in"
            
            last_node = next_ent
            role = node_roles.get(i+1, "{Unknown}")
            
            # [NEW] Build branch signature: "rel1:out|rel2:in|..."
            branch_signature_parts.append(f"{rel}:{direction}")
            branch_triples.append([s, rel, o])
            
            # Collect CVT nodes
            if matcher.is_cvt(next_ent):
                cvt_by_role[role].add(next_ent)
            
            # Build tree structure with relation-based key (NOT entity-based)
            # This allows paths with same relation structure to merge into one branch
            # [FIX] Use relation+direction for ALL nodes (including leaf hop)
            simple_rel = rel.split('.')[-1]  # Simplify relation for readability
            if direction == "out":
                key = f"-(out: {simple_rel})-> {role}"
            else:
                key = f"<-(in: {simple_rel})- {role}"
            
            # Always create subtree with key first
            if key not in curr_tree:
                curr_tree[key] = {}
            curr_tree = curr_tree[key]
            
            # [NEW] If intermediate node (not leaf), add entity layer
            # This allows branching by specific intermediate entity (both standard Entities and CVTs)
            if i < len(triples) - 1:
                # Use a distinct format for entity nodes
                ent_key = f"[{role}]: {next_ent}"
                if ent_key not in curr_tree:
                    curr_tree[ent_key] = {}
                curr_tree = curr_tree[ent_key]
            
            if i == len(triples) - 1:
                # Leaf node: append to _leaves list of current subtree
                if '_leaves' not in curr_tree:
                    curr_tree['_leaves'] = []
                curr_tree['_leaves'].append(next_ent)
        
        # Collect non-CVT endpoints
        if last_node and not matcher.is_cvt(last_node) and last_node not in IGNORED_ENTITY_STRINGS:
            found_ends.add(last_node)
            
            # [NEW] Build branch record for structured data
            path_length = len(triples)
            branch_signature = "|".join(branch_signature_parts)
            
            branches_raw.append({
                'path_length': path_length,
                'path_triples': branch_triples,
                'leaf_entity': last_node,
                'branch_signature': branch_signature,
            })
            
            # [NEW] Track shortest path for each leaf entity
            if last_node not in leaf_to_shortest_path or path_length < leaf_to_shortest_path[last_node]:
                leaf_to_shortest_path[last_node] = path_length
    
    # [NEW] Sort branches by path_length (shortest first) for compression priority
    branches_sorted = sorted(branches_raw, key=lambda b: b['path_length'])

    # --- C. Build output lines ---
    lines = [
        f"Execution Results for ({start_entity}):",
        "-" * 50,
        "[Logic Pattern]:",
        "(Format: Sequence of RDF Triples. Subject-Predicate-Object.)"
    ]
    lines.extend(f"  {lt}" for lt in logic_triples)
    lines.append("-" * 10)
    lines.append("[Data Topology]:")
    lines.append(f"  [Root] ({start_entity}):")
    
    # Render full topology - compression handled by frontend (plug_v10.py)
    # [FIX] Global deduplication - track displayed leaves across all branches
    displayed_leaves = set()
    
    # [MODIFIED] render_topology now accepts output_list and is deferred
    def render_topology(tree_dict, output_list, indent=2):
        prefix = "  " * indent
        sub_keys = sorted(k for k in tree_dict.keys() if k != '_leaves')
        
        truncated = len(sub_keys) > AdaptiveConfig.MAX_TOPOLOGY_NODES
        if truncated:
            sub_keys = sub_keys[:AdaptiveConfig.MAX_TOPOLOGY_NODES]
            
        for k in sub_keys:
            subtree = tree_dict[k]
            if '_leaves' in subtree:
                # [FIX] Show all leaves for this branch (removed deduplication to show full topology)
                branch_leaves = sorted(set(subtree['_leaves']))
                
                output_list.append(f"{prefix}- {k}:")
                
                if len(branch_leaves) == 1:
                    output_list.append(f"{prefix}    - {{End}} ({branch_leaves[0]})")
                else:
                    leaf_preview = branch_leaves[:5]
                    val_str = "[" + ", ".join(leaf_preview)
                    if len(branch_leaves) > 5:
                        val_str += f"...(+{len(branch_leaves)-5})]"
                    else:
                        val_str += "]"
                    output_list.append(f"{prefix}    - {{End}}: {val_str}")
            else:
                # Intermediate node - check if we can merge with single child
                child_keys = [ck for ck in subtree.keys() if ck != '_leaves']
                
                # [OPTIMIZE] If this relation node has exactly one entity child, merge them
                if len(child_keys) == 1 and child_keys[0].startswith('['):
                    # Single entity under this relation - merge the output
                    entity_key = child_keys[0]
                    # Extract entity name from "[{Node_1}]: EntityName" format
                    entity_subtree = subtree[entity_key]
                    
                    # Combine relation and entity: "-(out: rel)-> {Node_1} (Entity):"
                    merged_key = f"{k} ({entity_key.split(': ', 1)[1] if ': ' in entity_key else entity_key})"
                    output_list.append(f"{prefix}- {merged_key}:")
                    render_topology(entity_subtree, output_list, indent + 2)
                else:
                    # Multiple entities or no entity layer - render normally
                    output_list.append(f"{prefix}- {k}:")
                    render_topology(subtree, output_list, indent + 2)
        
        if truncated:
            output_list.append(f"{prefix}... (+ more branches)")

    # [DEFERRED] Do not call render_topology here.
    # We will call it after filtering, and insert it at 'topo_insert_idx'.
    topo_insert_idx = len(lines)
    
    lines.append("")

    # --- E. Extract entities from CVT nodes if found_ends is empty ---
    # (Moved from below - Must run before filtering)
    if not found_ends and cvt_by_role:
        for role, node_ids in cvt_by_role.items():
            for nid in list(node_ids)[:50]:  # Limit to avoid explosion
                rels = matcher.get_relations(nid, "out")
                cvt_leaves_for_node = []  # Track leaves for this CVT node
                for _, p, v in rels.get('out', []):
                    if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."):
                        continue
                    # Add non-CVT entities to cvt_expanded_ends (separate from found_ends)
                    if v and not matcher.is_cvt(v) and v not in IGNORED_ENTITY_STRINGS:
                        cvt_expanded_ends.add(v)
                        cvt_leaves_for_node.append(v)
                # [NEW] Build cvt_leaf_map for CVT-ending paths (enables branch_map polyfill)
                if cvt_leaves_for_node:
                    cvt_leaf_map[nid] = cvt_leaves_for_node

    # Combine for compatibility but keep tracking separate
    # Apply filtered_ends constraint if provided
    current_filtered_ends_ref = filtered_ends
    
    # [V17 FIX] Save original values BEFORE filtering for fallback check
    original_found_ends = found_ends.copy()
    original_cvt_expanded_ends = cvt_expanded_ends.copy()
    
    if current_filtered_ends_ref is not None:
        # Check intersection first
        temp_found = found_ends & current_filtered_ends_ref
        temp_cvt = cvt_expanded_ends & current_filtered_ends_ref
        
        # If intersection is empty but we had results, it means constraints failed.
        # Fallback to showing everything (user request: "return original results instead of pruning")
        if not temp_found and not temp_cvt and (original_found_ends or original_cvt_expanded_ends):
            lines.insert(0, "> [WARNING] Constraints were not satisfied by any path. Showing all results.")
            current_filtered_ends_ref = None
            # [V17 FIX] Restore original values - DO NOT overwrite with empty sets!
            found_ends = original_found_ends
            cvt_expanded_ends = original_cvt_expanded_ends
        else:
            found_ends = temp_found
            cvt_expanded_ends = temp_cvt
    
    all_leaf_ends = found_ends | cvt_expanded_ends

    # --- G. [NEW] Late Topology Pruning & Rendering ---
    # Now that all_leaf_ends is fully accurate (and filtered), we can prune the tree.
    
    # [NEW] Helper to collect valid CVTs from pruned tree
    def collect_tree_cvts(tree_dict):
        cvts = set()
        if not isinstance(tree_dict, dict): return cvts
        if '_leaves' in tree_dict:
            for l in tree_dict['_leaves']:
                if matcher.is_cvt(l): cvts.add(l)
        for k, v in tree_dict.items():
            if k == '_leaves': continue
            if k.startswith('['):
                parts = k.split(': ', 1)
                if len(parts) == 2 and matcher.is_cvt(parts[1]):
                    cvts.add(parts[1])
            cvts.update(collect_tree_cvts(v))
        return cvts

    if current_filtered_ends_ref is not None:
        def prune_tree(tree_dict):
            """Recurive prune."""
            if not isinstance(tree_dict, dict): return False
            has_valid = False
            
            if '_leaves' in tree_dict:
                leaves = tree_dict['_leaves']
                valid_l = []
                for l in leaves:
                    # Direct match or CVT match
                    if l in all_leaf_ends:
                        valid_l.append(l)
                    elif l in cvt_leaf_map and any(ce in all_leaf_ends for ce in cvt_leaf_map[l]):
                        valid_l.append(l)
                
                if valid_l:
                    tree_dict['_leaves'] = valid_l
                    has_valid = True
                else:
                    del tree_dict['_leaves']
            
            keys_to_rm = []
            for k in [x for x in tree_dict.keys() if x != '_leaves']:
                if prune_tree(tree_dict[k]):
                    has_valid = True
                else:
                    keys_to_rm.append(k)
            for k in keys_to_rm: del tree_dict[k]
            
            return has_valid

        # Prune root results
        root_keys_rm = []
        for k in [x for x in data_tree.keys() if x != '_leaves']:
            if not prune_tree(data_tree[k]):
                root_keys_rm.append(k)
        for k in root_keys_rm: del data_tree[k]

    # Determine visible CVTs (if filtering active)
    visible_cvts = None
    if current_filtered_ends_ref is not None:
        visible_cvts = collect_tree_cvts(data_tree)

    # Render Topology to temporary buffer
    topo_lines = []
    render_topology(data_tree, topo_lines)
    
    # Handle root leaves for single-hop
    if '_leaves' in data_tree:
        root_leaves = sorted(set(data_tree['_leaves']))
        # Logic to check against displayed_leaves (which is updated inside render_topology)
        new_leaves = [l for l in root_leaves if l not in displayed_leaves]
        displayed_leaves.update(new_leaves)
        if new_leaves:
            MAX_SINGLE = int(os.getenv('MAX_SINGLE_BRANCH_LEAVES', '30'))
            count = min(len(new_leaves), MAX_SINGLE)
            val = "[" + ", ".join(new_leaves[:count])
            val += f"...(+{len(new_leaves)-count})]" if len(new_leaves) > count else "]"
            topo_lines.append(f"    └─ {{End}}: {val}")
    
    # Insert Topology into main lines
    lines[topo_insert_idx:topo_insert_idx] = topo_lines

    # --- D. CVT Node Details (Moved & Pruned) ---
    # Collect structured CVT data for StructuredData
    cvt_structured_data = {}
    
    if cvt_by_role:
        det_lines = []
        det_lines.append("[Node Details]:")
        det_lines.append("  (Legend: '(prop: val)' means Node has property; '(<- prop: src)' means src points to Node)")
        
        has_any_details = False
        for role, node_ids in cvt_by_role.items():
            if not node_ids:
                continue
            
            nodes_to_show = list(node_ids)
            # [FIX] Filter irrelevant CVT nodes if pruning occurred
            if visible_cvts is not None:
                nodes_to_show = [n for n in nodes_to_show if n in visible_cvts]
            
            if not nodes_to_show:
                continue
                
            has_any_details = True
            det_lines.append(f"  {role}:")
            
            # Limit display count
            nodes_to_show = nodes_to_show[:AdaptiveConfig.MAX_DETAIL_NODES]
            
            node_data = {}
            prop_values_stats = defaultdict(list)

            for nid in nodes_to_show:
                rels = matcher.get_relations(nid, "both")
                node_props = defaultdict(list)
                
                # OUT Edges
                for _, p, v in rels.get('out', []):
                    if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."):
                        continue
                    node_props[_simplify_relation_name(p)].append(v)
                
                # IN Edges
                for s, p, _ in rels.get('in', []):
                    if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."):
                        continue
                    if s == start_entity:
                        continue
                    node_props[f"<- {_simplify_relation_name(p)}"].append(s)
                
                node_data[nid] = node_props
                # Store in structured data (convert defaultdict to dict)
                cvt_structured_data[nid] = {k: list(v) for k, v in node_props.items()}
                for prop, vals in node_props.items():
                    prop_values_stats[prop].append(tuple(sorted(vals)))

            # Identify Common Properties
            common_props = {}
            total_n = len(nodes_to_show)
            if total_n > 1:
                threshold_count = total_n * 0.6
                for prop, val_tuples in prop_values_stats.items():
                    if not val_tuples: continue
                    counts = Counter(val_tuples)
                    most_common_val_tuple, count = counts.most_common(1)[0]
                    if count >= threshold_count:
                        common_props[prop] = most_common_val_tuple

            # Render Common
            if common_props:
                det_lines.append("    (Shared by all nodes below):")
                comps = []
                for k, v_tuple in common_props.items():
                    val_disp = v_tuple[0] if len(v_tuple) == 1 else str(list(v_tuple))
                    comps.append(f"({k}: {val_disp})")
                det_lines.append(f"      {', '.join(comps)}")

            # Render Specific
            det_lines.append("    [Specific Properties]:")
            for nid, props in node_data.items():
                diff_parts = []
                for k, vals in props.items():
                    if k in common_props and tuple(sorted(vals)) == common_props[k]:
                        continue
                    if len(vals) == 1:
                        v_str = vals[0]
                    else:
                        preview = vals[:AdaptiveConfig.MAX_INLINE_LIST]
                        v_str = "[" + ", ".join(preview)
                        if len(vals) > AdaptiveConfig.MAX_INLINE_LIST: v_str += "...]"
                        else: v_str += "]"
                    diff_parts.append(f"({k}: {v_str})")
                
                if diff_parts:
                    det_lines.append(f"      - {nid}: {', '.join(diff_parts)}")
                else:
                    det_lines.append(f"      - {nid}")
        
        if has_any_details:
            lines.extend(det_lines)

    # --- F. Footer (Leaf Entities) ---
    lines.append("-" * 10)
    
    # Show native path leaves first (if any)
    if found_ends:
        native_list = sorted(found_ends)
        lines.append(f"Leaf Entities ({len(native_list)}):")
        if len(native_list) <= 20:
            lines.append(f"  {native_list}")
        else:
            lines.append("  [NOTE: Large result set. All entities listed below:]")
            for i in range(0, len(native_list), 10):
                batch = native_list[i:i+10]
                lines.append(f"  {batch}")
    
    # Show CVT-expanded leaves separately (if any)
    if cvt_expanded_ends:
        cvt_list = sorted(cvt_expanded_ends)
        lines.append(f"CVT-Expanded Entities ({len(cvt_list)}):")
        if len(cvt_list) <= 20:
            lines.append(f"  {cvt_list}")
        else:
            lines.append("  [NOTE: Large result set. All entities listed below:]")
            for i in range(0, len(cvt_list), 10):
                batch = cvt_list[i:i+10]
                lines.append(f"  {batch}")


    # Build logic pattern string
    logic_pattern_str = " -> ".join(logic_triples)
    
    # Build structured data dict
    structured_data_dict = {
        'leaf_entities': sorted(list(all_leaf_ends)),  # Combined for compatibility
        'native_leaves': sorted(list(found_ends)),  # [NEW] Native path leaves only
        'cvt_expanded_leaves': sorted(list(cvt_expanded_ends)),  # [NEW] CVT-expanded leaves
        'cvt_nodes': cvt_structured_data,
        'path_triples': all_path_triples,
        'logic_pattern': logic_pattern_str,
        # [NEW] Branches for compression - sorted by path_length (shortest first)
        'branches': branches_sorted,
        # [NEW] CVT-to-leaf map for CVT-ending paths (enables branch_map polyfill)
        'cvt_leaf_map': cvt_leaf_map,
        # [NEW] Leaf entity to shortest path length mapping
        'leaf_to_shortest_path': leaf_to_shortest_path,
        # [NEW] Full data tree for multi-hop topology reconstruction
        'data_tree': data_tree,
    }
    
    return "\n".join(lines), structured_data_dict
# ==============================================================================
# 6. Core Logic
# ==============================================================================

class KGMatcher:
    def __init__(self, entities: List[str]):
        self.entities = list(dict.fromkeys(entities))
        self.exact_map = {self._normalize(e): e for e in self.entities}
        for e in self.entities: self.exact_map[e.lower().strip()] = e
        self.index = defaultdict(list)
        for idx, entity in enumerate(self.entities):
            for word in re.findall(r'\w+', entity.lower()):
                self.index[word].append(idx)
                if len(word) > 3: self.index[word[:3]].append(idx)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.sub(r'[^\w\s]', '', text).lower().split())

    @lru_cache(maxsize=10000)
    def find_entities(self, query: str, limit: int = 10) -> List[Dict]:
        norm = self._normalize(query)
        # Collect ALL candidates (including exact match) — never short-circuit
        candidates_idx = set()
        for word in re.findall(r'\w+', query.lower()):
            if word in self.index: candidates_idx.update(self.index[word])
            elif len(word) >= 3 and word[:3] in self.index: candidates_idx.update(self.index[word[:3]])
        results = []
        for idx in candidates_idx:
            orig = self.entities[idx]
            norm_orig = self._normalize(orig)
            score = difflib.SequenceMatcher(None, norm, norm_orig).ratio() * 100
            if norm in norm_orig or norm_orig in norm: score = max(score, 70)
            results.append({'entity': orig, 'score': score})
        results.sort(key=lambda x: (-x['score'], len(x['entity'])))
        # Hard cap: return at most limit results, minimum score threshold 50
        results = [r for r in results if r['score'] >= 50]
        return results[:limit]

class GraphPatternMatcher:
    def __init__(self, graph_obj, id2entity, entity2id, id2relation, relation2id, entity_matcher, cvt_id_set: Set[str] = None):
        self.graph = graph_obj
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.relation2id = relation2id
        self.entity_matcher = entity_matcher
        self.cvt_id_set = cvt_id_set if cvt_id_set else set()
        self.ep_relation_id = self.graph.edge_properties["relation_id"]
        self.name2ids = defaultdict(list)
        self.lower_name2ids = defaultdict(list)
        for idx, name in self.id2entity.items():
            self.name2ids[name].append(idx)
            self.lower_name2ids[KGMatcher._normalize(name)].append(idx)
        
        # [OPTIMIZATION] Precompute neighbor cache for fast lookups
        self._neighbor_cache = {}

    def is_cvt(self, entity_name: str) -> bool:
        """Check if an entity is a CVT node using Ground Truth set.
        
        Logic:
        - If cvt_id_set is populated (normal case), use it as Ground Truth.
        - Only fallback to regex if cvt_id_set is empty (legacy data compatibility).
        """
        if self.cvt_id_set:
            # Ground Truth available: use set membership only
            return entity_name in self.cvt_id_set
        else:
            # Fallback to regex for legacy data without cvt_id_set
            if not entity_name: return False
            return bool(re.match(r'^[mg]\.[a-zA-Z0-9_]+$', str(entity_name)))

    def _resolve_anchor(self, entity_name: str, hint_relation: str = None) -> Optional[int]:
        norm = KGMatcher._normalize(entity_name)
        candidates = self.lower_name2ids.get(norm, [])
        if not candidates:
            candidates = self.name2ids.get(entity_name, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        best, max_score = candidates[0], -1
        target_rel_id = self.relation2id.get(hint_relation) if hint_relation else None
        for nid in candidates:
            v = self.graph.vertex(nid)
            score = v.out_degree() + v.in_degree()
            if target_rel_id is not None:
                for e in v.all_edges():
                    if self.ep_relation_id[e] == target_rel_id:
                        score += 1000
                        break
            if score > max_score:
                max_score = score
                best = nid
        return best

    def get_relations(self, entity: str, direction: str) -> Dict:
        """Get relations with caching for repeated lookups."""
        nid = self._resolve_anchor(entity)
        if nid is None:
            return {}
        
        # Check cache first
        cache_key = (nid, direction)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        
        v = self.graph.vertex(nid)
        res = {}
        
        if direction in ('out', 'both'):
            out_edges = []
            for e in v.out_edges():
                out_edges.append((
                    self.id2entity[int(e.source())],
                    self.id2relation[self.ep_relation_id[e]],
                    self.id2entity[int(e.target())]
                ))
            res['out'] = sorted(out_edges, key=lambda x: x[1])
        
        if direction in ('in', 'both'):
            in_edges = []
            for e in v.in_edges():
                in_edges.append((
                    self.id2entity[int(e.source())],
                    self.id2relation[self.ep_relation_id[e]],
                    self.id2entity[int(e.target())]
                ))
            res['in'] = sorted(in_edges, key=lambda x: x[1])
        
        # Cache the result
        self._neighbor_cache[cache_key] = res
        return res

    def expand_node_v2(self, node_name: str) -> List[str]:
        rels = self.get_relations(node_name, "both")
        expanded = set()
        if 'out' in rels:
            for s, p, o in rels['out']:
                if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."): continue
                if o not in IGNORED_ENTITY_STRINGS: expanded.add(o)
        # [CWQ FIX] Some value/CVT-like nodes only have meaningful incoming edges.
        # Prefer outgoing expansion first; only fall back to incoming neighbors when
        # no outgoing human-readable values exist.
        if not expanded and 'in' in rels:
            for s, p, o in rels['in']:
                if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."): continue
                if s not in IGNORED_ENTITY_STRINGS: expanded.add(s)
        return list(expanded)

    def find_entities(self, query, limit=5):
        return self.entity_matcher.find_entities(query, limit)

    def find_similar_relations(self, query: str, limit: int = 5) -> List[Dict]:
        # 使用 difflib 替代 thefuzz，无需额外安装库
        all_rels = list(self.relation2id.keys())
        # get_close_matches 返回最相似的字符串列表
        matches = difflib.get_close_matches(query, all_rels, n=limit, cutoff=0.4)
        
        results = []
        for m in matches:
            # 计算相似度分数 (0-100)
            ratio = difflib.SequenceMatcher(None, query, m).ratio()
            results.append({"relation": m, "score": int(ratio * 100)})
        
        # 按分数降序排列
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def explore_schema(self, pattern: str) -> List[str]:
        clean = pattern.replace("*", "").strip().rstrip('.')
        prefix = clean + "."
        return sorted([r for r in self.relation2id.keys() if r.startswith(prefix) or r == clean])

    def find_paths_with_relation(self, start: str, relation: str, max_hops: int, limit: int) -> List[Dict]:
        sid = self._resolve_anchor(start, hint_relation=relation)
        if sid is None or relation not in self.relation2id: return []
        tid = self.relation2id[relation]
        queue = deque([(sid, [])])
        visited = {sid}
        collected = []
        early_stop = os.getenv("KGQA_PATH_EARLY_STOP", "1").strip().lower() in {"1", "true", "yes", "on"}
        use_global_visited = os.getenv("KGQA_PATH_GLOBAL_VISITED", "1").strip().lower() in {"1", "true", "yes", "on"}
        for _ in range(max_hops):
            level_found = False
            for _ in range(len(queue)):
                curr, path = queue.popleft()
                v = self.graph.vertex(curr)
                path_ids = {int(e.source()) for e in path} | {int(e.target()) for e in path}
                for e in v.all_edges():
                    rid = self.ep_relation_id[e]
                    neighbor = int(e.target() if e.source() == v else e.source())
                    if neighbor in path_ids: continue
                    new_path = path + [e]
                    if rid == tid:
                        # [V20] Allow both directions (Logic Instance handles semantic validation)
                        collected.append({"triples": self._edges_to_triples(new_path)})
                        level_found = True
                    elif not level_found or not early_stop:
                        if use_global_visited:
                            if neighbor in visited:
                                continue
                            visited.add(neighbor)
                        queue.append((neighbor, new_path))
            if (level_found and early_stop) or len(collected) >= limit:
                break
        return collected[:limit]
    
    def execute_match_pattern(self, start: str, steps: List[Dict]) -> Tuple[List[str], List[Dict]]:
        if not steps: return [], []
        sid = self._resolve_anchor(start, hint_relation=steps[0]['relation'])
        if sid is None: return [], []
        queue = deque([(sid, 0, [])])
        final_paths = []
        final_entities = set()
        seen_sigs = set()
        while queue:
            curr, idx, path = queue.popleft()
            if idx >= len(steps):
                ename = self.id2entity[curr]
                sig = str(path)
                if sig not in seen_sigs:
                    final_paths.append({"triples": path})
                    final_entities.add(ename)
                    seen_sigs.add(sig)
                continue
            rule = steps[idx]
            rel_id = self.relation2id.get(rule['relation'])
            if rel_id is None: continue
            v = self.graph.vertex(curr)
            direction = rule['direction']
            edges = v.out_edges() if direction == 'out' else (v.in_edges() if direction == 'in' else v.all_edges())
            for e in edges:
                if self.ep_relation_id[e] == rel_id:
                    nxt = int(e.target()) if direction == 'out' else int(e.source())
                    if direction == 'both': nxt = int(e.target()) if int(e.source()) == curr else int(e.source())
                    t = [self.id2entity[int(e.source())], self.id2relation[rel_id], self.id2entity[int(e.target())]]
                    queue.append((nxt, idx+1, path + [t]))
        return sorted(list(final_entities)), final_paths

    def _edges_to_triples(self, edges):
        return [[self.id2entity[int(e.source())], self.id2relation[self.ep_relation_id[e]], self.id2entity[int(e.target())]] for e in edges]

class DataManager:
    def __init__(self): self.matchers = {}; self.samples = {}
    def load_data(self, base, ds, sp):
        path = os.path.join(base, 'data', 'cwq_processed', f'{sp}.pkl') if ds == 'cwq' else os.path.join(base, 'retrieve', 'data_files', ds, 'processed', f'{sp}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                for s in pickle.load(f):
                    if 'id' in s:
                        g = gt.Graph(directed=True)
                        ents = s.get('text_entity_list', []) + s.get('non_text_entity_list', [])
                        g.add_vertex(len(ents))
                        if s.get('h_id_list'):
                            g.add_edge_list(zip(s['h_id_list'], s['t_id_list'], range(len(s['h_id_list'])), s['r_id_list']), eprops=[('triple_id', 'int'), ('relation_id', 'int')])
                        s['graph_obj'] = g
                        self.samples[s['id']] = s
    def get_matcher(self, sid):
        if sid in self.matchers: return self.matchers[sid]
        if sid not in self.samples: return None
        s = self.samples[sid]
        ents = s['text_entity_list'] + s['non_text_entity_list']
        rels = s['relation_list']
        cvt_set = set(s.get('non_text_entity_list', []))
        m = GraphPatternMatcher(s['graph_obj'], {i:e for i,e in enumerate(ents)}, {e:i for i,e in enumerate(ents)}, {i:r for i,r in enumerate(rels)}, {r:i for i,r in enumerate(rels)}, KGMatcher(ents), cvt_id_set=cvt_set)
        self.matchers[sid] = m
        return m

def collect_expanded_entities(matcher: GraphPatternMatcher, start_entity: str, paths: List[Dict]) -> List[str]:
    final_set = set()
    for p in paths:
        triples = p.get('triples', [])
        if not triples: continue
        curr = start_entity
        if triples and triples[0][0].lower() == start_entity.lower(): curr = triples[0][0]
        
        last_node = None
        for s, _, o in triples:
            if s == curr: last_node, curr = o, o
            else: last_node, curr = s, s
        
        if last_node:
            if last_node in IGNORED_ENTITY_STRINGS: continue
            if matcher.is_cvt(last_node):
                final_set.update(matcher.expand_node_v2(last_node))
            else:
                final_set.add(last_node)
    return sorted(list(final_set))

manager = DataManager()
app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
def startup():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager.load_data(root, os.getenv("DATASET_NAME", "webqsp"), os.getenv("SPLIT", "test"))

def format_entity_candidates_with_context(matcher, candidates: List[Dict]) -> List[str]:
    """
    Format entity candidates with adaptive context for disambiguation.
    extacted from api_find_entities for reuse.
    """
    if not candidates: return []
    
    # 1. Extract all relations for candidates
    entity_relations = [] # [{'name': str, 'rels': {rel: {vals}}}]
    all_seen_rels = set()
    
    for item in candidates:
        ent = item['entity']
        neighbors = matcher.get_relations(ent, 'out')
        rel_map = defaultdict(set)
        
        # Collect values (object) for each relation
        out_trips = neighbors.get('out', [])
        for s, r, o in out_trips:
            if len(o) < 50: # Skip long text
                rel_map[r].add(o)
                all_seen_rels.add(r)
        
        entity_relations.append({
            'name': ent,
            'rels': rel_map
        })
    
    # 2. Compute Discriminative Score
    rel_scores = {}
    for r in all_seen_rels:
        values_per_entity = []
        for er in entity_relations:
            values_per_entity.append(er['rels'].get(r, set()))
        
        non_empty = [v for v in values_per_entity if v]
        if not non_empty:
            score = 0.0
        elif len(non_empty) == 1:
            score = 10.0 # Only one entity -> Distinguishing
        else:
            total_diff = 0.0
            comparisons = 0
            vals_list = list(non_empty)
            for i in range(len(vals_list)):
                for j in range(i+1, len(vals_list)):
                    s1, s2 = vals_list[i], vals_list[j]
                    union = len(s1 | s2)
                    inter = len(s1 & s2)
                    if union > 0:
                        total_diff += (1.0 - inter/union)
                    comparisons += 1
            
            score = (total_diff / comparisons) if comparisons > 0 else 0.0
            # Bonus for 'type' relations
            if 'type' in r or 'category' in r:
                score += 2.0
        
        rel_scores[r] = score

    # 3. Generate Summary
    top_discriminators = sorted(rel_scores.items(), key=lambda x: x[1], reverse=True)
    top_rel_names = [x[0] for x in top_discriminators] # Full list sorted
    
    cand_lines = []
    for er in entity_relations:
        context_parts = []
        shown_count = 0
        
        for r in top_rel_names:
            if r in er['rels']:
                vals = list(er['rels'][r])[:3]
                val_str = ", ".join(vals)
                if len(er['rels'][r]) > 3: val_str += "..."
                context_parts.append(f"{r}: {val_str}")
                shown_count += 1
            
            if shown_count >= 3: break
        
        context_str = " | ".join(context_parts)
        line = f"{er['name']}"
        if context_str:
            line += f" [Context: {context_str}]"
        cand_lines.append(line)
        
    return cand_lines

@app.post("/v2/find_entities", response_model=FinalResponse)
async def api_find_entities(req: EntitiesRequest):
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            raise Exception("Sample ID not found")
        
        # Run blocking operation in thread pool
        res = await asyncio.to_thread(m.find_entities, req.entity_substring, req.limit)
        
        # [V12 USER REQ 1] Silence failure
        if not res:
            return FinalResponse(
                success=True, 
                status="KG_SUCCESS", 
                response_text="", 
                found_end_entities=[]
            )

        # [V12 USER REQ 2 - ADAPTIVE INFO GAIN]
        candidates = res[:10]
        
        cand_lines = await asyncio.to_thread(format_entity_candidates_with_context, m, candidates)
        
        # [V12 FIX] Always show ALL candidates - even 100% match could be wrong entity type
        # Unified format: no separate "matched" vs "candidates" - let model decide based on context
        txt = (
            f"Entity Candidates for '{req.entity_substring}' (Choose based on context):\n" +
            "\n".join([f"  - {line}" for line in cand_lines])
        )
        
        # Add guidance for model
        txt += "\n\n⚠️ NOTE: Only 100% string match counts as the same entity. If unsure, use explore_schema to check aliases."
        
        # Return top match in found_end_entities for state tracking, but text shows all
        top_match = res[0]
        return FinalResponse(
            success=True, 
            status="KG_SUCCESS", 
            response_text=txt, 
            # [V12 FIX] Return ALL found candidates. Let the model choose the correct one.
            # Filtering by strict score caused valid entities (e.g. partial matches) to be ignored by state tracking.
            found_end_entities=[r['entity'] for r in res]
        )

    except Exception as e: 
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))

@app.post("/v2/explore_schema", response_model=FinalResponse)
async def api_explore_schema(req: ExploreSchemaRequest):
    try:
        m = manager.get_matcher(req.sample_id)
        rels = await asyncio.to_thread(m.explore_schema, req.pattern)
        txt = format_schema_hierarchical(req.pattern, rels)
        
        # [NEW] Return structured relation list for hallucination validation
        return FinalResponse(
            success=True, 
            status="KG_SUCCESS", 
            response_text=txt,
            found_end_entities=rels  # Relation list as structured data
        )
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))


# ── Semantic Retrieval via GTE ──────────────────────────────────────────

import numpy as np

def _gte_encode_batch(gte_url: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Call GTE embedding service to encode a batch of texts. Returns (N, dim) array."""
    import aiohttp
    # This is a sync wrapper — call from asyncio.to_thread
    import requests
    resp = requests.post(f"{gte_url}/embed", json={"texts": texts, "batch_size": batch_size}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"], dtype=np.float32)


@app.post("/v2/semantic_retrieve", response_model=SemanticRetrieveResponse)
async def api_semantic_retrieve(req: SemanticRetrieveRequest):
    """Retrieve top-k entities and relations per query using GTE embedding similarity.

    Directly accesses the full subgraph (all entities, all relations) for the given
    sample_id — no check_entities or explore_schema needed.
    """
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            return SemanticRetrieveResponse(success=False, total_entities=0, total_relations=0)

        # 1. Get ALL entities and ALL relations from the subgraph
        all_entities = list(set(m.id2entity.values()))
        # Filter out CVT nodes (m.0xxx, g.xxx) — they are not human-readable
        all_entities = [e for e in all_entities if not _is_cvt_id(e) and len(e) > 1]
        all_relations = list(m.relation2id.keys())

        if not all_entities and not all_relations:
            return SemanticRetrieveResponse(success=True, total_entities=0, total_relations=0)

        # 2. Encode all entities, all relations, and all queries via GTE
        all_texts_to_encode = req.queries + all_entities + all_relations
        all_embs = await asyncio.to_thread(_gte_encode_batch, req.gte_url, all_texts_to_encode)

        n_queries = len(req.queries)
        q_embs = all_embs[:n_queries]                          # (Q, dim)
        ent_embs = all_embs[n_queries:n_queries + len(all_entities)]  # (E, dim)
        rel_embs = all_embs[n_queries + len(all_entities):]    # (R, dim)

        # Normalize for cosine similarity
        import torch.nn.functional as F
        q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)
        ent_embs = ent_embs / (np.linalg.norm(ent_embs, axis=1, keepdims=True) + 1e-8)
        rel_embs = rel_embs / (np.linalg.norm(rel_embs, axis=1, keepdims=True) + 1e-8)

        # 3. Compute similarity and rank per query
        entities_per_query = []
        relations_per_query = []

        for i in range(n_queries):
            q = q_embs[i:i+1]  # (1, dim)

            # Entity ranking
            ent_scores = (q @ ent_embs.T)[0]  # (E,)
            top_k_ent = min(req.top_k, len(all_entities))
            ent_top_idx = np.argsort(ent_scores)[::-1][:top_k_ent]
            entities_per_query.append([
                {"candidate": all_entities[idx], "score": float(ent_scores[idx])}
                for idx in ent_top_idx
            ])

            # Relation ranking
            rel_scores = (q @ rel_embs.T)[0]  # (R,)
            top_k_rel = min(req.top_k, len(all_relations))
            rel_top_idx = np.argsort(rel_scores)[::-1][:top_k_rel]
            relations_per_query.append([
                {"candidate": all_relations[idx], "score": float(rel_scores[idx])}
                for idx in rel_top_idx
            ])

        return SemanticRetrieveResponse(
            success=True,
            entities_per_query=entities_per_query,
            relations_per_query=relations_per_query,
            total_entities=len(all_entities),
            total_relations=len(all_relations),
        )

    except Exception as e:
        logging.error(f"semantic_retrieve error: {e}")
        return SemanticRetrieveResponse(success=False, total_entities=0, total_relations=0)


def _struct_logical_paths(start_entity: str, paths: List[Dict],
                          constraint_relations: List[str] = None, 
                          constraint_entities: List[str] = None) -> List[Dict]:
    """
    Generate structured logical paths (RDF + Action) for API response.
    Returns a list of deduplicated path objects.
    """
    if not paths: return []

    grouped = defaultdict(list)
    pattern_samples = {}

    for p in paths:
        triples = p.get('triples', [])
        if not triples: continue
        
        logic_steps = []
        if triples and triples[0][0].lower() == start_entity.lower(): 
            curr = triples[0][0]
        else:
            curr = start_entity
            
        for s, rel, o in triples:
            if s == curr: 
                direction, curr = "OUT", o
            else: 
                direction, curr = "IN", s
            logic_steps.append((rel, direction))
            
        signature = tuple(logic_steps)
        grouped[signature].append(p)
        if signature not in pattern_samples: 
            pattern_samples[signature] = triples

    sorted_patterns = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
    
    structured_results = []
    
    for idx, (sig, matches) in enumerate(sorted_patterns):
        count = len(matches)
        raw_triples = pattern_samples[sig]
        
        # Build RDF steps
        rdf_steps = []
        curr_node_var = f"({start_entity})" 
        
        for i, (rel, direction) in enumerate(sig):
            if i == len(sig) - 1: 
                next_node_var = "{Candidate}"      
            else:
                next_node_var = f"{{Intermediate_{i+1}}}" 
            
            # Simplify relation
            simple_rel = rel.split('.')[-1]
            
            if direction == "OUT":
                subj, obj = curr_node_var, next_node_var
            else:
                subj, obj = next_node_var, curr_node_var
            
            rdf_steps.append(f"Step {i+1}: (Subject: {subj}, Predicate: {simple_rel}, Object: {obj})")
            curr_node_var = next_node_var
            
        # Action Hint
        hint = _build_match_pattern_code(start_entity, raw_triples,
                                         select_relations=constraint_relations,
                                         select_entities=constraint_entities)
        
        # Example (Natural Language)
        example_str = ""
        if raw_triples:
            nl_parts = []
            for s, rel, o in raw_triples:
                s_name = s if isinstance(s, str) else str(s)
                o_name = o if isinstance(o, str) else str(o)
                # Simplify relation
                simple_rel = rel.split('.')[-1]
                nl_parts.append(f"{s_name}'s [{simple_rel}] is {o_name}")
            if nl_parts:
                example_str = "; ".join(nl_parts) + "."
        
        structured_results.append({
            "index": idx + 1,
            "count": count,
            "rdf_steps": rdf_steps,
            "action_code": hint,
            "example": example_str,
            "signature": [{"relation": rel, "direction": dir} for rel, dir in sig],
            # [NEW] Fields for plug_v10.py compatibility
            "start_entity": start_entity,  # From function parameter
            "steps": [{"relation": rel, "direction": dir.lower()} for rel, dir in sig],  # Alias for signature with lowercase direction
        })
        
    return structured_results

def _sync_find_logical_path(m, req):
    """Synchronous helper for find_logical_path to run in thread pool."""
    errors = []
    valid_start = None
    
    entity_res = m.find_entities(req.start_entity, limit=5)
    
    if entity_res and entity_res[0]['score'] >= 100:
        valid_start = entity_res[0]['entity']
    else:
        # [V13 FIX] Use rich candidate formatting with context
        cand_lines = format_entity_candidates_with_context(m, entity_res)
        cands = "\n".join([f"  - {line}" for line in cand_lines]) if cand_lines else "(No candidates)"
        errors.append(f"❌ Start Entity '{req.start_entity}' not found (Exact match required).\nDid you mean:\n{cands}")

    rel_exists = req.contains_relation in m.relation2id
    if not rel_exists:
        sims = m.find_similar_relations(req.contains_relation, limit=5)
        cands = "\n".join([f"  - {x['relation']}" for x in sims]) if sims else "(No candidates)"
        errors.append(f"❌ Relation '{req.contains_relation}' not found.\nTop 5 Candidates:\n{cands}")

    if errors:
        status = "KG_BOTH_ERROR" if len(errors) == 2 else ("KG_ENTITY_ERROR" if "Entity" in errors[0] else "KG_RELATION_ERROR")
        return FinalResponse(success=True, status=status, response_text="[DIAGNOSTIC REPORT]\n" + "\n".join(errors))

    paths = m.find_paths_with_relation(valid_start, req.contains_relation, req.max_hops, req.limit)
    
    if not paths:
        return FinalResponse(success=True, status="KG_NO_PATHS", response_text=f"No logical paths found between ({valid_start}) containing [{req.contains_relation}].")
    
    txt = format_logical_paths(valid_start, paths)
    
    # [NEW] Use structured helper for action_hints
    structured_hints = _struct_logical_paths(valid_start, paths)
    
    return FinalResponse(
        success=True, 
        status="KG_SUCCESS", 
        response_text=txt, 
        action_hints=structured_hints,
        structured_data=StructuredData(start_entity=valid_start)
    )


@app.post("/v2/find_logical_path_with_relation", response_model=FinalResponse)
async def api_find_logical_path(req: PathRequest):
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            return FinalResponse(success=False, status="KG_ERROR", response_text="Sample ID not found")
            
        return await asyncio.to_thread(_sync_find_logical_path, m, req)
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))


class PlanRequest(BaseModel):
    sample_id: str = "default"
    question: str = Field(..., description="Intent description (replaces subquestion)")
    # [V21] Allow anchor to be a list (max 3) for multi-anchor exploration
    anchor: Union[str, List[str]] = Field(..., description="Verified entities to anchor the plan on (can be single or list)")
    related: List[str] = Field(..., description="Core relations with high confidence")
    maybe_related: List[str] = Field(default_factory=list, description="Hesitant relations")
    constraint_relations: List[str] = Field(default_factory=list, description="Attribute relations for leaf filtering")
    constraint_entities: List[str] = Field(default_factory=list, description="Entities for connectivity filtering")
    # [V19] New optional params for hop and limit control (RL Configurable)
    max_hops: int = Field(default_factory=lambda: int(os.getenv("MAX_HOPS", "2")), description="Maximum hops for path finding")
    path_limit: int = Field(default_factory=lambda: int(os.getenv("PATH_LIMIT", "3")), description="Maximum number of shortest paths to return")

@app.post("/v2/plan", response_model=FinalResponse)
@app.post("/v2/plan_subquestion", response_model=FinalResponse)
async def api_plan_subquestion(req: PlanRequest):
    """
    Backend implementation of plan_subquestion.
    Auto-expands to find_logical_path and aggregates results.
    Includes TYPE CHECKING for entities in relation fields.
    """
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            return FinalResponse(success=False, status="KG_ERROR", response_text="Sample ID not found")

        # 1. Validation & input normalization
        if not req.anchor:
            return FinalResponse(success=True, status="KG_ERROR", response_text="plan_subquestion error: anchor is empty")
        
        all_relations = []
        # Normalization helper
        def collect_rels(source_list):
            for item in source_list:
                if isinstance(item, dict): all_relations.append(item.get('relation', ''))
                elif isinstance(item, str): all_relations.append(item)
                
        collect_rels(req.related)
        collect_rels(req.maybe_related[:3]) # Limit maybe_related
        
        if not all_relations:
             return FinalResponse(success=True, status="KG_ERROR", response_text="plan_subquestion error: both related and maybe_related are empty")

        # [CRITICAL] Heuristic Check for Entity-in-Relation (related/maybe_related)
        for rel in all_relations:
            if '.' not in rel and (' ' in rel or rel[0].isupper()):
                return FinalResponse(
                    success=True, 
                    status="KG_TYPE_ERROR",
                    response_text="", 
                    # Return error via action_hints to leverage existing frontend diagnostic parsing
                    action_hints=[{
                        "error_type": "TYPE_MISMATCH",
                        "value": rel,
                        "source_field": "related/maybe_related",
                        "suggestion": "Did you mean to put this in 'anchor'?",
                        "error": "This looks like an Entity Name, but it was found in a Relation field."
                    }]
                )

        # [NEW] Validate Constraint Relations
        for rel in req.constraint_relations:
            # Type check
            if '.' not in rel and (' ' in rel or rel[0].isupper()):
                 return FinalResponse(
                    success=True, status="KG_TYPE_ERROR", response_text="", 
                    action_hints=[{"error_type": "TYPE_MISMATCH", "value": rel, "source_field": "constraint_relations", "suggestion": "Did you mean to put this in 'constraint_entities'?", "error": "This looks like an Entity Name."}]
                )
            # Existence check
            if rel not in m.relation2id:
                sims = m.find_similar_relations(rel, limit=5)
                cands = "\n".join([f"  - {x['relation']}" for x in sims]) if sims else "(No candidates)"
                return FinalResponse(
                    success=True, status="KG_RELATION_ERROR", 
                    response_text=f"Constraint Relation '{rel}' not found.\nDid you mean:\n{cands}"
                )

        # [NEW] Validate Constraint Entities
        for ent in req.constraint_entities:
            res = await asyncio.to_thread(m.find_entities, ent, 3)
            if not res or res[0]['score'] < 100:
                # Construct candidates
                cand_lines = await asyncio.to_thread(format_entity_candidates_with_context, m, res)
                cands = "\n".join([f"  - {line}" for line in cand_lines])
                return FinalResponse(
                    success=True, status="KG_ENTITY_ERROR", 
                    response_text=f"Constraint Entity '{ent}' not found (Unverified).\nDid you mean:\n{cands}"
                )
        
        # 2. Build tasks
        tasks = []
        queries = []
        # [V21] Support multiple anchors (max 3)
        anchors = [req.anchor] if isinstance(req.anchor, str) else req.anchor[:3]
        
        for entity in anchors:
            for rel in all_relations[:5]:    # Limit 5 relations per anchor
                sub_req = PathRequest(
                    sample_id=req.sample_id,
                    start_entity=entity,
                    contains_relation=rel,
                    # [V19] Use dynamic hop count and limit
                    max_hops=req.max_hops,
                    limit=req.path_limit
                )
                queries.append({'entity': entity, 'rel': rel})
                tasks.append(asyncio.to_thread(_sync_find_logical_path, m, sub_req))
        
        if not tasks:
            return FinalResponse(success=True, status="KG_NO_PATHS", response_text="plan_subquestion: No valid (entity, relation) pairs to query")

        # 3. Execute Parallel
        results = await asyncio.gather(*tasks)
        
        # [V13 FIX] Update action hints with constraints BEFORE generating text display
        if req.constraint_relations or req.constraint_entities:
            for r in results:
                if r.status == 'KG_SUCCESS' and r.action_hints:
                    for hint in r.action_hints:
                        hint['constraint_relations'] = req.constraint_relations
                        hint['constraint_entities'] = req.constraint_entities
                        # Regenerate action_code with select parameters
                        anchor = hint.get('start_entity', req.anchor if isinstance(req.anchor, str) else (req.anchor[0] if req.anchor else ''))
                        steps = hint.get('steps', [])
                        path_str = ", ".join([f'{{"relation": "{s["relation"]}", "direction": "{s["direction"]}"}}' for s in steps])
                        select_rel_str = f', constraint_relations={req.constraint_relations}' if req.constraint_relations else ''
                        select_ent_str = f', constraint_entities={req.constraint_entities}' if req.constraint_entities else ''
                        hint['action_code'] = f'match_pattern(anchor="{anchor}", path=[{path_str}]{select_rel_str}{select_ent_str})'
        
        # 4. Aggregate
        all_entities = []
        collected_action_hints = []
        lines = [f"[Subquestion]: {req.question}"]
        lines.append(f"[Anchor]: {req.anchor}")
        lines.append(f"[Relations]: {', '.join(all_relations)}")
        lines.append("")
        lines.append("[Action Space Generated]:")
        
        has_any_path = False
        for q, r in zip(queries, results):
            if r.status == 'KG_SUCCESS':
                has_any_path = True
                all_entities.extend(r.found_end_entities)
                if r.action_hints:
                    collected_action_hints.extend(r.action_hints)
                    
                    # [FORMATTING] Show hints grouped by source relation
                    lines.append("")
                    lines.append(f"► From Relation: \"{q['rel']}\" (Anchor: {q['entity']})")
                    
                    for i, hint in enumerate(r.action_hints, 1):
                        lines.append(f"  {i}. Action: {hint.get('action_code', 'match_pattern(...)')}")
                        lines.append(f"     Logic Pattern:")
                        # rdf_steps is a list of strings like "Step 1: (...)"
                        rdf_steps = hint.get('rdf_steps', [])
                        for step in rdf_steps:
                             lines.append(f"       {step}")
                        
                        # [FORMATTING] Add Analogical Example
                        example = hint.get('example', '')
                        if example:
                            lines.append(f"     Analogical Example:")
                            lines.append(f"       {example}")
                else:
                    # Fallback if no structured hints but success (should not happen with new logic)
                    lines.append("")
                    lines.append(f"► From Relation: \"{q['rel']}\" (Anchor: {q['entity']})")
                    lines.append("  (Path found but no structured hints available)")

            # [CRITICAL] If any sub-query returned a DIAGNOSTIC REPORT
            elif 'DIAGNOSTIC REPORT' in r.response_text:
                 lines.append("")
                 lines.append(f"► From Relation: \"{q['rel']}\" (Anchor: {q['entity']})")
                 lines.append(f"  (Check failed: {r.status} - {r.response_text.splitlines()[0]})")

        # 5. Constraints Warning
        if req.constraint_relations or req.constraint_entities:
            lines.append("")
            constraint_strs = []
            if req.constraint_relations:
                constraint_strs.extend(req.constraint_relations)
            if req.constraint_entities:
                constraint_strs.extend([f"(entity: {e})" for e in req.constraint_entities])
            lines.append(f"[Planned Constraints]: {', '.join(constraint_strs)}")
            lines.append("→ Stage 5 will auto-execute constraint queries for filtering.")

        final_status = 'KG_SUCCESS' if has_any_path else 'KG_NO_PATHS'
        

        
        return FinalResponse(
            success=True,
            status=final_status,
            response_text='\n'.join(lines),
            found_end_entities=list(set(all_entities)),
            action_hints=collected_action_hints
        )
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=f"plan_subquestion failed: {str(e)}")





# ==============================================================================
# SELECT FEATURE HELPERS
# ==============================================================================

def _fetch_relation_value(m, entity: str, relation: str, max_hops: int = 2) -> List[str]:
    """
    Fetch ALL values of a relation for an entity (1-2 hop), OUT direction only.
    Returns a list of values (strings). Returns empty list if not found.
    """
    values = set()
    
    # Try direct 1-hop (OUT only)
    rels = m.get_relations(entity, "out")
    
    # Check OUT edges (Entity -[rel]-> Value)
    for _, p, v in rels.get('out', []):
        if p == relation:
            if not m.is_cvt(v):
                values.add(v)
            else:
                # If CVT, try to expand
                cvt_rels = m.get_relations(v, "out")
                for _, p2, v2 in cvt_rels.get('out', []):
                    if not m.is_cvt(v2):
                        values.add(v2)
    
    # Try 2-hop via CVT intermediates
    if max_hops >= 2:
        for _, p, v in rels.get('out', []):
            if m.is_cvt(v):
                cvt_rels = m.get_relations(v, "out")
                for _, p2, v2 in cvt_rels.get('out', []):
                    if p2 == relation and not m.is_cvt(v2):
                        values.add(v2)
    
    return sorted(list(values))


def _find_path_to_entity(m, source: str, target: str, max_hops: int = 2) -> Optional[List[str]]:
    """
    Check if there's a path from source to target within max_hops.
    Returns the path as list of strings, or None if not found.
    """
    # Try direct 1-hop
    rels = m.get_relations(source, "both")
    for direction in ['out', 'in']:
        for s, p, o in rels.get(direction, []):
            other = o if direction == 'out' else s
            if other == target:
                return [f"{source} -({p})-> {target}" if direction == 'out' else f"{source} <-({p})- {target}"]
            
            # Try 2-hop via intermediate
            if max_hops >= 2:
                int_rels = m.get_relations(other, "both")
                for dir2 in ['out', 'in']:
                    for s2, p2, o2 in int_rels.get(dir2, []):
                        other2 = o2 if dir2 == 'out' else s2
                        if other2 == target:
                            # Truncate intermediate node name for readability
                            int_name = other[:20] + "..." if len(other) > 20 else other
                            return [f"via {int_name}"]
    
    return None


def _execute_select(m, found_ends: set, select_relations: List[str], select_entities: List[str]) -> Tuple[Dict, Dict, set, Dict]:
    """
    Execute Select queries for relation values and entity connectivity.
    Returns:
      - relation_results: {entity: {relation: List[str]}}
      - entity_results: {entity: {target: path}}
      - matched_entities: set of entities that matched at least one EFFECTIVE constraint
      - constraint_stats: {constraint_name: {'matched': count, 'total': count, 'effective': bool}}
    """
    relation_results = {}
    entity_results = {}
    total_leaves = len(found_ends)
    
    # Track per-constraint match counts
    rel_match_counts = {rel: 0 for rel in select_relations}
    ent_match_counts = {ent: 0 for ent in select_entities}
    
    for leaf in found_ends:
        rel_matches = {}
        ent_matches = {}
        
        # Check select_relations
        for rel in select_relations:
            values = _fetch_relation_value(m, leaf, rel)  # Now returns List[str]
            if values:  # Non-empty list
                rel_matches[rel] = values
                rel_match_counts[rel] += 1
        
        # Check select_entities
        for target in select_entities:
            # Resolve target entity name
            target_res = m.find_entities(target, limit=3)
            resolved_target = target_res[0]['entity'] if target_res and target_res[0]['score'] >= 100 else target
            
            path = _find_path_to_entity(m, leaf, resolved_target)
            if path:
                ent_matches[target] = path
                ent_match_counts[target] += 1
        
        if rel_matches:
            relation_results[leaf] = rel_matches
        if ent_matches:
            entity_results[leaf] = ent_matches
    
    # Determine constraint effectiveness & Match Logic (UNION)
    # Rule: A constraint is effective if it matches SOME leaves (count > 0).
    # [V16 REFINEMENT]: Discriminative Constraint Logic
    constraint_stats = {}
    
    # Bucket constraints by their coverage
    global_constraints = []   # Matched ALL
    partial_constraints = []  # Matched SOME ( > 0 and < ALL)
    
    # 1. Analyze Relation Constraints
    for rel, count in rel_match_counts.items():
        effective = count > 0
        if count == total_leaves and total_leaves > 0:
            global_constraints.append(f"rel:{rel}")
            constraint_stats[f"rel:{rel}"] = {'matched': count, 'total': total_leaves, 'effective': True, 'type': 'GLOBAL'}
        elif count > 0:
            partial_constraints.append(f"rel:{rel}")
            constraint_stats[f"rel:{rel}"] = {'matched': count, 'total': total_leaves, 'effective': True, 'type': 'PARTIAL'}
        else:
            constraint_stats[f"rel:{rel}"] = {'matched': 0, 'total': total_leaves, 'effective': False, 'type': 'NONE'}
            
    # 2. Analyze Entity Constraints
    for ent, count in ent_match_counts.items():
        effective = count > 0
        if count == total_leaves and total_leaves > 0:
            global_constraints.append(f"ent:{ent}")
            constraint_stats[f"ent:{ent}"] = {'matched': count, 'total': total_leaves, 'effective': True, 'type': 'GLOBAL'}
        elif count > 0:
            partial_constraints.append(f"ent:{ent}")
            constraint_stats[f"ent:{ent}"] = {'matched': count, 'total': total_leaves, 'effective': True, 'type': 'PARTIAL'}
        else:
            constraint_stats[f"ent:{ent}"] = {'matched': 0, 'total': total_leaves, 'effective': False, 'type': 'NONE'}
    
    # 3. Decide which constraints to use for Filtering
    active_constraints = []
    
    if partial_constraints:
        active_constraints = partial_constraints
    elif global_constraints:
        active_constraints = global_constraints
    else:
        active_constraints = []
        
    # 4. Compute Union of Matches based on ACTIVE constraints
    matched_entities = set()
    
    if active_constraints:
        for leaf in found_ends:
            matches_any = False
            
            for c_key in active_constraints:
                c_type, c_val = c_key.split(':', 1)
                if c_type == 'rel':
                    if leaf in relation_results and c_val in relation_results[leaf]:
                        matches_any = True
                        break
                elif c_type == 'ent':
                    if leaf in entity_results and c_val in entity_results[leaf]:
                        matches_any = True
                        break
            
            if matches_any:
                matched_entities.add(leaf)
    
    return relation_results, entity_results, matched_entities, constraint_stats


def _suggest_discriminative_relations(m, candidates: List[str], min_candidates: int = 2) -> List[str]:
    """
    Analyze candidates and return common relations that have different values across them.
    This helps the model identify useful constraint relations for filtering.
    
    Logic:
    1. For each candidate, collect OUT relations and their values (1-hop, CVT-expanded 2-hop)
    2. Find relations that exist in ALL candidates (common relations)
    3. Among common relations, select those where values DIFFER between at least 2 candidates
    
    Args:
        m: GraphPatternMatcher instance
        candidates: List of candidate entity names
        min_candidates: Minimum number of candidates to trigger analysis
    
    Returns:
        List of discriminative relation names (sorted by discriminative score descending)
    """
    if len(candidates) < min_candidates:
        return []
    
    # Limit candidates to analyze (performance)
    candidates_to_analyze = candidates[:20]
    
    # 1. Collect relations and values for each candidate
    candidate_rel_values = {}  # {candidate: {relation: set(values)}}
    
    for cand in candidates_to_analyze:
        rel_values = defaultdict(set)
        rels = m.get_relations(cand, "out")
        
        for _, p, v in rels.get('out', []):
            if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."):
                continue
            
            # Check if v is CVT - if so, expand to get actual values
            if m.is_cvt(v):
                cvt_rels = m.get_relations(v, "out")
                for _, p2, v2 in cvt_rels.get('out', []):
                    if p2 in IGNORED_RELATIONS or p2.startswith("type.") or p2.startswith("common."):
                        continue
                    if not m.is_cvt(v2):
                        # Store the full 2-hop relation (both relations in path)
                        rel_values[p2].add(v2)  # Use the terminal relation as key
            else:
                rel_values[p].add(v)
        
        candidate_rel_values[cand] = rel_values
    
    # 2. Find common relations (exist in ALL candidates)
    if not candidate_rel_values:
        return []
    
    all_relation_sets = [set(rv.keys()) for rv in candidate_rel_values.values()]
    common_relations = set.intersection(*all_relation_sets) if all_relation_sets else set()
    
    # 3. Calculate discriminative score for each common relation
    discriminative_relations = []
    
    for rel in common_relations:
        # Collect values per candidate for this relation
        values_per_candidate = []
        for cand in candidates_to_analyze:
            vals = candidate_rel_values[cand].get(rel, set())
            values_per_candidate.append(frozenset(vals))
        
        # Count unique value combinations
        unique_value_sets = set(values_per_candidate)
        
        if len(unique_value_sets) > 1:
            # Values differ across candidates - this relation is discriminative
            # Score: number of unique value sets / total candidates (higher = more discriminative)
            score = len(unique_value_sets) / len(candidates_to_analyze)
            discriminative_relations.append((rel, score))
    
    # Sort by score descending, return top relations
    discriminative_relations.sort(key=lambda x: -x[1])
    
    # Return only relation names, limit to top 10
    return [rel for rel, score in discriminative_relations[:10]]


def _format_discriminative_relations(relations: List[str]) -> str:
    """Format discriminative relations as a hint block for the model."""
    if not relations:
        return ""
    
    lines = [
        "",
        "[Suggested Filter Relations]:",
        "The following relations have different values across candidates and can be used for filtering:",
    ]
    for rel in relations:
        lines.append(f"  - {rel}")
    
    lines.append("")
    lines.append("Hint: Use the filter() tool to verify which relation satisfies the question constraints.")
    
    return "\n".join(lines)


def _format_select_results(relation_results: Dict, entity_results: Dict, constraint_stats: Dict = None, matched_entities: set = None) -> str:
    """
    Format Select results as a text block for display.
    Display Logic:
    1. If matched_entities is provided, only show those entities.
    2. Only show PARTIAL constraints (discriminative), not GLOBAL (all-hit) ones.
    3. Do NOT show entities with no matches.
    """
    if not relation_results and not entity_results:
        return ""
    
    lines = ["[Select Results]:"]
    
    # Determine which constraints to display (only PARTIAL, not GLOBAL)
    partial_constraints = set()
    global_constraints = set()
    if constraint_stats:
        for c, stats in constraint_stats.items():
            if stats.get('type') == 'PARTIAL':
                partial_constraints.add(c)
            elif stats.get('type') == 'GLOBAL':
                global_constraints.add(c)
    
    # Show constraint effectiveness summary first (Brief)
    if constraint_stats:
        unmatched = [k for k, v in constraint_stats.items() if v['matched'] == 0]
        
        if unmatched:
            lines.append("  [Constraints with NO matches]:")
            for c in unmatched:
                lines.append(f"    - {c} (0 matched)")
    
    # Determine entities to display
    if matched_entities is not None:
        # Use provided filter set
        entities_to_show = matched_entities
    else:
        # Show all entities with results
        entities_to_show = set(relation_results.keys()) | set(entity_results.keys())
    
    has_displayed_entity = False
    for entity in sorted(entities_to_show):
        # Skip entities not in our filter set
        if matched_entities is not None and entity not in matched_entities:
            continue
            
        # Gather all props for this entity
        entity_props = []
        
        # Relation values - only show PARTIAL constraints (or all if no PARTIAL)
        if entity in relation_results:
            for rel, values in relation_results[entity].items():
                # [NEW] Skip GLOBAL constraints if PARTIAL constraints exist
                rel_key = f"rel:{rel}"
                if partial_constraints and rel_key in global_constraints:
                    continue  # Skip non-discriminative constraints
                    
                simple_rel = rel.split('.')[-1]
                # [FIX] Handle multiple values
                if isinstance(values, list):
                    val_str = ", ".join(values)
                else:
                    val_str = str(values)
                entity_props.append(f"{simple_rel}: {val_str}")
        
        # Entity links - only show PARTIAL constraints (or all if no PARTIAL)
        if entity in entity_results:
            for target, path in entity_results[entity].items():
                # [NEW] Skip GLOBAL entity constraints if PARTIAL exist
                ent_key = f"ent:{target}"
                if partial_constraints and ent_key in global_constraints:
                    continue
                    
                entity_props.append(f"linked to [{target}]")
                
        # Only display if there are properties found (Constraint Satisfied)
        if entity_props:
            has_displayed_entity = True
            lines.append(f"  {entity}:")
            for prop in entity_props:
                lines.append(f"    - {prop}")
                
    if not has_displayed_entity:
        lines.append("  (No entities satisfied any constraints)")
    
    return '\n'.join(lines)




def _sync_match_pattern(m, req):
    """Synchronous helper for match_pattern to run in thread pool.
    Supports both single entity and batch entity execution.
    """
    # Normalize anchor to list
    if isinstance(req.anchor, str):
        entities = [req.anchor]
    else:
        entities = req.anchor
    
    # Validate path relations first (same for all entities)
    path_errors = []
    for i, step in enumerate(req.path):
        rel = step.relation
        if rel not in m.relation2id:
            sims = m.find_similar_relations(rel, limit=5)
            cands = "\n".join([f"  - {x['relation']}" for x in sims]) if sims else "(No candidates)"
            path_errors.append(f"❌ Relation at Step {i+1} ('{rel}') not found.\nTop 5 Candidates:\n{cands}")
    
    if path_errors:
        return FinalResponse(success=True, status="KG_DIAGNOSIS_ERROR", response_text="[DIAGNOSTIC REPORT]\n" + "\n".join(path_errors))
    
    # Process each entity
    all_results = {}  # entity -> (paths, ends)
    entity_errors = []
    valid_entities = []
    
    for ent in entities:
        start_res = m.find_entities(ent, limit=5)
        if start_res and start_res[0]['score'] >= 100:
            valid_ent = start_res[0]['entity']
            valid_entities.append(valid_ent)
            
            steps = [s.dict() for s in req.path]
            raw_ends, paths = m.execute_match_pattern(valid_ent, steps)
            
            if paths:
                final_ends = collect_expanded_entities(m, valid_ent, paths)
                all_results[valid_ent] = {'paths': paths, 'ends': final_ends}
        else:
            cands = "\n".join([f"  - {x['entity']}" for x in start_res]) if start_res else "(No candidates)"
            entity_errors.append(f"Entity '{ent}' not found. Did you mean:\n{cands}")
    
    # If all entities failed
    if not all_results:
        if entity_errors:
            return FinalResponse(success=True, status="KG_ENTITY_ERROR", response_text="[DIAGNOSTIC REPORT]\n" + "\n".join(entity_errors))
        return FinalResponse(success=True, status="KG_NO_MATCHES", response_text="No matches found (Path logic valid, but no data instances).")
    
    # ===========================================================================
    # [NEW] SELECT FEATURE: Execute Select and Hard-Filter
    # ===========================================================================
    select_relation_results = {}
    select_entity_results = {}
    select_matched_entities = set()
    select_text = ""
    
    # [V13 FIX] Merge Aliased Constraints
    # The model may output 'constraint_relations'/'constraint_entities' but backend logic uses 'select_*'
    effective_select_relations = req.select_relations + req.constraint_relations
    effective_select_entities = req.select_entities + req.constraint_entities
    
    should_filter = False # Default: do not filter unless constraints exist and matching was done
    
    if effective_select_relations or effective_select_entities:
        # Collect all leaf entities across all start entities
        all_leaf_ends = set()
        for data in all_results.values():
            all_leaf_ends.update(data['ends'])
        
        # [FIX] Filter out redundant select_relations (those already in path)
        path_relations = set(s.relation for s in req.path)
        filtered_select_relations = [r for r in effective_select_relations if r not in path_relations]
        
        # [FIX] Filter out select_entities that equal the anchor (self-reference)
        anchor_names = set(entities)  # Original anchor names
        anchor_lower = set(e.lower() for e in entities)
        filtered_select_entities = [e for e in effective_select_entities 
                                     if e.lower() not in anchor_lower]
        
        # Execute Select queries with constraint effectiveness detection
        select_relation_results, select_entity_results, select_matched_entities, constraint_stats = _execute_select(
            m, all_leaf_ends, filtered_select_relations, filtered_select_entities
        )
        
        # Hard-filter: Remove leaf entities that don't match EFFECTIVE constraints
        # Logic: 
        # 1. If select_matched_entities is NON-EMPTY, keeps only those (Union logic success).
        # 2. If select_matched_entities is EMPTY (but constraints existed):
        #    - Means NO overlap between data and constraints.
        #    - FALLBACK: Keep ALL original data (don't return empty). Assumes constraints were too strict or misaligned.
        
        # [V13 DECISION] SEPARATION OF PATTERN MATCHING & CONSTRAINTS
        # We DO NOT filter the main results based on constraints.
        # Constraints are an annotation layer. We return ALL path-found entities.
        # The user can see which ones matched via the [Select Results] section.
        #
        # [V14 UPDATED]: User requested strict pruning.
        # If constraints are present and matched, we ONLY return what matched.
        should_filter = True # Enable hard filtering
        
        # Logic for Constraint Status Message
        fallback_msg = ""
        if (effective_select_relations or effective_select_entities) and not select_matched_entities:
             # Case: Constraints existed but NO matches found.
             # Option A: Return empty (Strict)
             # Option B: Fallback (Lenient) -> "No entities matched, showing all."
             # User requested "Pruning". Strict pruning implies empty result.
             # However, for better UX, if EVERYTHING is pruned, we might want to warn.
             # Let's stick to the simulation: "User sees nothing for mismatch".
             # BUT, if result is empty, the UI might show "No matches".
             # Let's use the explicit request: "Refine output".
             pass
        elif select_matched_entities:
             fallback_msg = f"\n  [Constraint Status]: {len(select_matched_entities)} candidates satisfied the constraints."

        # Format Select results text with constraint effectiveness info
        select_text = _format_select_results(select_relation_results, select_entity_results, constraint_stats, select_matched_entities)
        if fallback_msg:
            select_text = select_text + fallback_msg
    
    # Format results
    if len(all_results) == 1:
        # Single entity - use original format with structured data
        ent = list(all_results.keys())[0]
        # [V13] Pass filtered_ends if filtering is enabled and we have matches
        # If should_filter is True and we have select_matched_entities, use it.
        # If no matches found but constraints existed (and we are strict), pass empty set?
        # Let's define the filter set:
        final_filter_set = None
        if should_filter and (effective_select_relations or effective_select_entities):
            if select_matched_entities:
                final_filter_set = select_matched_entities
            else:
                # No matches for constraints — return ALL candidates instead of empty.
                # LLM can still extract useful info from the full set, but not from empty.
                final_filter_set = None
                
        txt, struct_dict = format_match_results(ent, all_results[ent]['paths'], m, filtered_ends=final_filter_set)
        
        # Build StructuredData object - pass ALL fields from struct_dict
        structured = StructuredData(
            start_entity=ent,
            path=[s.dict() for s in req.path],
            leaf_entities=struct_dict.get('leaf_entities', []),
            cvt_nodes=struct_dict.get('cvt_nodes', {}),
            path_triples=struct_dict.get('path_triples', []),
            logic_pattern=struct_dict.get('logic_pattern', ''),
            # [FIX] Include ALL structured data fields
            branches=struct_dict.get('branches', []),
            leaf_to_shortest_path=struct_dict.get('leaf_to_shortest_path', {}),
            data_tree=struct_dict.get('data_tree', {}),
            native_leaves=struct_dict.get('native_leaves', []),
            cvt_expanded_leaves=struct_dict.get('cvt_expanded_leaves', []),
            cvt_leaf_map=struct_dict.get('cvt_leaf_map', {}),
        )
        
        # [NEW] Append Select results if any
        if select_text:
            txt += "\n" + select_text
        
        # [NEW] Suggest Discriminative Relations if:
        # 1. Multiple candidates exist (need filtering)
        # 2. No constraints were specified OR constraints didn't successfully filter
        leaf_entities = struct_dict.get('leaf_entities', [])
        has_constraints = bool(effective_select_relations or effective_select_entities)
        constraints_hit = bool(select_matched_entities)
        
        if len(leaf_entities) > 1 and (not has_constraints or not constraints_hit):
            disc_relations = _suggest_discriminative_relations(m, leaf_entities)
            if disc_relations:
                txt += _format_discriminative_relations(disc_relations)
        
        return FinalResponse(
            success=True, 
            status="KG_SUCCESS", 
            response_text=txt, 
            found_end_entities=all_results[ent]['ends'],
            structured_data=structured
        )
    else:
        # Multiple entities - choose format based on query type
        if req.is_constraint_query:
            # Constraint query: use constraint format with CVT expansion
            txt = format_constraint_results(all_results, req.path, m, req.constraint_relation)
        else:
            # Regular batch: use compact entity pairs format
            txt = format_batch_match_results(all_results, req.path, m)
        
        all_ends = []
        for data in all_results.values():
            all_ends.extend(data['ends'])
        
        # [NEW] Append Select results if any
        if select_text:
            txt += "\n" + select_text
        
        # [NEW] Suggest Discriminative Relations if:
        # 1. Multiple candidates exist (need filtering)
        # 2. No constraints were specified OR constraints didn't successfully filter
        unique_ends = list(set(all_ends))
        has_constraints = bool(effective_select_relations or effective_select_entities)
        constraints_hit = bool(select_matched_entities)
        
        if len(unique_ends) > 1 and (not has_constraints or not constraints_hit):
            disc_relations = _suggest_discriminative_relations(m, unique_ends)
            if disc_relations:
                txt += _format_discriminative_relations(disc_relations)
        
        return FinalResponse(success=True, status="KG_SUCCESS", response_text=txt, found_end_entities=unique_ends)


@app.post("/v2/match_pattern", response_model=FinalResponse)
async def api_match_pattern(req: MatchRequestV2):
    try:
        m = manager.get_matcher(req.sample_id)
        return await asyncio.to_thread(_sync_match_pattern, m, req)
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))


def _sync_get_neighbors(m, req):
    """Synchronous helper for get_neighbors to run in thread pool."""
    entity_res = m.find_entities(req.entity, limit=5)
    valid_entity = None
    if entity_res and entity_res[0]['score'] >= 100:
        valid_entity = entity_res[0]['entity']
    else:
        cands = "\n".join([f"  - {x['entity']}" for x in entity_res]) if entity_res else "(No candidates)"
        return FinalResponse(success=True, status="KG_ENTITY_ERROR", response_text=f"❌ Entity '{req.entity}' not found.\nTop 5 Candidates:\n{cands}")

    neighbors = m.get_relations(valid_entity, req.direction)
    
    limited_neighbors = {}
    for d, trips in neighbors.items():
        limited_neighbors[d] = trips[:req.limit * 5]

    txt = format_neighbors(valid_entity, limited_neighbors)
    
    found_entities = set()
    for d, trips in neighbors.items():
        for s, p, o in trips[:req.limit]:
            target = o if d == 'out' else s
            if not m.is_cvt(target):
                found_entities.add(target)
            
    return FinalResponse(success=True, status="KG_SUCCESS", response_text=txt, found_end_entities=list(found_entities))


@app.post("/v2/get_neighbors", response_model=FinalResponse)
async def api_get_neighbors(req: NeighborsRequest):
    try:
        m = manager.get_matcher(req.sample_id)
        return await asyncio.to_thread(_sync_get_neighbors, m, req)
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))


@app.post("/v2/find_logical_path_with_relation", response_model=FinalResponse)
async def api_find_logical_path(req: PathRequest):
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            return FinalResponse(success=False, status="KG_ERROR", response_text="Sample ID not found")
        return await asyncio.to_thread(_sync_find_logical_path, m, req)
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))

class RelationsRequest(BaseModel):
    sample_id: str = "default"
    relation: str

@app.post("/v2/find_relations", response_model=FinalResponse)
async def api_find_relations(req: RelationsRequest):
    try:
        # [CRITICAL] Heuristic Check for Entity-in-Relation (Global)
        # Fail fast if input looks like an entity (invalid for relation search)
        rel = req.relation
        if '.' not in rel and (' ' in rel or rel[0].isupper()):
            return FinalResponse(
                success=True,
                status="KG_TYPE_ERROR",
                response_text = f"❌ TYPE ERROR: '{rel}' looks like an Entity Name.",
                action_hints=[{
                    "error_type": "TYPE_MISMATCH",
                    "value": rel,
                    "source_field": "relation",
                    "suggestion": "Did you mean to use 'check_entities'?",
                    "error": "This looks like an Entity Name, but you are searching for a Relation."
                }]
            )

        m = manager.get_matcher(req.sample_id)
        # Use existing find_similar_relations
        # Logic: If exact match found (score=100), verified. Else return similar.
        candidates = m.find_similar_relations(req.relation, limit=5)
        
        # Check for exact match
        found = any(c['relation'] == req.relation and c['score'] == 100 for c in candidates)
        if found:
            return FinalResponse(success=True, status="KG_SUCCESS", response_text=f"Relation '{req.relation}' exists.", found_end_entities=[req.relation])
        
        # If not found, return error with candidates
        cand_str = "\n".join([f"  - {x['relation']} (score: {x['score']})" for x in candidates])
        return FinalResponse(success=True, status="KG_RELATION_ERROR", response_text=f"Relation '{req.relation}' not found.\nDid you mean:\n{cand_str}")
        
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))


def _process_single_constraint(m, candidate: str, req: BatchConstraintRequest) -> Tuple[str, Dict, Optional[str]]:
    """Process a single candidate for constraint filtering."""
    # Resolve entity name
    entity_res = m.find_entities(candidate, limit=3)
    if not entity_res or entity_res[0]['score'] < 100:
        return candidate, {'paths': [], 'ends': []}, "ENTITY_ERROR"
    
    valid_entity = entity_res[0]['entity']
    
    # Find paths containing constraint relation
    paths = m.find_paths_with_relation(
        valid_entity, 
        req.constraint_relation, 
        req.max_hops, 
        req.limit_per_entity
    )
    
    if not paths:
        return valid_entity, {'paths': [], 'ends': []}, None
    
    # Collect leaf entities with CVT expansion
    ends = collect_expanded_entities(m, valid_entity, paths)
    return valid_entity, {'paths': paths, 'ends': ends}, None


@app.post("/v2/filter_by_constraint", response_model=FinalResponse)
async def api_filter_by_constraint(req: BatchConstraintRequest):
    """
    Filter candidates by constraint API (Concurrent Version).
    Executes graph queries in parallel to avoid timeout.
    """
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            raise Exception("Sample ID not found")
            
        loop = asyncio.get_running_loop()
        tasks = []
        
        # Launch parallel tasks for each candidate
        for candidate in req.candidates:
            tasks.append(
                loop.run_in_executor(None, _process_single_constraint, m, candidate, req)
            )
        
        # Wait for all to complete
        task_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        results = {}
        entity_errors = []
        
        for original_cand, result_data, error in task_results:
            if error == "ENTITY_ERROR":
                entity_errors.append(original_cand)
                results[original_cand] = result_data
            else:
                results[original_cand] = result_data
        
        # Check if all failed
        if not any(data['ends'] for data in results.values()):
            if entity_errors:
                return FinalResponse(
                    success=True, 
                    status="KG_ENTITY_ERROR", 
                    response_text=f"Entities not found: {entity_errors[:5]}{'...' if len(entity_errors) > 5 else ''}",
                    found_end_entities=[]
                )
            return FinalResponse(
                success=True, 
                status="KG_NO_MATCHES", 
                response_text=f"No paths found from any candidate to relation '{req.constraint_relation}'",
                found_end_entities=[]
            )
        
        # Format results using constraint format
        txt = format_constraint_results(
            results, 
            path=None,
            matcher=m,
            constraint_relation=req.constraint_relation
        )
        
        # Collect all end entities
        all_ends = []
        for data in results.values():
            all_ends.extend(data['ends'])
        
        return FinalResponse(
            success=True, 
            status="KG_SUCCESS", 
            response_text=txt, 
            found_end_entities=list(set(all_ends))
        )
        
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=str(e))


# ==============================================================================
# NEW: filter (Simplified Constraint Verification Tool)
# ==============================================================================

class FilterRequest(BaseModel):
    """Request for the new 'filter' tool."""
    sample_id: str = "default"
    candidates: List[str] = Field(..., description="Candidate entities to filter (provided by frontend)")
    constraint_relations: List[str] = Field(default_factory=list, description="Relations to check as constraints")
    constraint_entities: List[str] = Field(default_factory=list, description="Entities to check connectivity with")
    plan_relations: List[str] = Field(default_factory=list, description="Main plan relations for restricted CVT neighbor display")


def _domains_from_relations(relations: List[str]) -> set:
    return {r.split(".", 1)[0] for r in relations if isinstance(r, str) and "." in r}


def _restricted_cvt_neighbors(m, cvt_node: str, allowed_domains: set) -> List[str]:
    rels = m.get_relations(cvt_node, "both")
    previews = []
    seen = set()
    for direction in ["out", "in"]:
        for s, p, o in rels.get(direction, []):
            if p in IGNORED_RELATIONS or p.startswith("type.") or p.startswith("common."):
                continue
            domain = p.split(".", 1)[0] if "." in p else ""
            if allowed_domains and domain not in allowed_domains:
                continue
            neighbor = o if direction == "out" else s
            if not neighbor or neighbor in IGNORED_ENTITY_STRINGS:
                continue
            label = f"{_simplify_relation_name(p)} -> {neighbor}" if direction == "out" else f"<- {_simplify_relation_name(p)}: {neighbor}"
            if label not in seen:
                previews.append(label)
                seen.add(label)
            if len(previews) >= 8:
                return previews
    return previews


def _fetch_relation_matches(m, entity: str, relation: str, allowed_domains: set, max_hops: int = 2) -> Dict:
    """Fetch constraint-relation evidence with CVT-aware restricted diagnostics."""
    matched_entities = set()
    cvt_hits = []
    rels = m.get_relations(entity, "out")

    for _, p, v in rels.get('out', []):
        if p != relation:
            continue
        if not m.is_cvt(v):
            if v not in IGNORED_ENTITY_STRINGS:
                matched_entities.add(v)
        else:
            cvt_hits.append({
                "cvt": v,
                "neighbors": _restricted_cvt_neighbors(m, v, allowed_domains),
            })

    if max_hops >= 2:
        for _, _, v in rels.get('out', []):
            if not m.is_cvt(v):
                continue
            cvt_rels = m.get_relations(v, "out")
            for _, p2, v2 in cvt_rels.get('out', []):
                if p2 != relation:
                    continue
                if not m.is_cvt(v2) and v2 not in IGNORED_ENTITY_STRINGS:
                    matched_entities.add(v2)
                elif m.is_cvt(v2):
                    cvt_hits.append({
                        "cvt": v2,
                        "neighbors": _restricted_cvt_neighbors(m, v2, allowed_domains),
                    })

    return {
        "entities": sorted(matched_entities),
        "cvt_hits": cvt_hits,
    }


def _check_single_candidate_constraints(
    m,
    candidate: str,
    constraint_relations: List[str],
    constraint_entities: List[str],
    plan_relations: List[str],
) -> Dict:
    """
    Check a single candidate against constraint relations and entities.
    Returns: {
        'entity': resolved_name,
        'relation_matches': {rel: value, ...},
        'entity_matches': {ent: True/False, ...},
        'error': None or error_type
    }
    """
    # Resolve entity name
    entity_res = m.find_entities(candidate, limit=3)
    if not entity_res or entity_res[0]['score'] < 100:
        return {'entity': candidate, 'relation_matches': {}, 'entity_matches': {}, 'error': 'ENTITY_ERROR'}
    
    valid_entity = entity_res[0]['entity']
    result = {'entity': valid_entity, 'relation_matches': {}, 'entity_matches': {}, 'error': None}
    allowed_domains = _domains_from_relations(constraint_relations + plan_relations)
    
    # Check constraint relations (direct / CVT-bridged attribute lookup)
    for rel in constraint_relations:
        info = _fetch_relation_matches(m, valid_entity, rel, allowed_domains, max_hops=2)
        if info["entities"] or info["cvt_hits"]:
            result['relation_matches'][rel] = info
    
    # Check constraint entities (connectivity check)
    for target_ent in constraint_entities:
        # Resolve target entity
        target_res = m.find_entities(target_ent, limit=3)
        resolved_target = target_res[0]['entity'] if target_res and target_res[0]['score'] >= 100 else target_ent
        
        path = _find_path_to_entity(m, valid_entity, resolved_target, max_hops=2)
        if path:
            result['entity_matches'][target_ent] = True
        else:
            result['entity_matches'][target_ent] = False
    
    return result


def _format_filter_results(all_results: List[Dict], constraint_relations: List[str], constraint_entities: List[str]) -> Tuple[str, List[str]]:
    """
    Format filter results with global constraint hiding logic.
    
    Logic:
    - If a constraint matches ALL entities AND there are multiple constraints, hide it (not discriminative)
    - If only one constraint and it's global, show it anyway
    
    Returns: (formatted_text, matched_entities_list)
    """
    total_candidates = len(all_results)
    valid_candidates = [r for r in all_results if r['error'] is None]
    valid_count = len(valid_candidates)
    
    if valid_count == 0:
        return "[Filter Result] No valid candidates to filter.", []
    
    # Count matches per constraint
    rel_match_counts = {rel: 0 for rel in constraint_relations}
    ent_match_counts = {ent: 0 for ent in constraint_entities}
    
    for r in valid_candidates:
        for rel in constraint_relations:
            if rel in r['relation_matches']:
                rel_match_counts[rel] += 1
        for ent in constraint_entities:
            if r['entity_matches'].get(ent, False):
                ent_match_counts[ent] += 1
    
    # Determine which constraints are "global" (match ALL) vs "partial" (match SOME)
    global_constraints = []
    partial_constraints = []
    
    for rel, count in rel_match_counts.items():
        if count == valid_count:
            global_constraints.append(f"rel:{rel}")
        elif count > 0:
            partial_constraints.append(f"rel:{rel}")
    
    for ent, count in ent_match_counts.items():
        if count == valid_count:
            global_constraints.append(f"ent:{ent}")
        elif count > 0:
            partial_constraints.append(f"ent:{ent}")
    
    # Decide which constraints to show
    total_constraints = len(constraint_relations) + len(constraint_entities)
    if total_constraints == 1:
        # Only one constraint - show it even if global
        active_constraints = global_constraints + partial_constraints
    elif partial_constraints:
        # Multiple constraints exist - hide global ones, show partial (discriminative)
        active_constraints = partial_constraints
    else:
        # All constraints are global - show them anyway
        active_constraints = global_constraints
    
    # Build output
    lines = ["[Filter Results]"]
    lines.append(f"Candidates Checked: {total_candidates} (Valid: {valid_count})")
    lines.append("")
    
    # Show constraint effectiveness
    lines.append("[Constraint Effectiveness]:")
    for rel in constraint_relations:
        count = rel_match_counts[rel]
        is_global = count == valid_count
        is_active = f"rel:{rel}" in active_constraints
        status = "GLOBAL (hidden)" if is_global and not is_active else ("DISCRIMINATIVE" if count > 0 else "NO MATCH")
        lines.append(f"  - {rel}: {count}/{valid_count} matched ({status})")
    
    for ent in constraint_entities:
        count = ent_match_counts[ent]
        is_global = count == valid_count
        is_active = f"ent:{ent}" in active_constraints
        status = "GLOBAL (hidden)" if is_global and not is_active else ("DISCRIMINATIVE" if count > 0 else "NO MATCH")
        lines.append(f"  - (entity) {ent}: {count}/{valid_count} connected ({status})")
    
    lines.append("")
    
    # Show per-candidate results (only for active constraints)
    lines.append("[Per-Candidate Matches]:")
    matched_entities = []
    
    for r in valid_candidates:
        entity = r['entity']
        matches = []
        
        for rel in constraint_relations:
            if f"rel:{rel}" in active_constraints and rel in r['relation_matches']:
                info = r['relation_matches'][rel]
                vals = info.get('entities', [])
                cvt_hits = info.get('cvt_hits', [])
                if vals:
                    matches.append(f"{rel.split('.')[-1]}={vals}")
                for cvt in cvt_hits[:2]:
                    neighbors = cvt.get('neighbors', [])
                    preview = "; ".join(neighbors[:4]) if neighbors else "(no restricted neighbors)"
                    matches.append(f"{rel.split('.')[-1]}=>{cvt.get('cvt')} [{preview}]")
        
        for ent in constraint_entities:
            if f"ent:{ent}" in active_constraints and r['entity_matches'].get(ent, False):
                matches.append(f"→{ent}")
        
        if matches:
            matched_entities.append(entity)
            lines.append(f"  ✓ {entity}: {', '.join(matches)}")
        else:
            lines.append(f"  ✗ {entity}: (no match)")
    
    lines.append("")
    lines.append(f"[Summary]: {len(matched_entities)}/{valid_count} candidates passed filter.")
    
    return "\n".join(lines), matched_entities


@app.post("/v2/filter", response_model=FinalResponse)
async def api_filter(req: FilterRequest):
    """
    New simplified constraint filter tool.
    
    - Candidates provided by frontend (environment)
    - Checks constraint_relations (attribute values) and constraint_entities (connectivity)
    - Implements "global constraint hiding" for better discrimination
    """
    try:
        m = manager.get_matcher(req.sample_id)
        if not m:
            raise Exception("Sample ID not found")
        
        if not req.candidates:
            return FinalResponse(success=True, status="KG_ERROR", response_text="[Filter Error] No candidates provided.")
        
        if not req.constraint_relations and not req.constraint_entities:
            return FinalResponse(success=True, status="KG_ERROR", response_text="[Filter Error] No constraints specified.")
        
        loop = asyncio.get_running_loop()
        tasks = []
        
        # Launch parallel checks for each candidate
        for candidate in req.candidates:
            tasks.append(
                loop.run_in_executor(
                    None, 
                    _check_single_candidate_constraints, 
                    m, candidate, req.constraint_relations, req.constraint_entities, req.plan_relations
                )
            )
        
        # Wait for all to complete
        all_results = await asyncio.gather(*tasks)
        
        # Format results
        txt, matched_entities = _format_filter_results(
            all_results, req.constraint_relations, req.constraint_entities
        )
        
        return FinalResponse(
            success=True,
            status="KG_SUCCESS",
            response_text=txt,
            found_end_entities=matched_entities
        )
        
    except Exception as e:
        return FinalResponse(success=False, status="KG_ERROR", response_text=f"[Filter Error] {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset", default="webqsp")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    os.environ["DATASET_NAME"], os.environ["SPLIT"] = args.dataset, args.split
    uvicorn.run(app, host="0.0.0.0", port=args.port)
