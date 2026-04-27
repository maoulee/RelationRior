# file: kg_matcher_refactored.py
# (您可以将此内容放入 kg_matcher.py 或直接放入 graph_server.py)

import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

class KGMatcher:
    """
    知识图谱实体匹配器 (重构版)
    - 修复了分数爆炸 Bug，采用绝对归一化。
    - 优化了分层匹配逻辑，提高了准确性和可靠性。
    - 保持零依赖特性。
    """
    
    def __init__(self, entities: List[str], popularities: Optional[Dict[str, float]] = None):
        """
        初始化匹配器
        
        Args:
            entities: 实体名称列表
            popularities: 可选的实体流行度分数（例如 log(degree)）
        """
        self.entities = list(dict.fromkeys(entities))
        self.popularities = popularities or {}
        
        # --- 可配置参数 ---
        self.config = {
            'scorers': {
                'jaccard': {'weight': 1.5},
                'ngram': {'weight': 1.0},
                'levenshtein': {'weight': 0.8},
            },
            'short_query_len': 4,
            'prefix_bonus': 5,
            'exact_inclusion_bonus': 15,
            'popularity_weight': 2.0,
        }
        
        # 理论最高分，用于绝对归一化
        self.THEORETICAL_MAX_SCORE = sum(100 * v['weight'] for v in self.config['scorers'].values())
        
        self._build_indexes()
    
    def _build_indexes(self):
        """构建用于快速召回的索引"""
        # L0: 精确匹配索引
        self.exact_map = {entity.lower().strip(): entity for entity in self.entities}
        
        # L1: 词汇倒排索引
        self.word_index = defaultdict(list)
        for entity in self.entities:
            for word in self._tokenize(entity):
                self.word_index[word].append(entity)
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词，返回唯一的、排序的词语列表"""
        return sorted(list(set(re.findall(r'\b\w+\b', text.lower()))))
    
    @lru_cache(maxsize=10000)
    def find_entities(self, query: str, limit: int = 5, threshold: int = 70) -> List[Dict[str, Any]]:
        """
        主查询接口 - 重构后的多阶段匹配流程
        """
        query = query.strip()
        query_lower = query.lower()
        if not query:
            return []
        
        # 1. L0: 精确匹配 (最高优先级)
        if query_lower in self.exact_map:
            return [{"entity": self.exact_map[query_lower], "score": 100}]
        
        # 2. L1: 候选者生成
        candidates = self._generate_candidates(query)
        if not candidates:
            return []
        
        # 3. L2: 多通道评分
        candidate_scores = {}
        query_words = self._tokenize(query)

        for entity in candidates:
            entity_words = self._tokenize(entity)
            
            # --- 计算各项特征分数 (0-100) ---
            score_jaccard = self._jaccard_similarity(query_words, entity_words)
            score_ngram = self._ngram_similarity(query_lower, entity.lower())
            score_lev = self._levenshtein_similarity(query_lower, entity.lower())
            
            # --- 融合原始分数 ---
            raw_score = (
                score_jaccard * self.config['scorers']['jaccard']['weight'] +
                score_ngram * self.config['scorers']['ngram']['weight'] +
                score_lev * self.config['scorers']['levenshtein']['weight']
            )
            
            # --- 应用启发式加分/减分 ---
            # a. 精确包含加分
            if query_lower in entity.lower():
                raw_score += self.config['exact_inclusion_bonus']
            # b. 前缀匹配加分
            elif entity.lower().startswith(query_lower):
                raw_score += self.config['prefix_bonus']
            
            candidate_scores[entity] = raw_score

        if not candidate_scores:
            return []

        # 4. 绝对归一化、加入流行度、排序和决策
        final_results = []
        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        
        for entity, raw_score in sorted_candidates:
            # [核心修复] 绝对归一化
            normalized_score = int((raw_score / self.THEORETICAL_MAX_SCORE) * 100)
            
            # 加入流行度分数
            normalized_score += self.popularities.get(entity, 0) * self.config['popularity_weight']
            
            # 确保分数在 0-100 之间
            final_score = max(0, min(100, int(normalized_score)))

            if final_score >= threshold:
                final_results.append({"entity": entity, "score": final_score})
        
        return final_results[:limit]
    
    def _generate_candidates(self, query: str) -> set:
        """使用倒排索引快速生成候选者"""
        query_words = self._tokenize(query)
        if not query_words:
            return set()
        
        candidate_sets = [set(self.word_index.get(word, [])) for word in query_words]
        
        # 优先取交集
        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        
        # 如果交集结果太少，用并集补充
        if len(candidates) < 20:
            union_candidates = set.union(*candidate_sets) if candidate_sets else set()
            # 限制补充的数量，避免性能问题
            candidates.update(list(union_candidates)[:100])
            
        return candidates

    # --- 特征计算函数 ---
    @staticmethod
    def _jaccard_similarity(set1: list, set2: list) -> float:
        s1, s2 = set(set1), set(set2)
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return (intersection / union) * 100 if union > 0 else 0.0

    @staticmethod
    def _ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
        s1, s2 = s1.replace(" ", ""), s2.replace(" ", "")
        if not s1 or not s2: return 0.0
        
        ngrams1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
        ngrams2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
        
        return KGMatcher._jaccard_similarity(list(ngrams1), list(ngrams2))

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        dist = KGMatcher._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return (1 - dist / max_len) * 100 if max_len > 0 else 100.0

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2): s1, s2 = s2, s1
        if len(s2) == 0: return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions, deletions = prev_row[j + 1] + 1, curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
        
    def find_corrected_entity(self, query: str, threshold: int = 80, limit: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """兼容旧接口"""
        results = self.find_entities(query, limit, threshold)
        if not results: return query, []
        best = results[0]
        if best['score'] >= threshold:
            return best['entity'], results
        return query, results