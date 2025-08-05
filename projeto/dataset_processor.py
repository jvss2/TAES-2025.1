#!/usr/bin/env python3

import random
import logging
from typing import List, Dict, Any, Tuple, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


class DatasetProcessor:
    
    @staticmethod
    def load_dataset(name: str, split: str, min_executed: int) -> List[Dict[str, Any]]:
        logger.info(f"Loading dataset '{name}' (split={split})")
        
        try:
            ds = load_dataset(name, split=split)
            total_samples = len(ds)
            logger.info(f"Total samples in dataset: {total_samples}")
            
            filtered = [
                sample for sample in ds 
                if len(sample.get("executed_lines", [])) >= min_executed
            ]
            
            filtered_count = len(filtered)
            logger.info(f"Samples with >= {min_executed} executed lines: {filtered_count}")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    @staticmethod
    def prepare_prompt(source: str, executed_lines: List[int], placeholder: str, docstring: Optional[str] = None) -> Tuple[str, str]:
        lines = source.split("\n")
        
        if not executed_lines:
            raise ValueError("executed_lines list is empty")
        
        valid_code_lines = DatasetProcessor._find_valid_code_lines(lines)
        
        if not valid_code_lines:
            raise ValueError("No valid code lines found in function")
        
        target_idx = random.choice(valid_code_lines)
        target_line = lines[target_idx].strip()
        context_lines = lines[:target_idx]
        context = "\n".join(context_lines)

        if docstring:
            docstring_cleaned = docstring.strip()
            if docstring_cleaned:
                context = f'"""{docstring_cleaned}"""\n\n' + context
        
        return context, target_line
    
    @staticmethod
    def _find_valid_code_lines(lines: List[str]) -> List[int]:
        valid_indices = []
        in_docstring = False
        docstring_delim = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                continue
            
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_delim = stripped[:3]
                    in_docstring = True
                    if stripped.count(docstring_delim) >= 2 and len(stripped) > 3:
                        in_docstring = False
                    continue
            else:
                if docstring_delim in stripped:
                    in_docstring = False
                continue
            
            if in_docstring:
                continue
            
            if stripped.startswith('#'):
                continue
            
            if stripped.startswith('def ') or stripped.startswith('class '):
                continue
            
            if stripped.startswith('@'):
                continue
            
            valid_indices.append(i)
        
        return valid_indices