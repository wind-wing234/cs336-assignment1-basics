import os
import regex as re
import time
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Iterator

class PreTokenizer:
    def __init__(self, special_tokens: list[str]):
        """
        初始化预处理器
        
        Args:
            special_tokens: 特殊token列表
        """
        # 按长度降序排列特殊token，确保先匹配最长的token
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else r"(?!)" # 当没有特殊token时，使用永远不分割的表达式
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def read_corpus(self, input_path: str) -> List[str]:
        """读取语料并按特殊token进行分割"""
        with open(input_path, 'rb') as f:
            corpus = f.read()
        return re.split(self.special_tokens_pattern, corpus.decode('utf-8'))
    
    def build_word_frequency(self, docs: List[str]) -> Counter:
        """构建词频字典"""
        word_freq = Counter()
        
        for doc in docs:
            if not doc:
                continue
            word_freq.update(word.group(0).encode('utf-8') for word in self.word_pattern.finditer(doc))
        
        return word_freq
    
    def pretokenize(self, text: str) -> List[bytes]:
        """
        将输入文本预分词，包括常规token和特殊token
        
        Args:
            text: 输入文本
        
        Returns:
            预分词结果bytes列表
        """
        # 首先按特殊token进行分割，但保留特殊token
        parts = re.split(f'({self.special_tokens_pattern})', text)
        
        result = []
        
        for part in parts:
            if part in self.special_tokens:
                # 特殊token直接放入
                result.append(part.encode('utf-8'))
            elif part:  # 跳过空字符串
                # 对普通文本使用word_pattern进行分词
                tokens = [match.group(0).encode('utf-8') for match in self.word_pattern.finditer(part)]
                result.extend(tokens)
        
        return result
    
    def pretokenize_iter(self, texts: Iterable[str]) -> Iterator[bytes]:
        """
        将输入的可迭代字符串对象预分词，并以迭代器形式返回词元
        
        Args:
            texts: 可迭代的字符串对象（例如文件句柄或字符串列表）
        
        Returns:
            生成预分词结果bytes的迭代器
        """
        for text in texts:
            # 首先按特殊token进行分割，但保留特殊token
            parts = re.split(f'({self.special_tokens_pattern})', text)
            
            for part in parts:
                if part in self.special_tokens:
                    # 特殊token直接生成
                    yield part.encode('utf-8')
                elif part:  # 跳过空字符串
                    # 对普通文本使用word_pattern进行分词并生成
                    for match in self.word_pattern.finditer(part):
                        yield match.group(0).encode('utf-8')
