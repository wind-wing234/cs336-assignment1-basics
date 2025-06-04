import os
import regex as re
import time
from collections import Counter
from typing import BinaryIO, Dict, List, Tuple, Iterable, Iterator
from tqdm import tqdm

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
        
    def find_chunk_boundaries(
        self,
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def read_corpus(self, input_path: str) -> Iterable[List[str]]:
        """读取语料并按特殊token进行分割"""
        with open(input_path, 'rb') as f:
            # 获取分块边界
            boundaries = self.find_chunk_boundaries(f, 100, "<|endoftext|>".encode('utf-8'))
        
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])), desc="读取语料"):
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
                yield re.split(self.special_tokens_pattern, chunk)
    
    def build_word_frequency(self, docs: Iterable[str]) -> Counter:
        """构建词频字典"""
        word_freq = Counter()
        str_freq = Counter()
        
        for doc in docs:
            if not doc:
                # 跳过空字符串
                continue
            # 收集doc中所有单词
            matches = [word.group(0) for word in self.word_pattern.finditer(doc)]
            # 批量更新计数器
            str_freq.update(matches)

        
        # 将字符串词频转换为字节词频
        for word, freq in str_freq.items():
            word_freq[word.encode('utf-8')] = freq

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
