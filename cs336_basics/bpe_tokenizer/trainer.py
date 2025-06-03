import os
import regex as re
import time
from collections import Counter
from typing import Dict, List, Set, Tuple
import cProfile
import pstats

from cs336_basics.bpe_tokenizer.pre_tokenizer import PreTokenizer

class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        """
        初始化BPE分词器
        
        Args:
            vocab_size: 词汇表大小
            special_tokens: 特殊token列表
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.preprocessor = PreTokenizer(special_tokens)
        self.token_vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.splits: Dict[bytes, List[bytes]] = {}  # b"going" -> [b'g', b'o', b'ing']
        self.pair_freqs: Dict[Tuple[bytes, bytes], int] = {}

        # 反向索引，记录每个pair出现在哪些单词中
        self.pair_to_words: Dict[Tuple[bytes, bytes], Set[bytes]] = {}
    
    def initialize_splits_and_pairs(self, word_freq: Counter) -> None:
        """初始化splits和pair_freqs"""
        # 将单词分割成字节序列
        self.splits = {}
        for word, freq in word_freq.items():
            self.splits[word] = [bytes([b]) for b in word]
        
        # 初始化pair_freqs
        self.pair_freqs = {}
        for word, freq in word_freq.items():
            word_pieces = self.splits[word]
            if len(word_pieces) == 1:
                continue
            
            for j in range(len(word_pieces) - 1):
                pair = (word_pieces[j], word_pieces[j + 1])
                self.pair_freqs[pair] = self.pair_freqs.get(pair, 0) + freq

                # 记录pair出现在哪些单词中，构建反向索引
                if pair not in self.pair_to_words:
                    self.pair_to_words[pair] = set()
                self.pair_to_words[pair].add(word)
    
    def find_best_pair(self) -> Tuple[bytes, bytes]:
        """找到频率最高的字节对"""
        return max(self.pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
    
    def update_splits_and_pairs(self, best_pair: Tuple[bytes, bytes], new_token: bytes, word_freq: Counter) -> None:
        """更新splits和pair_freqs"""
        # 哪些词包含best_pair，需要被更新
        # 直接从反向索引中获取
        affected_words = list(self.pair_to_words.get(best_pair, set()))

        # 更新splits
        for word in affected_words:
            word_pieces = self.splits[word]
            i = 0
            while i < len(word_pieces) - 1:
                if word_pieces[i] == best_pair[0] and word_pieces[i + 1] == best_pair[1]:
                    # 如果找到best_pair，合并
                    word_pieces[i] = new_token
                    word_pieces.pop(i + 1)
                    # 更新pair_to_words，如果合并后左侧或右侧还有元素，则会出现新对
                    if i > 0:
                        new_pair_left = (word_pieces[i - 1], new_token)
                        self.pair_to_words.setdefault(new_pair_left, set()).add(word)
                    if i < len(word_pieces) - 1:
                        new_pair_right = (new_token, word_pieces[i + 1])
                        self.pair_to_words.setdefault(new_pair_right, set()).add(word)
                else:
                    i += 1
        
        # 更新pair_freqs
        self._update_pair_freqs(affected_words, best_pair, new_token, word_freq)
    
    def _update_pair_freqs(self, affected_words: List[bytes], best_pair: Tuple[bytes, bytes], 
                          new_token: bytes, word_freq: Counter) -> None:
        """更新pair_freqs字典"""
        for word in affected_words:
            freq = word_freq[word]
            word_pieces = self.splits[word]

            # 删除 best_pair
            if best_pair in self.pair_freqs:
                del self.pair_freqs[best_pair]
                
            # 寻找词中的所有可能受影响的字节对
            for j in range(len(word_pieces) - 1):
                pair = (word_pieces[j], word_pieces[j + 1])
                if pair[0] == new_token and pair[1] == new_token:
                    # new_token为AB 如果pair为AB,AB 只需要增加AB,AB
                    self.pair_freqs[(new_token, new_token)] = self.pair_freqs.get((new_token, new_token), 0) + freq
                elif pair[0] == new_token:
                    # new_token为AB 如果pair为AB,D 需要减少B,D且增加AB,D
                    if (best_pair[1], pair[1]) in self.pair_freqs:
                        self.pair_freqs[(best_pair[1], pair[1])] -= freq
                        if self.pair_freqs[(best_pair[1], pair[1])] <= 0:
                            del self.pair_freqs[(best_pair[1], pair[1])]
                    self.pair_freqs[(new_token, pair[1])] = self.pair_freqs.get((new_token, pair[1]), 0) + freq
                    
                elif pair[1] == new_token:
                    # new_token为AB 如果pair为C,AB 需要减少C,A且增加C,AB
                    if (pair[0], best_pair[0]) in self.pair_freqs:
                        self.pair_freqs[(pair[0], best_pair[0])] -= freq
                        if self.pair_freqs[(pair[0], best_pair[0])] <= 0:
                            del self.pair_freqs[(pair[0], best_pair[0])]
                    self.pair_freqs[(pair[0], new_token)] = self.pair_freqs.get((pair[0], new_token), 0) + freq
    
    def add_special_tokens(self) -> None:
        """将特殊token添加到词汇表中"""
        for m, token in enumerate(self.special_tokens):
            self.token_vocab[self.vocab_size - len(self.special_tokens) + m] = token.encode('utf-8')
    
    def train(self, input_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """训练BPE模型"""
        # 读取语料
        docs = self.preprocessor.read_corpus(input_path)
        
        # 预分词，建立词频字典
        word_freq = self.preprocessor.build_word_frequency(docs)
        
        # BPE训练开始
        
        # 初始化词汇表和合并表
        self.token_vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        self.merges = []
        
        # 初始化splits和pair_freqs
        self.initialize_splits_and_pairs(word_freq)
        
        # 执行合并
        for num_merge in range(num_merges):
            if not self.pair_freqs:
                break
            
            # 选择频率最高的pair
            best_pair = self.find_best_pair()
            self.merges.append(best_pair)
            
            # 更新词汇表，添加新的合并token
            new_token = best_pair[0] + best_pair[1]
            self.token_vocab[256 + num_merge] = new_token
            
            # 更新splits和pair_freqs
            self.update_splits_and_pairs(best_pair, new_token, word_freq)
        
        # 添加特殊token到词汇表
        self.add_special_tokens()
        
        return self.token_vocab, self.merges
    
    def to_files(self, vocab_filepath: str, merges_filepath: str) -> None:
        """将训练结果保存到文件"""
        # 保存词汇表
        with open(vocab_filepath, 'wb') as f:
            # 写入词汇表大小
            f.write(len(self.token_vocab).to_bytes(4, byteorder='little'))
            
            # 写入每个token: <id(4字节)><长度(4字节)><token内容(bytes)>
            for token_id, token in self.token_vocab.items():
                f.write(token_id.to_bytes(4, byteorder='little'))
                f.write(len(token).to_bytes(4, byteorder='little'))
                f.write(token)
        
        # 保存合并规则
        with open(merges_filepath, 'wb') as f:
            # 写入合并规则数量
            f.write(len(self.merges).to_bytes(4, byteorder='little'))
            
            # 写入每个合并规则: <第一部分长度(4字节)><第一部分内容(bytes)><第二部分长度(4字节)><第二部分内容(bytes)>
            for first, second in self.merges:
                f.write(len(first).to_bytes(4, byteorder='little'))
                f.write(first)
                f.write(len(second).to_bytes(4, byteorder='little'))
                f.write(second)
        

# 保持原有函数接口不变
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    使用BPE算法训练tokenizer
    
    Args:
        input_path: 语料文件路径
        vocab_size: 词汇表大小
        special_tokens: 特殊token列表
    
    Returns:
        token_vocab: token词汇表
        merges: 合并操作列表
    """
    tokenizer = BPETrainer(vocab_size, special_tokens)
    return tokenizer.train(input_path)

if __name__ == "__main__":
    # # Example usage
    # input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    # vocab_size = 1000
    # special_tokens = ["<|endoftext|>"]
    # import cProfile
    # cProfile.run('train_bpe(input_path, vocab_size, special_tokens)', 'tokenizer_stats')
    # import pstats
    # from pstats import SortKey

    # # 分析完成后
    # p = pstats.Stats('tokenizer_stats')
    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)  # 显示累计时间最长的20个函数

    # Example usage
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    trainer = BPETrainer(vocab_size, special_tokens)
    # token_vocab, merges = trainer.train(input_path)
    cProfile.run('trainer.train(input_path)', 'tokenizer_stats') 
    p = pstats.Stats('tokenizer_stats')
    p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
    # trainer.to_files("./data/output/token_vocab.bin", "./data/output/merges.bin")

    # print("Token Vocabulary:")
    # for idx, token in token_vocab.items():
    #     if isinstance(token, bytes):
    #         try:
    #             decoded = token.decode('utf-8')
    #             print(f"{idx}: {decoded}")
    #         except UnicodeDecodeError:
    #             print(f"{idx}: {list(token)}")  # 以字节列表形式显示
    #     else:
    #         print(f"{idx}: {token}")
    
    # print("\nMerges:")
    # for merge in merges:
    #     try:
    #         first = merge[0].decode('utf-8')
    #     except UnicodeDecodeError:
    #         first = list(merge[0])
        
    #     try:
    #         second = merge[1].decode('utf-8')
    #     except UnicodeDecodeError:
    #         second = list(merge[1])
        
    #     print(f"{first}{second}")