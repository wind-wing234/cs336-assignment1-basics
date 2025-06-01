from typing import Dict, Iterable, List, Tuple
from .pre_tokenizer import PreTokenizer

class BPETokenizer:
    def __init__(self, vocab:Dict[int, bytes], merges:List[Tuple[bytes, bytes]], special_tokens:List[str] = None) -> None:
        """
        初始化BPE分词器
        """
        self.token_vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id: Dict[bytes,int] = {token: idx for idx, token in self.token_vocab.items()}
        self.pre_tokenizer = PreTokenizer(self.special_tokens)

        self.word_to_ids: Dict[bytes, List[int]] = {} # 缓存已经计算过的词的对应id序列
    
    @classmethod
    def from_files(cls, vocab_filepath: str, mergers_filepath: str, special_tokens: List[str] = None) -> 'BPETokenizer':
        """
        从文件加载BPE分词器
        
        Args:
            vocab_filepath: 词汇表文件路径
            mergers_filepath: 合并规则文件路径
            special_tokens: 特殊token列表
        
        Returns:
            BPETokenizer 实例
        """
        # 读取vocab
        vocab = {}
        with open(vocab_filepath, 'rb') as f:
            # 读取词汇表大小
            vocab_size_bytes = f.read(4)
            vocab_size = int.from_bytes(vocab_size_bytes, byteorder='little')
            
            # 读取每个token: <id(4字节)><长度(4字节)><token内容(bytes)>
            for _ in range(vocab_size):
                token_id_bytes = f.read(4)
                token_id = int.from_bytes(token_id_bytes, byteorder='little')
                
                token_len_bytes = f.read(4)
                token_len = int.from_bytes(token_len_bytes, byteorder='little')
                
                token = f.read(token_len)
                vocab[token_id] = token
        
        # 读取merges
        merges = []
        with open(mergers_filepath, 'rb') as f:
            # 读取合并规则数量
            merges_count_bytes = f.read(4)
            merges_count = int.from_bytes(merges_count_bytes, byteorder='little')
            
            # 读取每个合并规则: <第一部分长度(4字节)><第一部分内容(bytes)><第二部分长度(4字节)><第二部分内容(bytes)>
            for _ in range(merges_count):
                first_len_bytes = f.read(4)
                first_len = int.from_bytes(first_len_bytes, byteorder='little')
                
                first = f.read(first_len)
                
                second_len_bytes = f.read(4)
                second_len = int.from_bytes(second_len_bytes, byteorder='little')
                
                second = f.read(second_len)
                
                merges.append((first, second))
        
        return cls(vocab, merges, special_tokens)

    def calculate_token_ids(self, word: bytes) -> List[int]:
        """
        将一个bytes根据merges不断合并，得到其token ID序列
        """
        token_ids = []
        # 将每个字节作为独立的bytes对象
        bytes_list = [bytes([b]) for b in word]  # 将每个字节作为单独的bytes对象

        while len(bytes_list) > 1:
            # 一轮中可能同时满足多个合并规则，选择index最小的合并规则进行合并
            min_rule_idx = None
            min_merge_pos = None
            
            # 遍历当前字节列表中所有可能的合并规则
            for i, pair in enumerate(zip(bytes_list[:-1], bytes_list[1:])):
            # for i in range(len(bytes_list) - 1):
                # pair = bytes_list[i] + bytes_list[i + 1]
                idx = self.token_to_id.get(pair[0] + pair[1])
                if (idx is not None) and ((min_rule_idx is None) or (idx < min_rule_idx)):
                    # 找到一个更小的合并规则，更新最小index和位置
                    min_rule_idx = idx
                    min_merge_pos = i
            
            if min_rule_idx is None:
                # 没有可合并的规则
                break
            
            # 执行合并
            bytes_list[min_merge_pos:min_merge_pos + 2] = [bytes_list[min_merge_pos] + bytes_list[min_merge_pos + 1]]
        
        # 出循环说明已经合并完成，开始翻译为ids
        for part in bytes_list:
            try:
                id = self.token_to_id[part]
                token_ids.append(id)
            except KeyError:
                # 如果没有找到对应的ID，可能是未训练的token,暂时不处理
                print(f"Warning: Token {part} not found in vocabulary.")
                pass
        return token_ids
    
    def encode(self, text:str) -> List[int]:
        """
        将文本编码为BPE token ID列表
        """
        # 预分词，把str转为list[bytes]
        words = self.pre_tokenizer.pretokenize(text) # word是bytes
        ids = []
        for word in words:
            if word in self.token_to_id:
                # 如果是特殊token/其他单token，直接返回对应的ID
                ids.append(self.token_to_id[word])
            elif word in self.word_to_ids:
                # 如果已经计算过这个词，直接使用缓存
                ids.extend(self.word_to_ids[word])
            else:
                # 计算该词对应token ID序列
                token_ids = self.calculate_token_ids(word)
                self.word_to_ids[word] = token_ids  # 缓存结果
                ids.extend(token_ids)
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        对可迭代对象（例如文件句柄）中的每个文本进行编码，每次调用返回一个token ID
        """
        words_iter = self.pre_tokenizer.pretokenize_iter(iterable)
        for word in words_iter:
            if word in self.token_to_id:
                # 如果是特殊token/其他单token，直接返回对应的ID
                yield self.token_to_id[word]
            elif word in self.word_to_ids:
                # 如果已经计算过这个词，直接使用缓存
                yield from self.word_to_ids[word]
            else:
                # 计算该词对应token ID序列
                token_ids = self.calculate_token_ids(word)
                self.word_to_ids[word] = token_ids
                yield from token_ids

    def decode(self, ids: List[int]) -> str:
        """
        将BPE token ID列表解码为文本
        """
        text_bytes = b""
        for id in ids:
            if id in self.token_vocab:
                text_bytes += self.token_vocab[id]
            else:
                print(f"Warning: ID {id} not found in vocabulary.")
                continue
        return text_bytes.decode('utf-8', errors='ignore')

if __name__ == "__main__":
    # Example usage
    special_tokens = ["<|endoftext|>"]
    vocab_path = "./data/output/token_vocab.bin"
    mergers_path = "./data/output/merges.bin"
    # input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    tokenizer = BPETokenizer.from_files(vocab_path, mergers_path, special_tokens)

    # with open(input_path, 'r', encoding='utf-8') as f:
    #     text = f.read()

    # encoded_ids = tokenizer.encode(text)
    # print("Encoded IDs:", encoded_ids[:1000])

    # decoded_text = tokenizer.decode(encoded_ids[:1000])
    # print("Decoded Text:", decoded_text)

    # 迭代器使用
    token_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for id in tokenizer.encode_iterable(f):
            print(id, end=' ')
            token_list.append(id)
            if id == 999:
                break
    decoded_text = tokenizer.decode(token_list)
    print("\nDecoded Text:", decoded_text)
            