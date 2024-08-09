//! b-p-e for Byte Pair Encoding

mod algorithm;

use crate::{
    utok,
    vocab::{CollectedVocab, CompressedVocab},
    Method,
};
use std::{
    collections::{HashMap, HashSet},
    iter::zip,
    ops::Deref,
    pin::Pin,
    ptr::NonNull,
};

pub struct Bpe {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    _vocabs: Pin<Box<[u8]>>,
    /// 按 token 顺序保存元信息
    tokens: Box<[TokenMeta]>,
    /// 按字符串的字典序排序的 token 索引，用于从字符串二分查找 token。
    /// 建立索引时直接剔除了不可能从 piece 构造的所有单字节
    sorted_pieces: Box<[utok]>,
    /// 用于索引单字节 token，因此不需要其他元信息
    bytes: Box<[utok; 256]>,
    /// token: <unk>
    unk: utok,
}

struct TokenMeta {
    /// 指向字符串内容的指针
    ptr: NonNull<u8>,
    /// 字符串长度
    len: u32,
    /// 字符串的合并排名，从 0 开始
    rank: u32,
}

// SAFETY: TokenMeta 中的指针是指向 Bpe 内容的自引用指针，且仅用于不可变引用。
unsafe impl Send for TokenMeta {}
unsafe impl Sync for TokenMeta {}

impl Deref for TokenMeta {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as _) }
    }
}

impl Bpe {
    /// 解析 tokenizer.model 文件并构造一个 bpe 分词器。
    pub fn from_tokenizer_model(model: &[u8]) -> Self {
        // 遍历文件，标记所有词汇的位置
        let offsets = (0..)
            .scan(0usize, |offset, _| match &model[*offset..] {
                [10, total_len, 10, content @ ..] => {
                    let total_len = *total_len as usize;
                    *offset += total_len + 2;
                    Some(&content[..total_len - 2])
                }
                [..] => None,
            })
            .collect::<Vec<_>>();
        // 产生词迭代器
        let vocabs = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            std::str::from_utf8(&slice[1..][..len]).unwrap()
        });
        // 产生评分迭代器
        let scores = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            let ptr = slice[len + 2..].as_ptr().cast::<f32>();
            unsafe { ptr.read_unaligned() }
        });
        // 产生字节标记迭代器
        let mut i = 0;
        let is_byte = std::iter::from_fn(|| {
            if i < 3 {
                i += 1;
                Some(false)
            } else if i < 3 + 256 {
                i += 1;
                Some(true)
            } else {
                Some(false)
            }
        });
        // 构造分词器
        Self::new(vocabs, scores, is_byte, 0)
    }

    pub fn new<'a>(
        vocabs: impl IntoIterator<Item = &'a str>,
        scores: impl IntoIterator<Item = f32>,
        is_byte: impl IntoIterator<Item = bool>,
        unk: utok,
    ) -> Self {
        let CollectedVocab {
            vocabs,
            total_len,
            bytes,
        } = CollectedVocab::collect_with_hint(
            vocabs.into_iter().map(|s| s.as_bytes()),
            is_byte,
            unk,
        );
        let CompressedVocab { vocabs, slices } = CompressedVocab::new(&vocabs, total_len);
        // 收集合词评分
        let scores = scores.into_iter().collect::<Vec<_>>();
        assert_eq!(
            slices.len(),
            scores.len(),
            "scores size mismatch with vocab size"
        );
        // tokens 中直接引用字符串位置，绑定重新赋权并转换为整型的分词评分
        let tokens = zip(slices, rank(&scores))
            .map(|((off, len), rank)| TokenMeta {
                ptr: unsafe { NonNull::new_unchecked(vocabs[off..].as_ptr().cast_mut()) },
                len: len as _,
                rank,
            })
            .collect::<Box<_>>();
        // 对 token 按字符串的字典序排序，用于从字符串二分查找 token
        // <unk> 和 <0xyz> 不应该通过 piece 搜索到，使用 set 排除
        let bytes_set = bytes.iter().chain(&[unk]).cloned().collect::<HashSet<_>>();
        let mut sorted_pieces = (0..tokens.len() as utok)
            .filter(|i| !bytes_set.contains(i))
            .collect::<Box<_>>();
        sorted_pieces.sort_unstable_by_key(|&i| &*tokens[i as usize]);

        // println!(
        //     "Building BPE vocab, detected {} tokens, compressed to {} bytes from {total_len} bytes",
        //     tokens.len(),
        //     vocabs.len(),
        // );

        Self {
            _vocabs: vocabs,
            tokens,
            sorted_pieces,
            bytes,
            unk,
        }
    }

    /// BPE 词表中，并非所有词都是合词规则可达的。此算法可识别“内部不可达”的 token。
    pub fn inaccessible(&self) -> HashMap<&str, utok> {
        self.sorted_pieces
            .iter()
            .filter_map(|&t| {
                let s = unsafe { std::str::from_utf8_unchecked(self.token(t)) };
                if self.encode(s).into_iter().nth(1).is_some() {
                    Some((s, t))
                } else {
                    None
                }
            })
            .collect()
    }

    /// piece -> token
    #[inline]
    fn find_piece(&self, piece: &[u8]) -> Option<utok> {
        match self
            .sorted_pieces
            .binary_search_by_key(&piece, |&i| self.token(i))
        {
            Ok(i) => Some(self.sorted_pieces[i]),
            Err(_) => match *piece {
                [b] => Some(self.bytes[b as usize]),
                [..] => None,
            },
        }
    }

    /// token id -> token meta
    #[inline(always)]
    fn token(&self, token: utok) -> &TokenMeta {
        &self.tokens[token as usize]
    }
}

impl Method for Bpe {
    #[inline]
    fn unk_token(&self) -> utok {
        self.unk
    }
    #[inline]
    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    #[inline]
    fn internal_special(&self) -> impl IntoIterator<Item = (&str, utok)> {
        self.inaccessible()
    }
    #[inline]
    fn encode(&self, text: &str) -> impl IntoIterator<Item = utok> + '_ {
        let mut tokenizer = self.begin_merge(text);
        while tokenizer.merge() {}
        tokenizer.into_iter()
    }
    #[inline]
    fn decode(&self, token: utok) -> &[u8] {
        self.token(token)
    }
}

/// 对一组评分排序、去重并重新赋权，转换为保持相同顺序的整型序列
fn rank(scores: &[f32]) -> impl IntoIterator<Item = u32> + '_ {
    use std::{
        cmp::Ordering,
        collections::{BTreeMap, BTreeSet},
    };

    #[derive(PartialEq, Debug)]
    struct FloatOrd(f32);
    impl Eq for FloatOrd {}
    impl PartialOrd for FloatOrd {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for FloatOrd {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    let map = scores
        // 排序 + 去重
        .iter()
        .copied()
        .map(FloatOrd)
        .collect::<BTreeSet<_>>()
        // 重新赋权
        .into_iter()
        .rev()
        .enumerate()
        .map(|(i, f)| (f, i as u32))
        .collect::<BTreeMap<_, _>>();

    scores.iter().map(move |&f| map[&FloatOrd(f)])
}

#[cfg(test)]
mod bpe_tests {
    use super::*;

    #[test]
    fn test() {
        if let Ok(buf) = std::fs::read("tokenizer.model") {
            let bpe = Bpe::from_tokenizer_model(&buf);
            let inaccessible = bpe.inaccessible();
            println!(
                "bpe: detected {} tokens, compressed to {} bytes",
                bpe.vocab_size(),
                bpe._vocabs.len(),
            );
            println!("inaccessible: {inaccessible:#?}");
        }
    }

    #[test]
    fn test_bpe_new() {
        let vocabs = vec!["a", "b", "c", "ab", "bc"];
        let scores = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let is_byte = vec![false, false, false, false, false];
        let bpe = Bpe::new(vocabs, scores, is_byte, 0);
        assert_eq!(bpe.vocab_size(), 5);
    }

    #[test]
    fn test_bpe_encode() {
        let vocabs = vec!["a", "b", "c", "ab", "bc"];
        let scores = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let is_byte = vec![false, false, false, false, false];
        let bpe = Bpe::new(vocabs, scores, is_byte, 0);

        let encoded: Vec<_> = bpe.encode("abc").into_iter().collect();
        assert_eq!(encoded.len(), 2); // Should merge "ab" and leave "c"
        assert_eq!(encoded[0], 3);    // Assuming "ab" is assigned token ID 3
        assert_eq!(encoded[1], 2);    // Assuming "c" is assigned token ID 2
    }

    #[test]
    fn test_bpe_decode() {
        let vocabs = vec!["a", "b", "c", "ab", "bc"];
        let scores = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let is_byte = vec![false, false, false, false, false];
        let bpe = Bpe::new(vocabs, scores, is_byte, 0);

        assert_eq!(bpe.decode(3), b"ab");
        assert_eq!(bpe.decode(2), b"c");
    }

    #[test]
    fn test_bpe_encode_decode() {
        let vocabs = vec!["a", "b", "c", "ab", "bc"];
        let scores = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let is_byte = vec![false, false, false, false, false];
        let bpe = Bpe::new(vocabs, scores, is_byte, 0);

        let text = "abcbc";
        let encoded: Vec<_> = bpe.encode(text).into_iter().collect();
        let decoded: Vec<u8> = encoded.iter().flat_map(|&t| bpe.decode(t).iter().copied()).collect();
        assert_eq!(String::from_utf8(decoded).unwrap(), text);
    }

    #[test]
    fn test_bpe_unk_token() {
        let vocabs = vec!["a", "b", "c"];
        let scores = vec![1.0, 1.0, 1.0];
        let is_byte = vec![false, false, false];
        let unk_token = 100;
        let bpe = Bpe::new(vocabs, scores, is_byte, unk_token);

        assert_eq!(bpe.unk_token(), unk_token);
    }

    #[test]
    fn test_bpe_inaccessible() {
        let vocabs = vec!["a", "b", "c", "ab", "bcd", "d"];
        let scores = vec![1.0, 1.0, 1.0, 2.0, 1.5, 1.0];
        let is_byte = vec![false, false, false, false, false, false];
        let bpe = Bpe::new(vocabs, scores, is_byte, 0);

        let inaccessible = bpe.inaccessible();
        println!("Inaccessible tokens: {:?}", inaccessible);

        // 'd' is a single character, so it should be accessible
        assert!(!inaccessible.contains_key("d"), "Token 'd' should be accessible");

        // 'bcd' cannot be formed by merging other tokens, so it should be inaccessible
        assert!(inaccessible.contains_key("bcd"), "Token 'bcd' should be inaccessible");

        // 'ab' can be formed by merging 'a' and 'b', so it should be accessible
        assert!(!inaccessible.contains_key("ab"), "Token 'ab' should be accessible");
    }

    #[test]
    fn test_bpe_with_byte_tokens() {
        let vocabs = vec!["a", "b", "<0x41>", "<0x42>"];
        let scores = vec![1.0, 1.0, 1.0, 1.0];
        let is_byte = vec![false, false, true, true];
        let bpe = Bpe::new(vocabs, scores, is_byte, 0);

        let input = "aAB";
        let encoded: Vec<_> = bpe.encode(input).into_iter().collect();
        println!("Input: {:?}", input);
        println!("Encoded tokens: {:?}", encoded);
        println!("Vocabulary size: {}", bpe.vocab_size());

        assert_eq!(encoded.len(), 3, "Expected 3 tokens for input 'aAB'");
    }
}
