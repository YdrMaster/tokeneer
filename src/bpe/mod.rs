﻿mod algorithm;

use crate::{utok, Method};
use std::{
    collections::{HashMap, HashSet},
    iter::zip,
    ops::Deref,
    pin::Pin,
    ptr::NonNull,
};

pub struct Bpe {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    _vocab: Pin<Box<[u8]>>,
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
        // 遍历文件，标记所有词汇的位置并记录最大长度
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
        let mut bytes = Box::new([unk; 256]);
        let mut total_len = 0;
        // 收集词表字符内容和字节 token，同时计算内容总长度
        let vocabs = zip(vocabs, is_byte)
            .enumerate()
            .map(|(i, (piece, is_byte))| {
                let piece = if is_byte {
                    const BYTES: [u8; 256] = {
                        let mut bytes = [0u8; 256];
                        let mut i = 0usize;
                        while i < 256 {
                            bytes[i] = i as _;
                            i += 1;
                        }
                        bytes
                    };

                    let b = crate::as_byte_token(piece.as_bytes()).unwrap() as usize;
                    bytes[b] = i as utok;
                    std::slice::from_ref(&BYTES[b])
                } else {
                    piece.as_bytes()
                };
                total_len += piece.len();
                piece
            })
            .collect::<Vec<_>>();
        // 创建字符内容缓存
        let mut slices = vec![(0usize, 0usize); vocabs.len()];
        let mut text_buf = Vec::<u8>::with_capacity(total_len);
        let mut indices = (0..vocabs.len()).collect::<Vec<_>>();
        // 对词按内容长度从长到短排序，因为短的内容有可能是长内容的子串，可以避免重复存储相同内容
        indices.sort_unstable_by_key(|&i| -(vocabs[i].len() as isize));
        for i in indices {
            let v = vocabs[i];
            // 查找子串，若存在则复用，否则将新的内容追加到缓存
            let off = memchr::memmem::find(&text_buf, v).unwrap_or_else(|| {
                let off = text_buf.len();
                text_buf.extend(v);
                off
            });
            slices[i] = (off, v.len());
        }
        // 锁定字符串内容的位置，以实现安全的自引用
        let _vocab = unsafe { Pin::new_unchecked(text_buf.into_boxed_slice()) };
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
                ptr: unsafe { NonNull::new_unchecked(_vocab[off..].as_ptr().cast_mut()) },
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
        //     "Building BPE vocab, detected {} tokens, compressed to {} bytes from {len} bytes",
        //     tokens.len(),
        //     _vocab.len(),
        // );

        Self {
            _vocab,
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
        let mut tokenizer = self.build_tokenizer(text);
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

#[test]
fn test() {
    if let Ok(buf) = std::fs::read("tokenizer.model") {
        let bpe = Bpe::from_tokenizer_model(&buf);
        let inaccessible = bpe.inaccessible();
        println!(
            "bpe: detected {} tokens, compressed to {} bytes",
            bpe.vocab_size(),
            bpe._vocab.len(),
        );
        println!("inaccessible: {inaccessible:#?}");
    }
}
