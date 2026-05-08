//! VectorDB - ONNX-powered semantic code search
//!
//! Uses the all-MiniLM-L6-v2 sentence transformer model via ONNX Runtime
//! for dense 384-dimensional semantic embeddings with cosine similarity search.
//!
//! FIXES APPLIED:
//! - Replaced TF-IDF with real ONNX embedding model using onnxruntime 0.0.14 + tokenizers 0.15.
//! - EmbeddingModel wraps a 'static Session (via Box::leak) behind Arc<Mutex<>>.
//! - embed() tokenizes text, runs ONNX inference with Array2<i64> inputs, mean-pools
//!   token-level hidden states weighted by attention mask, and L2-normalizes.
//! - query() uses cosine similarity against the dense ONNX embeddings.
//! - Removed all TF-IDF logic (compute_tfidf, term_frequency, tokenize, idf map,
//!   total_docs counter, initialize_tfidf).

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{info, warn};

// ONNX Runtime — version 0.0.14 API
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

// ndarray — version 0.15
use ndarray::{Array2, Axis, IxDyn};

// HuggingFace tokenizers — version 0.15
use tokenizers::Tokenizer;

// ============================================================================
// EmbeddingModel
// ============================================================================

/// ONNX-based embedding model using all-MiniLM-L6-v2 architecture.
///
/// The `Environment` is leaked to `'static` so that `Session<'static>` can be
/// stored in `Arc<Mutex<>>` for shared multithreaded access at inference time.
///
/// Produces 384-dimensional dense embeddings with:
/// 1. Tokenize (WordPiece, add special tokens CLS/SEP).
/// 2. ONNX forward pass → last_hidden_state [1, seq_len, 384].
/// 3. Mean-pool token vectors weighted by the attention mask.
/// 4. L2-normalize so cosine similarity = dot product.
pub struct EmbeddingModel {
    session: Arc<Mutex<Session<'static>>>,
    tokenizer: Arc<Tokenizer>,
    embedding_dim: usize,
}

impl EmbeddingModel {
    /// Load the ONNX model and tokenizer from file paths.
    ///
    /// * `model_path` — Path to the exported all-MiniLM-L6-v2 .onnx file.
    /// * `tokenizer_path` — Path to the tokenizer.json file.
    /// * `embedding_dim` — Output dimension (384 for MiniLM-L6-v2).
    pub fn new(model_path: &Path, tokenizer_path: &Path, embedding_dim: usize) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("warden-embeddings")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .map_err(|e| anyhow!("Failed to create ONNX environment: {}", e))?;

        // The Session borrows from Environment — leak Environment to 'static
        // so we can store Session<'static> in Arc<Mutex<>> for shared access.
        // The model path must be owned (PathBuf) because Session<'static>
        // requires all borrowed data to outlive 'static.
        let env_static: &'static Environment = Box::leak(Box::new(environment));
        let model_path_owned = model_path.to_path_buf();

        let session = env_static
            .new_session_builder()
            .map_err(|e| anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .map_err(|e| anyhow!("Failed to set optimization level: {}", e))?
            .with_number_threads(1)
            .map_err(|e| anyhow!("Failed to set thread count: {}", e))?
            .with_model_from_file(model_path_owned)
            .map_err(|e| {
                anyhow!(
                    "Failed to load ONNX model from {}: {}",
                    model_path.display(),
                    e
                )
            })?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            )
        })?;

        info!(
            "ONNX embedding model loaded ({}), dim={}",
            model_path.display(),
            embedding_dim
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            embedding_dim,
        })
    }

    /// Embed a single text string into a normalized dense vector.
    ///
    /// Pipeline:
    /// 1. Tokenize with special tokens (CLS/SEP). Truncate to 512 tokens (MiniLM max).
    /// 2. Build ONNX input tensors: input_ids [1 × seq_len] i64, attention_mask same.
    /// 3. Run inference → last_hidden_state [1 × seq_len × dim] f32.
    /// 4. Mean-pool hidden states weighted by the attention mask.
    /// 5. L2-normalize the pooled embedding.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Bail early on empty / whitespace-only text
        if text.trim().is_empty() {
            return Ok(vec![0.0f32; self.embedding_dim]);
        }

        // ── 1. Tokenize ────────────────────────────────────────────────
        let encoding = self
            .tokenizer
            .encode(text, true) // add_special_tokens = true
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let all_token_ids: Vec<i64> =
            encoding.get_ids().iter().map(|&id| id as i64).collect();
        let all_attn_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        // Truncate to MiniLM max position (512).  Most code chunks are well
        // under this limit; the truncation guards against pathological inputs.
        const MAX_LEN: usize = 512;
        let original_len = all_token_ids.len();
        let seq_len = original_len.min(MAX_LEN);
        if seq_len == 0 {
            return Ok(vec![0.0f32; self.embedding_dim]);
        }
        if original_len > MAX_LEN {
            warn!(
                original = original_len,
                truncated = MAX_LEN,
                "Text truncated for embedding — semantic fidelity may be reduced"
            );
        }

        let token_ids = &all_token_ids[..seq_len];
        let attention_mask = &all_attn_mask[..seq_len];

        // ── 2. Build input tensors ─────────────────────────────────────
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), token_ids.to_vec())
            .map_err(|e| anyhow!("Shape error for input_ids: {}", e))?;

        let attention_mask_arr =
            Array2::from_shape_vec((1, seq_len), attention_mask.to_vec())
                .map_err(|e| anyhow!("Shape error for attention_mask: {}", e))?;

        // ── 3. ONNX inference ──────────────────────────────────────────
        // onnxruntime 0.0.14: session.run() takes Vec<Array<TIn, D>> and
        // returns Vec<OrtOwnedTensor<TOut, IxDyn>> (which derefs to ArrayView).
        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow!("Session lock error: {}", e))?;

        let outputs: Vec<OrtOwnedTensor<f32, IxDyn>> = session
            .run(vec![input_ids_arr, attention_mask_arr])
            .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;

        // onnxruntime 0.0.14 doesn't expose output names through the session,
        // so we use positional indexing.  Most MiniLM ONNX exports have a
        // single output (last_hidden_state) at index 0.
        if outputs.is_empty() {
            return Err(anyhow!("ONNX model returned no outputs"));
        }

        // ── 4. Mean-pool with attention mask ───────────────────────────
        // outputs[0] derefs to ArrayView<f32, IxDyn> with shape [1, seq_len, dim].
        let hidden = &outputs[0];
        let hidden_shape = hidden.shape();

        if hidden_shape.len() != 3 || hidden_shape[0] != 1 || hidden_shape[1] != seq_len {
            return Err(anyhow!(
                "Unexpected output shape {:?}, expected [1, {}, {}]",
                hidden_shape,
                seq_len,
                self.embedding_dim
            ));
        }

        let hidden_dim = hidden_shape[2];

        // Reshape to [seq_len, dim] for easier manipulation.
        // hidden is &ArrayView<f32, IxDyn> — index_axis gives us a subview.
        let hidden_2d = hidden.index_axis(Axis(0), 0); // shape [seq_len, dim]

        // Build an expanded mask: [seq_len] → [seq_len, 1]
        let mask: Vec<f32> = attention_mask.iter().map(|&m| m as f32).collect();
        let mask_sum: f32 = mask.iter().sum();

        // Weighted sum over tokens: (hidden_2d * mask_broadcast).sum_axis(Axis(0))
        let mut weighted_sum = vec![0.0f32; hidden_dim];
        for (tok_idx, &m_val) in mask.iter().enumerate() {
            let row = hidden_2d.index_axis(Axis(0), tok_idx);
            for (d, &val) in row.iter().enumerate() {
                weighted_sum[d] += val * m_val;
            }
        }

        let mut vector = if mask_sum > 1e-10 {
            weighted_sum.iter().map(|v| v / mask_sum).collect()
        } else {
            weighted_sum
        };

        // ── 5. L2 normalization ─────────────────────────────────────────
        let magnitude: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 1e-10 {
            for v in &mut vector {
                *v /= magnitude;
            }
        }

        Ok(vector)
    }

    /// Return the embedding dimension.
    pub fn dim(&self) -> usize {
        self.embedding_dim
    }
}

// ============================================================================
// Data types
// ============================================================================

/// Represents the type of a code chunk
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Function,
    Class,
    Module,
    #[allow(dead_code)]
    Comment,
}

/// A single chunk of code with its ONNX embedding
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub chunk_text: String,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: ChunkType,
    pub embedding: Vec<f32>,
}

/// A fully indexed document with all its code chunks
#[derive(Debug, Clone)]
pub struct IndexedDocument {
    #[allow(dead_code)]
    pub file_path: String,
    pub chunks: Vec<CodeChunk>,
}

/// A semantic search result returned by query()
#[derive(Debug, Clone)]
pub struct SemanticSearchResult {
    pub file_path: String,
    pub chunk_text: String,
    pub start_line: usize,
    #[allow(dead_code)]
    pub end_line: usize,
    pub chunk_type: ChunkType,
    pub relevance_score: f32,
}

// ============================================================================
// VectorDB
// ============================================================================

/// Vector database using ONNX dense embeddings with cosine similarity.
pub struct VectorDB {
    documents: Arc<RwLock<HashMap<String, IndexedDocument>>>,
    model: Option<Arc<EmbeddingModel>>,
    initialized: bool,
    #[allow(dead_code)]
    index_path: PathBuf,
    chunk_count: Arc<AtomicUsize>,
}

impl Clone for VectorDB {
    fn clone(&self) -> Self {
        Self {
            documents: self.documents.clone(),
            model: self.model.clone(),
            initialized: self.initialized,
            index_path: self.index_path.clone(),
            chunk_count: self.chunk_count.clone(),
        }
    }
}

impl VectorDB {
    pub fn new(index_path: PathBuf) -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            model: None,
            initialized: false,
            index_path,
            chunk_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Initialize the VectorDB with an ONNX embedding model.
    ///
    /// * `model_path` — Path to the exported all-MiniLM-L6-v2 .onnx file.
    /// * `tokenizer_path` — Path to the tokenizer.json file.
    pub fn initialize(&mut self, model_path: &Path, tokenizer_path: &Path) -> Result<()> {
        let model = EmbeddingModel::new(model_path, tokenizer_path, 384)?;
        self.model = Some(Arc::new(model));
        self.initialized = true;
        info!("VectorDB initialized with ONNX embedding model (all-MiniLM-L6-v2)");
        Ok(())
    }

    /// Check if the VectorDB has been initialized with a model.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Index a single file, splitting it into logical chunks and embedding each.
    pub async fn index_file(&self, file_path: &Path, content: &str) -> Result<()> {
        let path_str = file_path.to_string_lossy().to_string();

        if content.trim().is_empty() {
            return Ok(());
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("VectorDB not initialized with embedding model"))?;

        let chunks = self.chunk_content(content, &path_str);

        if chunks.is_empty() {
            return Ok(());
        }

        // Embed each chunk with the ONNX model
        let mut chunks_with_embeddings = Vec::with_capacity(chunks.len());
        for (chunk_text, start_line, end_line, chunk_type) in chunks {
            let embedding = model.embed(&chunk_text).unwrap_or_else(|e| {
                warn!(
                    file = %path_str,
                    lines = format!("{}-{}", start_line, end_line),
                    error = %e,
                    "Embedding failed for chunk, using zero vector"
                );
                vec![0.0f32; model.dim()]
            });

            chunks_with_embeddings.push(CodeChunk {
                file_path: path_str.clone(),
                chunk_text,
                start_line,
                end_line,
                chunk_type,
                embedding,
            });
        }

        let chunk_count = chunks_with_embeddings.len();
        let mut docs = self.documents.write().await;
        
        // If file was previously indexed, subtract old chunk count
        let old_chunk_count = if let Some(old_doc) = docs.get(&path_str) {
            old_doc.chunks.len()
        } else {
            0
        };
        if old_chunk_count > 0 {
            self.chunk_count.fetch_sub(old_chunk_count, Ordering::SeqCst);
        }
        
        docs.insert(
            path_str.clone(),
            IndexedDocument {
                file_path: path_str,
                chunks: chunks_with_embeddings,
            },
        );
        
        self.chunk_count.fetch_add(chunk_count, Ordering::SeqCst);

        Ok(())
    }

    /// Query the vector database for semantically similar chunks.
    ///
    /// Embeds the query text with the ONNX model and computes cosine similarity
    /// against all indexed chunk embeddings.
    pub async fn query(&self, query_text: &str, limit: usize) -> Result<Vec<SemanticSearchResult>> {
        if query_text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("VectorDB not initialized with embedding model"))?;

        // Embed the query
        let query_vec = model.embed(query_text)?;

        let docs = self.documents.read().await;

        let mut results: Vec<SemanticSearchResult> = Vec::new();

        for doc in docs.values() {
            for chunk in &doc.chunks {
                let similarity = cosine_similarity(&query_vec, &chunk.embedding);
                if similarity > 0.01 {
                    results.push(SemanticSearchResult {
                        file_path: chunk.file_path.clone(),
                        chunk_text: chunk.chunk_text.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        chunk_type: chunk.chunk_type.clone(),
                        relevance_score: similarity,
                    });
                }
            }
        }

        // Sort by relevance score descending
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);

        info!(
            query_len = query_text.len(),
            results = results.len(),
            "VectorDB ONNX query complete"
        );

        Ok(results)
    }

    /// Index multiple files in batch.
    #[allow(dead_code)]
    pub async fn index_batch(&self, files: Vec<(PathBuf, String)>) -> Result<usize> {
        let mut indexed = 0;
        for (file_path, content) in files {
            if self.index_file(&file_path, &content).await.is_ok() {
                indexed += 1;
            }
        }
        Ok(indexed)
    }

    /// Get list of all indexed file paths.
    #[allow(dead_code)]
    pub async fn get_indexed_files(&self) -> Vec<String> {
        self.documents.read().await.keys().cloned().collect()
    }

    /// Clear all indexed documents.
    #[allow(dead_code)]
    pub async fn clear(&self) {
        self.documents.write().await.clear();
        self.chunk_count.store(0, Ordering::SeqCst);
    }

    /// Get the number of chunks across all documents (sync, via atomic counter).
    pub fn len(&self) -> usize {
        self.chunk_count.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ============ Internal methods ============

    /// Split source content into logical chunks (function, class, module boundaries).
    ///
    /// TODO: Replace brace-counting with Tree-sitter AST-based chunking.
    /// The current approach fails for Python (indentation-based scoping),
    /// languages with braces in strings/comments, and nested functions/closures.
    /// TreeSitterParser::extract_skeleton can identify function/class boundaries
    /// for supported languages; chunking should align to those boundaries.
    fn chunk_content(
        &self,
        content: &str,
        _file_path: &str,
    ) -> Vec<(String, usize, usize, ChunkType)> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_start = 0;
        let mut current_type = ChunkType::Module;
        let mut in_function = false;
        let mut brace_depth = 0i32;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Detect chunk boundaries
            let is_function_start = trimmed.starts_with("fn ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("function ")
                || trimmed.starts_with("async def ")
                || trimmed.starts_with("async function ")
                || ((trimmed.starts_with("const ") || trimmed.starts_with("let ") || trimmed.starts_with("export const ") || trimmed.starts_with("export let "))
                    && trimmed.contains("=>")
                    && (trimmed.contains("= (") || trimmed.contains("= async (") || trimmed.ends_with("=> {")));

            let is_class_start = trimmed.starts_with("class ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("interface ")
                || trimmed.starts_with("impl ");

            if (is_function_start || is_class_start) && !in_function {
                // Save previous chunk if it has content
                if i > current_start && current_start < lines.len() {
                    let chunk_text = lines[current_start..i].join("\n");
                    if !chunk_text.trim().is_empty() {
                        chunks.push((
                            chunk_text,
                            current_start + 1,
                            i + 1,
                            current_type.clone(),
                        ));
                    }
                }
                current_start = i;
                current_type = if is_class_start {
                    ChunkType::Class
                } else {
                    ChunkType::Function
                };
                in_function = true;
            }

            // Track brace depth (simplified)
            brace_depth += trimmed.matches('{').count() as i32;
            brace_depth -= trimmed.matches('}').count() as i32;

            if in_function && brace_depth <= 0 && i > current_start {
                if i > current_start && current_start < lines.len() {
                    let chunk_text = lines[current_start..=i].join("\n");
                    if !chunk_text.trim().is_empty() {
                        chunks.push((
                            chunk_text,
                            current_start + 1,
                            i + 1,
                            current_type.clone(),
                        ));
                    }
                }
                current_start = i + 1;
                in_function = false;
                current_type = ChunkType::Module;
            }
        }

        // Add remaining content as final chunk
        if current_start < lines.len() {
            let chunk_text = lines[current_start..].join("\n");
            if !chunk_text.trim().is_empty() {
                chunks.push((chunk_text, current_start + 1, lines.len(), current_type));
            }
        }

        // If no chunks were created, use the entire file as one chunk
        if chunks.is_empty() {
            chunks.push((content.to_string(), 1, lines.len(), ChunkType::Module));
        }

        chunks
    }
}

// ============================================================================
// Utility: cosine similarity
// ============================================================================

/// Compute cosine similarity between two equal-length float vectors.
///
/// Returns a value in [-1.0, 1.0]. For L2-normalized vectors this reduces to
/// the dot product.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let len = a.len().min(b.len());

    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    let denominator = mag_a.sqrt() * mag_b.sqrt();
    if denominator < 1e-10 {
        return 0.0;
    }

    (dot / denominator).clamp(-1.0, 1.0)
}
