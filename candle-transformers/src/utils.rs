//! Apply penalty and repeat_kv

use candle::{DType, Result, Tensor};
use std::sync::OnceLock;
use std::time::Instant;

fn penalty_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = |key: &str| std::env::var(key).ok().as_deref() == Some("1");
        enabled("KOHARU_LLM_PROFILE") || enabled("CANDLE_PROFILE_SAMPLING")
    })
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let profile = penalty_profile_enabled();
    let logits = logits.to_dtype(DType::F32)?;
    let to_vec_start = Instant::now();
    let mut logits = logits.to_vec1::<f32>()?;
    let to_vec_dt = to_vec_start.elapsed();
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    let from_vec_start = Instant::now();
    let out = Tensor::from_vec(logits, logits_len, device)?;
    let from_vec_dt = from_vec_start.elapsed();
    if profile {
        tracing::debug!(
            "profile: repeat_penalty d2h {:.3}ms h2d {:.3}ms",
            to_vec_dt.as_secs_f64() * 1000.0,
            from_vec_dt.as_secs_f64() * 1000.0
        );
    }
    Ok(out)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        let xs = xs.unsqueeze(2)?;
        let xs = xs.broadcast_as((b_sz, n_kv_head, n_rep, seq_len, head_dim))?;
        xs.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
