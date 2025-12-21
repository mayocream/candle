//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use candle::{DType, Error, Result, Tensor, D};
use rand::{distr::Distribution, SeedableRng};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    // Note that the rng is not used for the Gumbel-Softmax sampling.
    GumbelSoftmax { temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

fn sampling_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = |key: &str| std::env::var(key).ok().as_deref() == Some("1");
        enabled("KOHARU_LLM_PROFILE") || enabled("CANDLE_PROFILE_SAMPLING")
    })
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        logits.argmax(D::Minus1)?.to_scalar::<u32>()
    }

    fn sample_gumbel_softmax(&mut self, logits: &Tensor, temperature: f64) -> Result<u32> {
        let sampled = candle_nn::sampling::gumbel_softmax(logits, temperature, D::Minus1)?;
        sampled.to_scalar::<u32>()
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&mut self, prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    fn ensure_cuda_logits_1d(&self, logits: &Tensor) -> Result<Option<Tensor>> {
        let dims = logits.dims();
        match dims {
            [] => Ok(None),
            [_] => Ok(Some(logits.clone())),
            [1, _] => Ok(Some(logits.squeeze(0)?)),
            _ => Ok(None),
        }
    }

    fn tensor_to_u32(&self, tensor: Tensor) -> Result<u32> {
        tensor.get(0)?.to_scalar::<u32>()
    }

    fn sample_topk_cuda(
        &mut self,
        logits: &Tensor,
        k: usize,
        temperature: f64,
        profile: bool,
    ) -> Result<u32> {
        let vocab = logits.dims1()?;
        let k = k.min(vocab);
        if k == 0 {
            return self.sample_argmax(logits.clone());
        }
        if k >= vocab {
            return self.sample_gumbel_softmax(logits, temperature);
        }
        let logits = logits.contiguous()?;
        let sort_start = Instant::now();
        let (sorted_logits, sorted_indices) = logits.sort_last_dim(false)?;
        let sort_dt = sort_start.elapsed();
        let top_logits = sorted_logits.narrow(D::Minus1, 0, k)?;
        let top_indices = sorted_indices.narrow(D::Minus1, 0, k)?;
        let gumbel_start = Instant::now();
        let sampled = candle_nn::sampling::gumbel_softmax(&top_logits, temperature, D::Minus1)?;
        let gumbel_dt = gumbel_start.elapsed();
        if sampled.elem_count() != 1 {
            return self.sample_argmax(logits);
        }
        let sampled = sampled.reshape(1)?;
        let token = top_indices.index_select(&sampled, D::Minus1)?;
        let to_scalar_start = Instant::now();
        let token = self.tensor_to_u32(token)?;
        let to_scalar_dt = to_scalar_start.elapsed();
        if profile {
            tracing::debug!(
                "profile: sample_topk_cuda sort {:.3}ms gumbel {:.3}ms d2h {:.3}ms",
                sort_dt.as_secs_f64() * 1000.0,
                gumbel_dt.as_secs_f64() * 1000.0,
                to_scalar_dt.as_secs_f64() * 1000.0
            );
        }
        Ok(token)
    }

    fn sample_topp_cuda(
        &mut self,
        logits: &Tensor,
        top_p: f64,
        temperature: f64,
        profile: bool,
    ) -> Result<u32> {
        if top_p <= 0.0 {
            return self.sample_argmax(logits.clone());
        }
        if top_p >= 1.0 {
            return self.sample_gumbel_softmax(logits, temperature);
        }
        let scaled = if temperature == 1.0 {
            logits.clone()
        } else {
            (logits / temperature)?
        };
        let scaled = scaled.to_dtype(DType::F32)?.contiguous()?;
        let sort_start = Instant::now();
        let probs = candle_nn::ops::softmax_last_dim(&scaled)?.contiguous()?;
        let (sorted_probs, sorted_indices) = probs.sort_last_dim(false)?;
        let sort_dt = sort_start.elapsed();
        let mask_start = Instant::now();
        let cumsum = sorted_probs.cumsum(D::Minus1)?;
        let prev_cumsum = cumsum.broadcast_sub(&sorted_probs)?;
        let mask = prev_cumsum.lt(top_p)?;
        let sorted_logits = scaled.gather(&sorted_indices, D::Minus1)?;
        let neg_inf = Tensor::full(
            f32::NEG_INFINITY,
            sorted_logits.shape(),
            sorted_logits.device(),
        )?;
        let filtered_logits = mask.where_cond(&sorted_logits, &neg_inf)?;
        let mask_dt = mask_start.elapsed();
        let gumbel_start = Instant::now();
        let sampled = candle_nn::sampling::gumbel_softmax(&filtered_logits, 1.0, D::Minus1)?;
        let gumbel_dt = gumbel_start.elapsed();
        if sampled.elem_count() != 1 {
            return self.sample_argmax(logits.clone());
        }
        let sampled = sampled.reshape(1)?;
        let token = sorted_indices.index_select(&sampled, D::Minus1)?;
        let to_scalar_start = Instant::now();
        let token = self.tensor_to_u32(token)?;
        let to_scalar_dt = to_scalar_start.elapsed();
        if profile {
            tracing::debug!(
                "profile: sample_topp_cuda sort {:.3}ms mask {:.3}ms gumbel {:.3}ms d2h {:.3}ms",
                sort_dt.as_secs_f64() * 1000.0,
                mask_dt.as_secs_f64() * 1000.0,
                gumbel_dt.as_secs_f64() * 1000.0,
                to_scalar_dt.as_secs_f64() * 1000.0
            );
        }
        Ok(token)
    }

    fn sample_topk_topp_cuda(
        &mut self,
        logits: &Tensor,
        top_k: usize,
        top_p: f64,
        temperature: f64,
        profile: bool,
    ) -> Result<u32> {
        if top_k == 0 {
            return self.sample_argmax(logits.clone());
        }
        let vocab = logits.dims1()?;
        let k = top_k.min(vocab);
        let logits = logits.contiguous()?;
        let sort_start = Instant::now();
        let (sorted_logits, sorted_indices) = logits.sort_last_dim(false)?;
        let sort_dt = sort_start.elapsed();
        let top_logits = sorted_logits.narrow(D::Minus1, 0, k)?;
        let top_indices = sorted_indices.narrow(D::Minus1, 0, k)?;
        if top_p <= 0.0 {
            return top_indices.get(0)?.to_scalar::<u32>();
        }
        if top_p >= 1.0 {
            let gumbel_start = Instant::now();
            let sampled = candle_nn::sampling::gumbel_softmax(&top_logits, temperature, D::Minus1)?;
            let gumbel_dt = gumbel_start.elapsed();
            if sampled.elem_count() != 1 {
                return self.sample_argmax(logits);
            }
            let sampled = sampled.reshape(1)?;
            let token = top_indices.index_select(&sampled, D::Minus1)?;
            let to_scalar_start = Instant::now();
            let token = self.tensor_to_u32(token)?;
            let to_scalar_dt = to_scalar_start.elapsed();
            if profile {
                tracing::debug!(
                    "profile: sample_topk_topp_cuda sort {:.3}ms gumbel {:.3}ms d2h {:.3}ms",
                    sort_dt.as_secs_f64() * 1000.0,
                    gumbel_dt.as_secs_f64() * 1000.0,
                    to_scalar_dt.as_secs_f64() * 1000.0
                );
            }
            return Ok(token);
        }
        let scaled = if temperature == 1.0 {
            top_logits
        } else {
            (&top_logits / temperature)?
        };
        let scaled = scaled.to_dtype(DType::F32)?.contiguous()?;
        let mask_start = Instant::now();
        let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
        let cumsum = probs.cumsum(D::Minus1)?;
        let prev_cumsum = cumsum.broadcast_sub(&probs)?;
        let mask = prev_cumsum.lt(top_p)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scaled.shape(), scaled.device())?;
        let filtered_logits = mask.where_cond(&scaled, &neg_inf)?;
        let mask_dt = mask_start.elapsed();
        let gumbel_start = Instant::now();
        let sampled = candle_nn::sampling::gumbel_softmax(&filtered_logits, 1.0, D::Minus1)?;
        let gumbel_dt = gumbel_start.elapsed();
        if sampled.elem_count() != 1 {
            return self.sample_argmax(logits);
        }
        let sampled = sampled.reshape(1)?;
        let token = top_indices.index_select(&sampled, D::Minus1)?;
        let to_scalar_start = Instant::now();
        let token = self.tensor_to_u32(token)?;
        let to_scalar_dt = to_scalar_start.elapsed();
        if profile {
            tracing::debug!(
                "profile: sample_topk_topp_cuda sort {:.3}ms mask {:.3}ms gumbel {:.3}ms d2h {:.3}ms",
                sort_dt.as_secs_f64() * 1000.0,
                mask_dt.as_secs_f64() * 1000.0,
                gumbel_dt.as_secs_f64() * 1000.0,
                to_scalar_dt.as_secs_f64() * 1000.0
            );
        }
        Ok(token)
    }

    fn sample_cuda(&mut self, logits: &Tensor) -> Result<Option<u32>> {
        if !logits.device().is_cuda() {
            return Ok(None);
        }
        let profile = sampling_profile_enabled();
        let logits = match self.ensure_cuda_logits_1d(logits)? {
            Some(logits) => logits,
            None => return Ok(None),
        };
        match &self.sampling {
            Sampling::TopK { k, temperature } => Ok(Some(self.sample_topk_cuda(
                &logits,
                *k,
                *temperature,
                profile,
            )?)),
            Sampling::TopP { p, temperature } => Ok(Some(self.sample_topp_cuda(
                &logits,
                *p,
                *temperature,
                profile,
            )?)),
            Sampling::TopKThenTopP { k, p, temperature } => Ok(Some(self.sample_topk_topp_cuda(
                &logits,
                *k,
                *p,
                *temperature,
                profile,
            )?)),
            _ => Ok(None),
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        if let Some(token) = self.sample_cuda(logits)? {
            return Ok(token);
        }
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        let profile = sampling_profile_enabled();
        let to_dtype_start = Instant::now();
        let logits = logits.to_dtype(DType::F32)?;
        let to_dtype_dt = to_dtype_start.elapsed();
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let softmax_start = Instant::now();
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let softmax_dt = softmax_start.elapsed();
            let to_vec_start = Instant::now();
            let mut prs = prs.to_vec1()?;
            let to_vec_dt = to_vec_start.elapsed();
            f(&mut prs);
            if profile {
                tracing::debug!(
                    "profile: sample_cpu to_dtype {:.3}ms softmax {:.3}ms d2h {:.3}ms",
                    to_dtype_dt.as_secs_f64() * 1000.0,
                    softmax_dt.as_secs_f64() * 1000.0,
                    to_vec_dt.as_secs_f64() * 1000.0
                );
            }
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::GumbelSoftmax { temperature } => {
                self.sample_gumbel_softmax(&logits, *temperature)?
            }
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}
