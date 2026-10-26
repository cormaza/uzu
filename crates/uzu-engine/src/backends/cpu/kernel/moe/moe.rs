use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::{
            gemm::{GemmBPrologueKind, GemmDTransform},
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        },
        cpu::kernel::activation_transform::hadamard_transform,
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeRouterTopK)]
#[variants(T, f32)]
#[variants(MAX_TOP_K, 8)]
pub fn moe_router_top_k<T: DataTypeRepr + Float, const MAX_TOP_K: u32>(
    logits: *const T,
    #[optional(has_biases)] biases: Option<*const T>,
    expert_ids: *mut u32,
    expert_weights: *mut f32,
    num_experts: u32,
    top_k: u32,
    batch_size: u32,
    logit_stride: u32,
    #[specialize] has_biases: bool,
) {
    let num_experts_usize = num_experts as usize;
    let logit_stride_usize = logit_stride as usize;
    let top_k_usize = top_k as usize;
    for token_idx in 0..batch_size as usize {
        let mut pairs: Vec<(f32, u32)> = (0..num_experts_usize)
            .map(|e| {
                let mut v = unsafe { *logits.add(token_idx * logit_stride_usize + e) }.to_f32().unwrap();
                if has_biases {
                    v += unsafe { *biases.unwrap().add(e) }.to_f32().unwrap();
                }
                if v.is_nan() {
                    v = f32::NEG_INFINITY;
                }
                (v, e as u32)
            })
            .collect();

        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let chosen = &pairs[..top_k_usize];
        let max_val = chosen.iter().map(|&(v, _)| v).fold(f32::NEG_INFINITY, f32::max);
        let mut exps = Vec::with_capacity(top_k_usize);
        let mut sum_exp = 0.0f32;
        for &(v, _) in chosen {
            let ev = if max_val == f32::NEG_INFINITY {
                1.0f32 / (top_k_usize as f32)
            } else {
                (v - max_val).exp()
            };
            exps.push(ev);
            sum_exp += ev;
        }
        let inv_sum = if sum_exp > 0.0 {
            1.0f32 / sum_exp
        } else {
            0.0
        };
        for k in 0..top_k_usize {
            unsafe {
                *expert_ids.add(token_idx * top_k_usize + k) = chosen[k].1;
                *expert_weights.add(token_idx * top_k_usize + k) = exps[k] * inv_sum;
            }
        }
    }
}

fn compute_route_gemv<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    expert_b: *const u8,
    expert_scales: Option<*const T>,
    expert_zero_points: Option<*const u8>,
    expert_biases: Option<*const T>,
    route_a: *const T,
    k_u: usize,
    n_u: usize,
    scale_output_stride: usize,
    scale_group_stride: usize,
    zero_point_output_stride: usize,
    zero_point_group_stride: usize,
    signed_codes: bool,
    b_prologue: GemmBPrologueKind,
) -> Vec<f32> {
    let bits = BITS as usize;
    let group_size = if GROUP_SIZE == 0 {
        1
    } else {
        GROUP_SIZE as usize
    };
    let mut row_out = vec![0.0f32; n_u];
    for (col, out_slot) in row_out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for inner in 0..k_u {
            let a_val = unsafe { *route_a.add(inner) }.to_f32().unwrap();
            let b_val = if BITS == 0 {
                let b_typed = expert_b as *const T;
                unsafe { *b_typed.add(col * k_u + inner) }.to_f32().unwrap()
            } else {
                let pack_factor = 32 / bits;
                let weight_linear_index = col * k_u + inner;
                let word_index = weight_linear_index / pack_factor;
                let code_index_in_word = weight_linear_index % pack_factor;
                let bit_offset = (code_index_in_word * bits) as u32;
                let weights_words = expert_b as *const u32;
                let word = unsafe { weights_words.add(word_index).read_unaligned() };
                let code_mask = (1u32 << bits) - 1;
                let mut weight_code = ((word >> bit_offset) & code_mask) as u8;
                if signed_codes {
                    weight_code ^= 1u8 << (bits - 1);
                }
                let quantized_value = f32::from(weight_code);
                let group_index = inner / group_size;
                let metadata_index = col * scale_output_stride + group_index * scale_group_stride;
                let scale = unsafe { *expert_scales.unwrap().add(metadata_index) }.to_f32().unwrap();
                let midpoint = (1u32 << (bits - 1)) as f32;
                let bias_term = match b_prologue {
                    GemmBPrologueKind::ScaleZeroPointDequant => {
                        let zp_ptr = expert_zero_points.unwrap();
                        let zp_index = col * zero_point_output_stride + group_index * zero_point_group_stride;
                        let zp = if bits == 4 {
                            let byte_index = zp_index / 2;
                            let byte_value = unsafe { *zp_ptr.add(byte_index) };
                            if zp_index.is_multiple_of(2) {
                                (byte_value & 0x0F) as f32
                            } else {
                                ((byte_value >> 4) & 0x0F) as f32
                            }
                        } else {
                            unsafe { *zp_ptr.add(zp_index) as f32 }
                        };
                        -scale * zp
                    },
                    GemmBPrologueKind::ScaleBiasDequant => {
                        unsafe { *expert_biases.unwrap().add(metadata_index) }.to_f32().unwrap()
                    },
                    GemmBPrologueKind::ScaleSymmetricDequant => -scale * midpoint,
                    GemmBPrologueKind::FullPrecision => unreachable!(),
                };
                scale * quantized_value + bias_term
            };
            acc += a_val * b_val;
        }
        *out_slot = acc;
    }
    row_out
}

#[kernel(MoeGemv)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 0, 32, 64)]
#[variants(BITS, 0, 4, 8)]
#[constraint((BITS == 0) == (GROUP_SIZE == 0))]
pub fn moe_gemv<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    b: *const u8,
    #[optional(b_prologue != GemmBPrologueKind::FullPrecision)] scales: Option<*const T>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] zero_points: Option<*const u8>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleBiasDequant)] biases: Option<*const T>,
    #[optional(has_shared)] shared_b: Option<*const u8>,
    #[optional(has_shared && b_prologue != GemmBPrologueKind::FullPrecision)] shared_scales: Option<*const T>,
    #[optional(has_shared && b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] shared_zero_points: Option<
        *const u8,
    >,
    #[optional(has_shared && b_prologue == GemmBPrologueKind::ScaleBiasDequant)] shared_biases: Option<*const T>,
    a: *const T,
    expert_ids: *const u32,
    d: *mut T,
    #[optional(output_transform.contains(GemmDTransform::BIAS))] output_bias: Option<*const T>,
    #[optional(output_transform.contains(GemmDTransform::RHT))] hadamard_factors: Option<*const i32>,
    #[optional(has_shared && output_transform.contains(GemmDTransform::BIAS))] shared_output_bias: Option<*const T>,
    #[optional(has_shared && output_transform.contains(GemmDTransform::RHT))] shared_hadamard_factors: Option<
        *const i32,
    >,
    in_vec_size: u32,
    out_vec_size: u32,
    num_routes: u32,
    active_experts_per_token: u32,
    #[allow(unused)] group_count_x: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    num_routed_routes: u32,
    #[specialize] b_prologue: GemmBPrologueKind,
    #[specialize] output_transform: GemmDTransform,
    #[allow(unused)]
    #[specialize]
    input_aligned: bool,
    #[specialize] signed_codes: bool,
    #[allow(unused)]
    #[specialize]
    full_tile: bool,
    #[specialize] a_per_token: bool,
    #[specialize] has_shared: bool,
) {
    let k_u = in_vec_size as usize;
    let n_u = out_vec_size as usize;
    let routes = num_routes as usize;
    let top_k = active_experts_per_token as usize;
    let _bits = BITS as usize;
    let _group_size = GROUP_SIZE as usize;

    for route_idx in 0..routes {
        let is_shared = has_shared && (route_idx >= num_routed_routes as usize);
        let expert_id = if is_shared {
            0
        } else {
            unsafe { *expert_ids.add(route_idx) as usize }
        };
        let a_row_idx = if a_per_token {
            if is_shared {
                route_idx - num_routed_routes as usize
            } else {
                route_idx / top_k
            }
        } else {
            route_idx
        };

        let expert_b = if is_shared {
            shared_b.unwrap()
        } else {
            unsafe { b.add(expert_id * b_expert_stride_bytes as usize) }
        };
        let expert_scales = if is_shared {
            shared_scales
        } else {
            scales.map(|ptr| unsafe { ptr.add(expert_id * scale_expert_stride as usize) })
        };
        let expert_zero_points = if is_shared {
            shared_zero_points
        } else {
            zero_points.map(|ptr| unsafe { ptr.add(expert_id * zero_point_expert_stride as usize) })
        };
        let expert_biases = if is_shared {
            shared_biases
        } else {
            biases.map(|ptr| unsafe { ptr.add(expert_id * scale_expert_stride as usize) })
        };
        let route_a = unsafe { a.add(a_row_idx * k_u) };
        let route_d = unsafe { d.add(route_idx * n_u) };
        let expert_output_bias = if is_shared {
            shared_output_bias
        } else {
            output_bias.map(|ptr| unsafe { ptr.add(expert_id * n_u) })
        };
        let expert_hadamard = if is_shared {
            shared_hadamard_factors
        } else {
            hadamard_factors.map(|ptr| unsafe { ptr.add(expert_id * n_u) })
        };

        let mut row_out = compute_route_gemv::<T, GROUP_SIZE, BITS>(
            expert_b,
            expert_scales,
            expert_zero_points,
            expert_biases,
            route_a,
            k_u,
            n_u,
            scale_output_stride as usize,
            scale_group_stride as usize,
            zero_point_output_stride as usize,
            zero_point_group_stride as usize,
            signed_codes,
            b_prologue,
        );

        if output_transform.contains(GemmDTransform::RHT) {
            let factors = expert_hadamard.expect("RHT requires hadamard_factors");
            let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            for block_start in (0..n_u).step_by(block_size) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
                for lane in 0..block_size {
                    stripe[lane] = T::from(row_out[block_start + lane]).unwrap().to_f32().unwrap();
                }
                hadamard_transform(&mut stripe);
                for lane in 0..block_size {
                    let sign = unsafe { *factors.add(block_start + lane) } as f32;
                    let transformed = T::from(stripe[lane] * sign).unwrap().to_f32().unwrap();
                    row_out[block_start + lane] = transformed;
                }
            }
        }

        if output_transform.contains(GemmDTransform::BIAS) {
            let bias_ptr = expert_output_bias.expect("BIAS requires output_bias");
            for (col, val) in row_out.iter_mut().enumerate() {
                let b_val = unsafe { *bias_ptr.add(col) }.to_f32().unwrap();
                *val += b_val;
            }
        }

        for (col, &val) in row_out.iter().enumerate() {
            unsafe {
                *route_d.add(col) = T::from(val).unwrap();
            }
        }
    }
}

#[kernel(MoeDownCombine)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 0, 32, 64)]
#[variants(BITS, 0, 4, 8)]
#[constraint((BITS == 0) == (GROUP_SIZE == 0))]
pub fn moe_down_combine<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    b: *const u8,
    #[optional(b_prologue != GemmBPrologueKind::FullPrecision)] scales: Option<*const T>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] zero_points: Option<*const u8>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleBiasDequant)] biases: Option<*const T>,
    #[optional(has_shared)] shared_b: Option<*const u8>,
    #[optional(has_shared && b_prologue != GemmBPrologueKind::FullPrecision)] shared_scales: Option<*const T>,
    #[optional(has_shared && b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] shared_zero_points: Option<
        *const u8,
    >,
    #[optional(has_shared && b_prologue == GemmBPrologueKind::ScaleBiasDequant)] shared_biases: Option<*const T>,
    a: *const T,
    expert_ids: *const u32,
    expert_weights: *const f32,
    #[optional(has_gate)] shared_gate: Option<*const f32>,
    output: *mut T,
    #[optional(output_transform.contains(GemmDTransform::BIAS))] output_bias: Option<*const T>,
    #[optional(output_transform.contains(GemmDTransform::RHT))] hadamard_factors: Option<*const i32>,
    #[optional(has_shared && output_transform.contains(GemmDTransform::BIAS))] shared_output_bias: Option<*const T>,
    #[allow(unused)]
    #[optional(has_shared && output_transform.contains(GemmDTransform::RHT))]
    shared_hadamard_factors: Option<*const i32>,
    in_vec_size: u32,
    out_vec_size: u32,
    active_experts_per_token: u32,
    #[allow(unused)] group_count_x: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    batch_size: u32,
    gate_stride: u32,
    #[specialize] b_prologue: GemmBPrologueKind,
    #[specialize] output_transform: GemmDTransform,
    #[allow(unused)]
    #[specialize]
    input_aligned: bool,
    #[specialize] signed_codes: bool,
    #[allow(unused)]
    #[specialize]
    full_tile: bool,
    #[specialize] has_shared: bool,
    #[specialize] has_gate: bool,
) {
    let k_u = in_vec_size as usize;
    let n_u = out_vec_size as usize;
    let active_experts = active_experts_per_token as usize;
    let gate_stride_u = gate_stride as usize;
    let num_routed_routes = (batch_size * active_experts_per_token) as usize;

    for token_idx in 0..batch_size as usize {
        let mut combined = vec![0.0f32; n_u];
        let base_routed_idx = token_idx * active_experts;

        for k in 0..active_experts {
            let route_idx = base_routed_idx + k;
            let expert_id = unsafe { *expert_ids.add(route_idx) } as usize;
            let weight = unsafe { *expert_weights.add(route_idx) };

            let expert_b = unsafe { b.add(expert_id * b_expert_stride_bytes as usize) };
            let expert_scales = scales.map(|s| unsafe { s.add(expert_id * scale_expert_stride as usize) });
            let expert_zp = zero_points.map(|z| unsafe { z.add(expert_id * zero_point_expert_stride as usize) });
            let expert_biases = biases.map(|b| unsafe { b.add(expert_id * scale_expert_stride as usize) });
            let route_a = unsafe { a.add(route_idx * k_u) };

            let mut row_out = compute_route_gemv::<T, GROUP_SIZE, BITS>(
                expert_b,
                expert_scales,
                expert_zp,
                expert_biases,
                route_a,
                k_u,
                n_u,
                scale_output_stride as usize,
                scale_group_stride as usize,
                zero_point_output_stride as usize,
                zero_point_group_stride as usize,
                signed_codes,
                b_prologue,
            );

            if output_transform.contains(GemmDTransform::BIAS) {
                let bias_ptr = output_bias.expect("BIAS requires output_bias");
                let expert_bias = unsafe { bias_ptr.add(expert_id * n_u) };
                for (col, val) in row_out.iter_mut().enumerate() {
                    let b_val = unsafe { *expert_bias.add(col) }.to_f32().unwrap();
                    *val += b_val;
                }
            }

            for (col, &val) in row_out.iter().enumerate() {
                combined[col] += weight * val;
            }
        }

        if has_shared {
            let shared_route_idx = num_routed_routes + token_idx;
            let expert_b = shared_b.unwrap();
            let expert_scales = shared_scales;
            let expert_zp = shared_zero_points;
            let expert_biases = shared_biases;
            let route_a = unsafe { a.add(shared_route_idx * k_u) };

            let mut row_out = compute_route_gemv::<T, GROUP_SIZE, BITS>(
                expert_b,
                expert_scales,
                expert_zp,
                expert_biases,
                route_a,
                k_u,
                n_u,
                scale_output_stride as usize,
                scale_group_stride as usize,
                zero_point_output_stride as usize,
                zero_point_group_stride as usize,
                signed_codes,
                b_prologue,
            );

            if output_transform.contains(GemmDTransform::BIAS) {
                let bias_ptr = shared_output_bias.expect("BIAS requires shared_output_bias");
                for (col, val) in row_out.iter_mut().enumerate() {
                    let b_val = unsafe { *bias_ptr.add(col) }.to_f32().unwrap();
                    *val += b_val;
                }
            }

            let shared_w = if has_gate {
                let g = unsafe { *shared_gate.unwrap().add(token_idx * gate_stride_u) };
                1.0f32 / (1.0f32 + (-g).exp())
            } else {
                1.0f32
            };

            for (col, &val) in row_out.iter().enumerate() {
                combined[col] += shared_w * val;
            }
        }

        if output_transform.contains(GemmDTransform::RHT) {
            let factors = hadamard_factors.expect("RHT requires hadamard_factors");
            let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            for block_start in (0..n_u).step_by(block_size) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
                stripe[..block_size].copy_from_slice(&combined[block_start..(block_start + block_size)]);
                hadamard_transform(&mut stripe);
                for lane in 0..block_size {
                    let sign = unsafe { *factors.add(block_start + lane) } as f32;
                    combined[block_start + lane] = stripe[lane] * sign;
                }
            }
        }

        for (col, &val) in combined.iter().enumerate() {
            unsafe {
                *output.add(token_idx * n_u + col) = T::from(val).unwrap();
            }
        }
    }
}

#[kernel(MoeCombine)]
#[variants(T, bf16)]
pub fn moe_combine<T: DataTypeRepr + Float>(
    expert_outputs: *const T,
    expert_weights: *const f32,
    #[optional(has_shared)] shared_output: Option<*const T>,
    #[optional(has_gate)] shared_gate: Option<*const f32>,
    output: *mut T,
    model_dim: u32,
    top_k: u32,
    batch_size: u32,
    gate_stride: u32,
    #[specialize] has_shared: bool,
    #[specialize] has_gate: bool,
) {
    let model_dim_u = model_dim as usize;
    let top_k_u = top_k as usize;
    let gate_stride_u = gate_stride as usize;
    for token_idx in 0..batch_size as usize {
        let shared_w = if has_shared {
            if has_gate {
                let g = unsafe { *shared_gate.unwrap().add(token_idx * gate_stride_u) };
                1.0f32 / (1.0f32 + (-g).exp())
            } else {
                1.0f32
            }
        } else {
            0.0f32
        };

        for col in 0..model_dim_u {
            let mut acc = 0.0f32;
            for k in 0..top_k_u {
                let w = unsafe { *expert_weights.add(token_idx * top_k_u + k) };
                let val =
                    unsafe { *expert_outputs.add((token_idx * top_k_u + k) * model_dim_u + col) }.to_f32().unwrap();
                acc += w * val;
            }
            if has_shared {
                let s_val = unsafe { *shared_output.unwrap().add(token_idx * model_dim_u + col) }.to_f32().unwrap();
                acc += shared_w * s_val;
            }
            unsafe {
                *output.add(token_idx * model_dim_u + col) = T::from(acc).unwrap();
            }
        }
    }
}
