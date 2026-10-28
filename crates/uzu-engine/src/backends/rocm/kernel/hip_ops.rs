use std::ffi::c_void;

use crate::{
    backends::{
        common::{
            CommandBufferEncoding,
            gpu_types::{
                activation_transform::ActivationTransformOp, activation_type::ActivationType,
                embedding::EmbeddingTableKind, gated_act_mul::GatedActMulOp, kv_cache_update::Copy,
                quantization::QuantizationMode, quantization_method::QuantizationMethod, ring::RingParams,
            },
        },
        rocm::{
            buffer::RocmBufferExt,
            command_buffer::RocmCommandBufferEncoding,
            kernel::{
                activation::activation::ActivationRocmKernel,
                activation_transform::activation_transform::ActivationTransformRocmKernel,
                attention::{
                    attention_prepare::AttentionPrepareRocmKernel,
                    attention_single_pass::AttentionSinglePassRocmKernel, kv_cache_update::KVCacheUpdateRocmKernel,
                    qkv_norm::QKVNormRocmKernel, sigmoid_gate::SigmoidGateRocmKernel,
                },
                embedding::input_embedding_lookup::InputEmbeddingLookupRocmKernel,
                gated_act_mul::gated_act_mul::GatedActMulRocmKernel,
                logit_transform::logit_transform::LogitTransformRocmKernel,
                normalization::normalization::NormalizationRocmKernel,
                pooling::pooling::PoolingMeanRocmKernel,
                short_conv::short_conv::{
                    ShortConvDecodeRocmKernel, ShortConvPackRocmKernel, ShortConvPrefillRocmKernel,
                },
                softmax::softmax::SoftmaxRocmKernel,
                tensor_add_bias::tensor_add_bias::TensorAddBiasRocmKernel,
                tensor_add_scale::tensor_add_scale::TensorAddScaleRocmKernel,
            },
        },
    },
    data_type::DataType,
    utils::pointers::{SendPtr, SendPtrMut},
};

#[inline]
pub(crate) fn dtype_code(dt: DataType) -> Option<u32> {
    match dt {
        DataType::F32 => Some(0),
        DataType::F16 => Some(1),
        DataType::BF16 => Some(2),
        _ => None,
    }
}

#[inline]
pub(crate) fn act_type_code(act: ActivationType) -> u32 {
    match act {
        ActivationType::SILU => 0,
        ActivationType::GELUApprox => 1,
        ActivationType::GELUExact => 2,
        ActivationType::IDENTITY => 3,
        ActivationType::SOFTPLUS => 4,
    }
}

pub(crate) fn try_encode_activation(
    kernel: &ActivationRocmKernel,
    input: Option<SendPtr<u8>>,
    output: SendPtrMut<u8>,
    n: u32,
    act_type: ActivationType,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let in_ptr = if kernel.in_place {
        SendPtr(output.as_ptr().cast_const())
    } else {
        input.expect("non-in-place activation requires input")
    };
    let act_code = act_type_code(act_type);
    let func = command_buffer.context().kernels.activation;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_ptr.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_n = n;
        let mut p_dtype = dtype;
        let mut p_act = act_code;
        let mut params: [*mut c_void; 5] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_n as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
            (&mut p_act as *mut u32).cast(),
        ];
        let block = 256u32;
        let grid = n.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_activation failed");
    });
    true
}

pub(crate) fn try_encode_tensor_add_bias(
    kernel: &TensorAddBiasRocmKernel,
    input: Option<SendPtr<u8>>,
    bias: SendPtr<u8>,
    output: SendPtrMut<u8>,
    num_cols: u32,
    length: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let (Some(t_dtype), Some(bias_dtype)) = (dtype_code(kernel.T), dtype_code(kernel.BiasT)) else {
        return false;
    };
    let in_ptr = if kernel.in_place {
        SendPtr(output.as_ptr().cast_const())
    } else {
        input.expect("non-in-place tensor_add_bias requires input")
    };
    let func = command_buffer.context().kernels.tensor_add_bias;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_ptr.as_ptr() as *const c_void;
        let mut p_bias = bias.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_cols = num_cols;
        let mut p_len = length;
        let mut p_td = t_dtype;
        let mut p_bd = bias_dtype;
        let mut params: [*mut c_void; 7] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_bias as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_cols as *mut u32).cast(),
            (&mut p_len as *mut u32).cast(),
            (&mut p_td as *mut u32).cast(),
            (&mut p_bd as *mut u32).cast(),
        ];
        let block = 256u32;
        let grid = length.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_tensor_add_bias failed");
    });
    true
}

pub(crate) fn try_encode_tensor_add_scale(
    kernel: &TensorAddScaleRocmKernel,
    input: Option<SendPtr<u8>>,
    bias: SendPtr<u8>,
    output: SendPtrMut<u8>,
    num_cols: u32,
    length: u32,
    scale: f32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let in_ptr = match input {
        Some(ptr) => ptr,
        None => SendPtr(output.as_ptr().cast_const()),
    };
    let func = command_buffer.context().kernels.tensor_add_scale;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_ptr.as_ptr() as *const c_void;
        let mut p_bias = bias.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_cols = num_cols;
        let mut p_len = length;
        let mut p_scale = scale;
        let mut p_dtype = dtype;
        let mut params: [*mut c_void; 7] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_bias as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_cols as *mut u32).cast(),
            (&mut p_len as *mut u32).cast(),
            (&mut p_scale as *mut f32).cast(),
            (&mut p_dtype as *mut u32).cast(),
        ];
        let block = 256u32;
        let grid = length.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_tensor_add_scale failed");
    });
    true
}

pub(crate) fn try_encode_logit_transform(
    kernel: &LogitTransformRocmKernel,
    logits: SendPtrMut<u8>,
    length: u32,
    scale: f32,
    soft_cap: f32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let has_soft_cap = u32::from(kernel.has_soft_cap);
    let func = command_buffer.context().kernels.logit_transform;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_logits = logits.as_ptr() as *mut c_void;
        let mut p_len = length;
        let mut p_scale = scale;
        let mut p_cap = soft_cap;
        let mut p_has_cap = has_soft_cap;
        let mut p_dtype = dtype;
        let mut params: [*mut c_void; 6] = [
            (&mut p_logits as *mut *mut c_void).cast(),
            (&mut p_len as *mut u32).cast(),
            (&mut p_scale as *mut f32).cast(),
            (&mut p_cap as *mut f32).cast(),
            (&mut p_has_cap as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
        ];
        let block = 256u32;
        let grid = length.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_logit_transform failed");
    });
    true
}

pub(crate) fn try_encode_sigmoid_gate(
    kernel: &SigmoidGateRocmKernel,
    gate: SendPtr<u8>,
    output: SendPtrMut<u8>,
    gate_dim: u32,
    batch_dim: u32,
    gate_row_stride: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let func = command_buffer.context().kernels.sigmoid_gate;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_gate = gate.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_gdim = gate_dim;
        let mut p_bdim = batch_dim;
        let mut p_stride = gate_row_stride;
        let mut p_dtype = dtype;
        let mut params: [*mut c_void; 6] = [
            (&mut p_gate as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_gdim as *mut u32).cast(),
            (&mut p_bdim as *mut u32).cast(),
            (&mut p_stride as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
        ];
        let total = batch_dim.saturating_mul(gate_dim);
        let block = 256u32;
        let grid = total.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_sigmoid_gate failed");
    });
    true
}

pub(crate) fn try_encode_pooling_mean(
    kernel: &PoolingMeanRocmKernel,
    input: SendPtr<u8>,
    output: SendPtrMut<u8>,
    seq_len: u32,
    hidden_dim: u32,
    batch_size: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let func = command_buffer.context().kernels.pooling_mean;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = input.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_seq = seq_len;
        let mut p_hid = hidden_dim;
        let mut p_batch = batch_size;
        let mut p_dtype = dtype;
        let mut params: [*mut c_void; 6] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_seq as *mut u32).cast(),
            (&mut p_hid as *mut u32).cast(),
            (&mut p_batch as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
        ];
        let total = batch_size.saturating_mul(hidden_dim);
        let block = 256u32;
        let grid = total.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_pooling_mean failed");
    });
    true
}

pub(crate) fn try_encode_softmax(
    kernel: &SoftmaxRocmKernel,
    values: SendPtrMut<u8>,
    sinks: Option<SendPtr<u8>>,
    row_length: u32,
    outer_dim: u32,
    batch_dim: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let sinks_ptr = sinks.unwrap_or(SendPtr(std::ptr::null()));
    let has_sinks = u32::from(kernel.has_sinks);
    let func = command_buffer.context().kernels.softmax;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_vals = values.as_ptr() as *mut c_void;
        let mut p_sinks = sinks_ptr.as_ptr() as *const c_void;
        let mut p_rlen = row_length;
        let mut p_outer = outer_dim;
        let mut p_batch = batch_dim;
        let mut p_has_sinks = has_sinks;
        let mut p_dtype = dtype;
        let mut params: [*mut c_void; 7] = [
            (&mut p_vals as *mut *mut c_void).cast(),
            (&mut p_sinks as *mut *const c_void).cast(),
            (&mut p_rlen as *mut u32).cast(),
            (&mut p_outer as *mut u32).cast(),
            (&mut p_batch as *mut u32).cast(),
            (&mut p_has_sinks as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
        ];
        let total_rows = outer_dim.saturating_mul(batch_dim);
        let block = 64u32;
        let grid = total_rows.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_softmax failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_normalization(
    kernel: &NormalizationRocmKernel,
    input: Option<SendPtr<u8>>,
    scales: Option<SendPtr<u8>>,
    biases: Option<SendPtr<u8>>,
    output: SendPtrMut<u8>,
    shortcut: Option<SendPtrMut<u8>>,
    hadamard_factors: Option<SendPtr<u8>>,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    scale_offset: f32,
    post_layer_scalar: f32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    if kernel.AccumT != DataType::F32 {
        return false;
    }
    let (Some(in_dtype), Some(affine_dtype), Some(out_dtype)) =
        (dtype_code(kernel.InputT), dtype_code(kernel.AffineT), dtype_code(kernel.OutputT))
    else {
        return false;
    };
    let in_ptr = if kernel.in_place {
        SendPtr(output.as_ptr().cast_const())
    } else {
        input.expect("non-in-place normalization requires input")
    };
    let scales_ptr = scales.unwrap_or(SendPtr(std::ptr::null()));
    let biases_ptr = biases.unwrap_or(SendPtr(std::ptr::null()));
    let shortcut_ptr = shortcut.unwrap_or(SendPtrMut(std::ptr::null_mut()));
    let hadamard_ptr = hadamard_factors.unwrap_or(SendPtr(std::ptr::null()));
    let flags = (u32::from(kernel.in_place))
        | (u32::from(kernel.subtract_mean) << 1)
        | (u32::from(kernel.full_layer) << 2)
        | (u32::from(kernel.copy_to_shortcut) << 3)
        | (u32::from(kernel.residual_add) << 4)
        | (u32::from(kernel.use_hadamard) << 5)
        | (u32::from(kernel.scale_residual_sum) << 6)
        | (u32::from(kernel.scale_output) << 7)
        | (u32::from(kernel.has_biases) << 8)
        | (u32::from(kernel.has_scales) << 9);
    let func = command_buffer.context().kernels.normalization;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_ptr.as_ptr() as *const c_void;
        let mut p_sc = scales_ptr.as_ptr() as *const c_void;
        let mut p_bi = biases_ptr.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_sh = shortcut_ptr.as_ptr() as *mut c_void;
        let mut p_had = hadamard_ptr.as_ptr() as *const c_void;
        let mut p_bs = batch_size;
        let mut p_ec = element_count;
        let mut p_eps = epsilon;
        let mut p_soff = scale_offset;
        let mut p_pls = post_layer_scalar;
        let mut p_id = in_dtype;
        let mut p_ad = affine_dtype;
        let mut p_od = out_dtype;
        let mut p_fl = flags;
        let mut params: [*mut c_void; 15] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_sc as *mut *const c_void).cast(),
            (&mut p_bi as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_sh as *mut *mut c_void).cast(),
            (&mut p_had as *mut *const c_void).cast(),
            (&mut p_bs as *mut u32).cast(),
            (&mut p_ec as *mut u32).cast(),
            (&mut p_eps as *mut f32).cast(),
            (&mut p_soff as *mut f32).cast(),
            (&mut p_pls as *mut f32).cast(),
            (&mut p_id as *mut u32).cast(),
            (&mut p_ad as *mut u32).cast(),
            (&mut p_od as *mut u32).cast(),
            (&mut p_fl as *mut u32).cast(),
        ];
        let block = 64u32;
        let grid = batch_size.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_normalization failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_qkv_norm(
    kernel: &QKVNormRocmKernel,
    qkvg_input: Option<SendPtr<u8>>,
    scales: Option<SendPtr<u8>>,
    qkvg_output: SendPtrMut<u8>,
    batch_size: u32,
    input_row_stride: u32,
    head_dim: u32,
    epsilon: f32,
    scale_offset: f32,
    head_offset: u32,
    head_count: u32,
    full_layer: bool,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    if kernel.AccumT != DataType::F32 {
        return false;
    }
    let (Some(in_dtype), Some(scale_dtype), Some(out_dtype)) =
        (dtype_code(kernel.InputT), dtype_code(kernel.ScaleT), dtype_code(kernel.OutputT))
    else {
        return false;
    };
    let in_ptr = if kernel.in_place {
        SendPtr(qkvg_output.as_ptr().cast_const())
    } else {
        qkvg_input.expect("non-in-place qkv_norm requires input")
    };
    let scales_ptr = scales.unwrap_or(SendPtr(std::ptr::null()));
    let flags = u32::from(full_layer) | (u32::from(kernel.has_scales) << 1);
    let func = command_buffer.context().kernels.qkv_norm;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_ptr.as_ptr() as *const c_void;
        let mut p_sc = scales_ptr.as_ptr() as *const c_void;
        let mut p_out = qkvg_output.as_ptr() as *mut c_void;
        let mut p_bs = batch_size;
        let mut p_stride = input_row_stride;
        let mut p_hdim = head_dim;
        let mut p_eps = epsilon;
        let mut p_soff = scale_offset;
        let mut p_hoff = head_offset;
        let mut p_hcnt = head_count;
        let mut p_id = in_dtype;
        let mut p_sd = scale_dtype;
        let mut p_od = out_dtype;
        let mut p_fl = flags;
        let mut params: [*mut c_void; 14] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_sc as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_bs as *mut u32).cast(),
            (&mut p_stride as *mut u32).cast(),
            (&mut p_hdim as *mut u32).cast(),
            (&mut p_eps as *mut f32).cast(),
            (&mut p_soff as *mut f32).cast(),
            (&mut p_hoff as *mut u32).cast(),
            (&mut p_hcnt as *mut u32).cast(),
            (&mut p_id as *mut u32).cast(),
            (&mut p_sd as *mut u32).cast(),
            (&mut p_od as *mut u32).cast(),
            (&mut p_fl as *mut u32).cast(),
        ];
        let total = batch_size.saturating_mul(head_count);
        let block = 64u32;
        let grid = total.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_qkv_norm failed");
    });
    true
}

pub(crate) fn try_encode_kv_cache_update(
    kernel: &KVCacheUpdateRocmKernel,
    in_place_keys: SendPtrMut<u8>,
    in_place_values: SendPtrMut<u8>,
    copies: &[Copy],
    copy_count: u32,
    element_dim: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    if copy_count == 0 || element_dim == 0 {
        return true;
    }
    let count = (copy_count as usize).min(copies.len());
    let byte_len = count * std::mem::size_of::<Copy>();
    let Ok(const_buf) = command_buffer.allocate_constant(byte_len) else {
        return false;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(copies.as_ptr().cast::<u8>(), const_buf.cpu_address().as_ptr().cast(), byte_len);
    }
    let copies_dev_ptr = SendPtr(const_buf.device_address().as_ptr().cast::<u8>().cast_const());
    drop(const_buf);

    let func = command_buffer.context().kernels.kv_cache_update;
    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_keys = in_place_keys.as_ptr() as *mut c_void;
        let mut p_vals = in_place_values.as_ptr() as *mut c_void;
        let mut p_copies = copies_dev_ptr.as_ptr() as *const c_void;
        let mut p_cnt = count as u32;
        let mut p_dim = element_dim;
        let mut p_dtype = dtype;
        let mut params: [*mut c_void; 6] = [
            (&mut p_keys as *mut *mut c_void).cast(),
            (&mut p_vals as *mut *mut c_void).cast(),
            (&mut p_copies as *mut *const c_void).cast(),
            (&mut p_cnt as *mut u32).cast(),
            (&mut p_dim as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
        ];
        let block = 256u32;
        let grid = element_dim.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_kv_cache_update failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_gated_act_mul(
    kernel: &GatedActMulRocmKernel,
    act_operand: SendPtr<u8>,
    value_operand: Option<SendPtr<u8>>,
    fp_out: Option<SendPtrMut<u8>>,
    q_out: Option<SendPtrMut<u8>>,
    scales_out: Option<SendPtrMut<u8>>,
    group_sums_out: Option<SendPtrMut<u8>>,
    hadamard_factors: Option<SendPtr<u8>>,
    gated_dim: u32,
    batch_dim: u32,
    value_offset: u32,
    value_row_stride: u32,
    act_type: ActivationType,
    activation_alpha: Option<f32>,
    gate_clip_min: Option<f32>,
    gate_clip_max: Option<f32>,
    value_clip_min: Option<f32>,
    value_clip_max: Option<f32>,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let val_ptr = value_operand.unwrap_or(SendPtr(std::ptr::null()));
    let had_ptr = hadamard_factors.unwrap_or(SendPtr(std::ptr::null()));
    let act_code = act_type_code(act_type);
    let alpha = activation_alpha.unwrap_or(1.0);
    let g_min = gate_clip_min.unwrap_or(f32::NEG_INFINITY);
    let g_max = gate_clip_max.unwrap_or(f32::INFINITY);
    let v_min = value_clip_min.unwrap_or(f32::NEG_INFINITY);
    let v_max = value_clip_max.unwrap_or(f32::INFINITY);

    if kernel.ops != GatedActMulOp::FullPrecision {
        let scale_group_size = kernel.activation_scale_group_size;
        let sum_group_size = kernel.sum_group_size;
        let emit_group_sums = kernel.ops == GatedActMulOp::QuantizeWithGroupSums;
        if !kernel.use_hadamard
            || scale_group_size == 0
            || scale_group_size > 256
            || !scale_group_size.is_multiple_of(32)
            || !gated_dim.is_multiple_of(scale_group_size)
            || (emit_group_sums
                && (sum_group_size == 0
                    || scale_group_size < sum_group_size
                    || !scale_group_size.is_multiple_of(sum_group_size)))
        {
            return false;
        }
        let (Some(q_out), Some(scales_out)) = (q_out, scales_out) else {
            return false;
        };
        let sums_ptr = group_sums_out.unwrap_or(SendPtrMut(std::ptr::null_mut()));
        let flags = u32::from(kernel.interleaved)
            | (u32::from(kernel.use_hadamard) << 1)
            | (u32::from(kernel.custom_activation_alpha) << 2)
            | (u32::from(kernel.clip_gate) << 3)
            | (u32::from(kernel.clip_value) << 4)
            | (u32::from(emit_group_sums) << 5)
            | (u32::from(kernel.grouped_by_weight_nibble) << 6);
        let func = command_buffer.context().kernels.gated_act_mul_quant;

        command_buffer.push_gpu_command(move |stream| unsafe {
            let mut p_act = act_operand.as_ptr() as *const c_void;
            let mut p_val = val_ptr.as_ptr() as *const c_void;
            let mut p_qout = q_out.as_ptr() as *mut c_void;
            let mut p_sout = scales_out.as_ptr() as *mut c_void;
            let mut p_gsum = sums_ptr.as_ptr() as *mut c_void;
            let mut p_had = had_ptr.as_ptr() as *const c_void;
            let mut p_gdim = gated_dim;
            let mut p_bdim = batch_dim;
            let mut p_voff = value_offset;
            let mut p_vstr = value_row_stride;
            let mut p_sgs = scale_group_size;
            let mut p_sumgs = sum_group_size;
            let mut p_atype = act_code;
            let mut p_alpha = alpha;
            let mut p_gmin = g_min;
            let mut p_gmax = g_max;
            let mut p_vmin = v_min;
            let mut p_vmax = v_max;
            let mut p_dtype = dtype;
            let mut p_flags = flags;
            let mut params: [*mut c_void; 20] = [
                (&mut p_act as *mut *const c_void).cast(),
                (&mut p_val as *mut *const c_void).cast(),
                (&mut p_qout as *mut *mut c_void).cast(),
                (&mut p_sout as *mut *mut c_void).cast(),
                (&mut p_gsum as *mut *mut c_void).cast(),
                (&mut p_had as *mut *const c_void).cast(),
                (&mut p_gdim as *mut u32).cast(),
                (&mut p_bdim as *mut u32).cast(),
                (&mut p_voff as *mut u32).cast(),
                (&mut p_vstr as *mut u32).cast(),
                (&mut p_sgs as *mut u32).cast(),
                (&mut p_sumgs as *mut u32).cast(),
                (&mut p_atype as *mut u32).cast(),
                (&mut p_alpha as *mut f32).cast(),
                (&mut p_gmin as *mut f32).cast(),
                (&mut p_gmax as *mut f32).cast(),
                (&mut p_vmin as *mut f32).cast(),
                (&mut p_vmax as *mut f32).cast(),
                (&mut p_dtype as *mut u32).cast(),
                (&mut p_flags as *mut u32).cast(),
            ];
            let total = batch_dim.saturating_mul(gated_dim / scale_group_size);
            let block = 128u32;
            let grid = total.div_ceil(block);
            stream
                .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                .expect("rocm_gated_act_mul_quant failed");
        });
        return true;
    }

    let Some(fp_out) = fp_out else {
        return false;
    };
    let flags = u32::from(kernel.interleaved)
        | (u32::from(kernel.use_hadamard) << 1)
        | (u32::from(kernel.custom_activation_alpha) << 2)
        | (u32::from(kernel.clip_gate) << 3)
        | (u32::from(kernel.clip_value) << 4);
    let use_hadamard = kernel.use_hadamard;
    let func = command_buffer.context().kernels.gated_act_mul_fp;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_act = act_operand.as_ptr() as *const c_void;
        let mut p_val = val_ptr.as_ptr() as *const c_void;
        let mut p_out = fp_out.as_ptr() as *mut c_void;
        let mut p_had = had_ptr.as_ptr() as *const c_void;
        let mut p_gdim = gated_dim;
        let mut p_bdim = batch_dim;
        let mut p_voff = value_offset;
        let mut p_vstr = value_row_stride;
        let mut p_atype = act_code;
        let mut p_alpha = alpha;
        let mut p_gmin = g_min;
        let mut p_gmax = g_max;
        let mut p_vmin = v_min;
        let mut p_vmax = v_max;
        let mut p_dtype = dtype;
        let mut p_flags = flags;
        let mut params: [*mut c_void; 16] = [
            (&mut p_act as *mut *const c_void).cast(),
            (&mut p_val as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_had as *mut *const c_void).cast(),
            (&mut p_gdim as *mut u32).cast(),
            (&mut p_bdim as *mut u32).cast(),
            (&mut p_voff as *mut u32).cast(),
            (&mut p_vstr as *mut u32).cast(),
            (&mut p_atype as *mut u32).cast(),
            (&mut p_alpha as *mut f32).cast(),
            (&mut p_gmin as *mut f32).cast(),
            (&mut p_gmax as *mut f32).cast(),
            (&mut p_vmin as *mut f32).cast(),
            (&mut p_vmax as *mut f32).cast(),
            (&mut p_dtype as *mut u32).cast(),
            (&mut p_flags as *mut u32).cast(),
        ];
        let total = if use_hadamard {
            batch_dim.saturating_mul(gated_dim / 32)
        } else {
            batch_dim.saturating_mul(gated_dim)
        };
        let block = 128u32;
        let grid = total.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_gated_act_mul_fp failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_activation_transform(
    kernel: &ActivationTransformRocmKernel,
    input: Option<SendPtr<u8>>,
    fp_out: Option<SendPtrMut<u8>>,
    bias: Option<SendPtr<u8>>,
    q_out: Option<SendPtrMut<u8>>,
    scales_out: Option<SendPtrMut<u8>>,
    group_sums_out: Option<SendPtrMut<u8>>,
    rht_factors: SendPtr<u8>,
    batch_size: u32,
    element_count: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let (Some(t_dtype), Some(bias_dtype)) = (dtype_code(kernel.T), dtype_code(kernel.BiasT)) else {
        return false;
    };
    if element_count == 0 || !element_count.is_multiple_of(32) {
        return false;
    }
    let quantize =
        matches!(kernel.ops, ActivationTransformOp::Quantize | ActivationTransformOp::QuantizeWithGroupSums);
    let emit_group_sums = kernel.ops == ActivationTransformOp::QuantizeWithGroupSums;
    let scale_group_size = kernel.activation_scale_group_size;
    let sum_group_size = kernel.sum_group_size;

    if quantize
        && (scale_group_size == 0
            || scale_group_size > 256
            || !scale_group_size.is_multiple_of(32)
            || !element_count.is_multiple_of(scale_group_size)
            || (emit_group_sums
                && (sum_group_size == 0
                    || scale_group_size < sum_group_size
                    || !scale_group_size.is_multiple_of(sum_group_size))))
    {
        return false;
    }

    let in_ptr = if kernel.in_place {
        let Some(ref fp_out) = fp_out else {
            return false;
        };
        SendPtr(fp_out.as_ptr().cast_const())
    } else {
        let Some(input) = input else {
            return false;
        };
        input
    };
    let fp_ptr = fp_out.unwrap_or(SendPtrMut(std::ptr::null_mut()));
    let bias_ptr = bias.unwrap_or(SendPtr(std::ptr::null()));
    let q_ptr = q_out.unwrap_or(SendPtrMut(std::ptr::null_mut()));
    let sc_ptr = scales_out.unwrap_or(SendPtrMut(std::ptr::null_mut()));
    let gs_ptr = group_sums_out.unwrap_or(SendPtrMut(std::ptr::null_mut()));

    let input_rht = kernel.ops != ActivationTransformOp::OutputRht;
    let flags = u32::from(input_rht)
        | (u32::from(quantize) << 1)
        | (u32::from(emit_group_sums) << 2)
        | (u32::from(kernel.grouped_by_weight_nibble) << 3)
        | (u32::from(kernel.has_bias) << 4);
    let func = command_buffer.context().kernels.activation_transform;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_ptr.as_ptr() as *const c_void;
        let mut p_fp = fp_ptr.as_ptr() as *mut c_void;
        let mut p_bias = bias_ptr.as_ptr() as *const c_void;
        let mut p_q = q_ptr.as_ptr() as *mut c_void;
        let mut p_sc = sc_ptr.as_ptr() as *mut c_void;
        let mut p_gs = gs_ptr.as_ptr() as *mut c_void;
        let mut p_rht = rht_factors.as_ptr() as *const c_void;
        let mut p_bs = batch_size;
        let mut p_ec = element_count;
        let mut p_sgs = scale_group_size;
        let mut p_sumgs = sum_group_size;
        let mut p_td = t_dtype;
        let mut p_bd = bias_dtype;
        let mut p_fl = flags;
        let mut params: [*mut c_void; 14] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_fp as *mut *mut c_void).cast(),
            (&mut p_bias as *mut *const c_void).cast(),
            (&mut p_q as *mut *mut c_void).cast(),
            (&mut p_sc as *mut *mut c_void).cast(),
            (&mut p_gs as *mut *mut c_void).cast(),
            (&mut p_rht as *mut *const c_void).cast(),
            (&mut p_bs as *mut u32).cast(),
            (&mut p_ec as *mut u32).cast(),
            (&mut p_sgs as *mut u32).cast(),
            (&mut p_sumgs as *mut u32).cast(),
            (&mut p_td as *mut u32).cast(),
            (&mut p_bd as *mut u32).cast(),
            (&mut p_fl as *mut u32).cast(),
        ];
        let total = if quantize {
            batch_size.saturating_mul(element_count / scale_group_size)
        } else {
            batch_size.saturating_mul(element_count / 32)
        };
        let block = 128u32;
        let grid = total.div_ceil(block);
        stream
            .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
            .expect("rocm_activation_transform failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_short_conv_pack(
    kernel: &ShortConvPackRocmKernel,
    state_in: SendPtr<u8>,
    in_proj: SendPtr<u8>,
    padded: SendPtrMut<u8>,
    state_stride: u32,
    suffix_len: u32,
    in_proj_stride: u32,
    model_dim: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let func = command_buffer.context().kernels.short_conv_pack;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_st = state_in.as_ptr() as *const c_void;
        let mut p_in = in_proj.as_ptr() as *const c_void;
        let mut p_pad = padded.as_ptr() as *mut c_void;
        let mut p_ss = state_stride;
        let mut p_sl = suffix_len;
        let mut p_ips = in_proj_stride;
        let mut p_md = model_dim;
        let mut p_dt = dtype;
        let mut params: [*mut c_void; 8] = [
            (&mut p_st as *mut *const c_void).cast(),
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_pad as *mut *mut c_void).cast(),
            (&mut p_ss as *mut u32).cast(),
            (&mut p_sl as *mut u32).cast(),
            (&mut p_ips as *mut u32).cast(),
            (&mut p_md as *mut u32).cast(),
            (&mut p_dt as *mut u32).cast(),
        ];
        let total = (state_stride.saturating_add(suffix_len)).saturating_mul(model_dim);
        let block = 128u32;
        let grid = total.div_ceil(block);
        stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_short_conv_pack failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_short_conv_prefill(
    kernel: &ShortConvPrefillRocmKernel,
    padded: SendPtr<u8>,
    in_proj: SendPtr<u8>,
    w: SendPtr<u8>,
    b: Option<SendPtr<u8>>,
    out: SendPtrMut<u8>,
    state_out: SendPtrMut<u8>,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    if kernel.WeightT != DataType::F32 {
        return false;
    }
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let b_ptr = b.unwrap_or(SendPtr(std::ptr::null()));
    let has_bias = u32::from(kernel.has_bias);
    let func = command_buffer.context().kernels.short_conv_prefill;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_pad = padded.as_ptr() as *const c_void;
        let mut p_in = in_proj.as_ptr() as *const c_void;
        let mut p_w = w.as_ptr() as *const c_void;
        let mut p_b = b_ptr.as_ptr() as *const c_void;
        let mut p_out = out.as_ptr() as *mut c_void;
        let mut p_sout = state_out.as_ptr() as *mut c_void;
        let mut p_sl = suffix_len;
        let mut p_ks = kernel_size;
        let mut p_ips = in_proj_stride;
        let mut p_ss = state_stride;
        let mut p_md = model_dim;
        let mut p_dt = dtype;
        let mut p_hb = has_bias;
        let mut params: [*mut c_void; 13] = [
            (&mut p_pad as *mut *const c_void).cast(),
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_w as *mut *const c_void).cast(),
            (&mut p_b as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_sout as *mut *mut c_void).cast(),
            (&mut p_sl as *mut u32).cast(),
            (&mut p_ks as *mut u32).cast(),
            (&mut p_ips as *mut u32).cast(),
            (&mut p_ss as *mut u32).cast(),
            (&mut p_md as *mut u32).cast(),
            (&mut p_dt as *mut u32).cast(),
            (&mut p_hb as *mut u32).cast(),
        ];
        let tap_count = kernel_size.saturating_sub(1);
        let total = (suffix_len.saturating_add(tap_count)).saturating_mul(model_dim);
        let block = 128u32;
        let grid = total.div_ceil(block);
        stream
            .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
            .expect("rocm_short_conv_prefill failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_short_conv_decode(
    kernel: &ShortConvDecodeRocmKernel,
    in_proj: SendPtr<u8>,
    w: SendPtr<u8>,
    b: Option<SendPtr<u8>>,
    state: Option<SendPtr<u8>>,
    out: SendPtrMut<u8>,
    next_state: SendPtrMut<u8>,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    if kernel.WeightT != DataType::F32 {
        return false;
    }
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let b_ptr = b.unwrap_or(SendPtr(std::ptr::null()));
    let state_ptr = state.unwrap_or(SendPtr(std::ptr::null()));
    let flags = u32::from(kernel.has_bias) | (u32::from(kernel.state_in_place) << 1);
    let func = command_buffer.context().kernels.short_conv_decode;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_in = in_proj.as_ptr() as *const c_void;
        let mut p_w = w.as_ptr() as *const c_void;
        let mut p_b = b_ptr.as_ptr() as *const c_void;
        let mut p_st = state_ptr.as_ptr() as *const c_void;
        let mut p_out = out.as_ptr() as *mut c_void;
        let mut p_nst = next_state.as_ptr() as *mut c_void;
        let mut p_sl = suffix_len;
        let mut p_ks = kernel_size;
        let mut p_ips = in_proj_stride;
        let mut p_ss = state_stride;
        let mut p_md = model_dim;
        let mut p_dt = dtype;
        let mut p_fl = flags;
        let mut params: [*mut c_void; 13] = [
            (&mut p_in as *mut *const c_void).cast(),
            (&mut p_w as *mut *const c_void).cast(),
            (&mut p_b as *mut *const c_void).cast(),
            (&mut p_st as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_nst as *mut *mut c_void).cast(),
            (&mut p_sl as *mut u32).cast(),
            (&mut p_ks as *mut u32).cast(),
            (&mut p_ips as *mut u32).cast(),
            (&mut p_ss as *mut u32).cast(),
            (&mut p_md as *mut u32).cast(),
            (&mut p_dt as *mut u32).cast(),
            (&mut p_fl as *mut u32).cast(),
        ];
        let block = 128u32;
        let grid = model_dim.div_ceil(block);
        stream
            .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
            .expect("rocm_short_conv_decode failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_attention_prepare(
    kernel: &AttentionPrepareRocmKernel,
    qkvg: SendPtr<u8>,
    queries: SendPtrMut<u8>,
    keys: Option<SendPtrMut<u8>>,
    values: Option<SendPtrMut<u8>>,
    cosines: Option<SendPtr<u8>>,
    sines: Option<SendPtr<u8>>,
    num_q_heads: u32,
    num_kv_heads: Option<u32>,
    head_dim: u32,
    rope_dim: Option<u32>,
    kv_token_offset: Option<u32>,
    input_row_stride: u32,
    batch_dim: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    if kernel.RopeT != DataType::F32 {
        return false;
    }
    let Some(dtype) = dtype_code(kernel.ElementT) else {
        return false;
    };
    let keys_ptr = keys.unwrap_or(SendPtrMut(std::ptr::null_mut()));
    let values_ptr = values.unwrap_or(SendPtrMut(std::ptr::null_mut()));
    let cos_ptr = cosines.unwrap_or(SendPtr(std::ptr::null()));
    let sin_ptr = sines.unwrap_or(SendPtr(std::ptr::null()));
    let n_kvh = num_kv_heads.unwrap_or(0);
    let r_dim = rope_dim.unwrap_or(0);
    let kv_off = kv_token_offset.unwrap_or(0);
    let flags = u32::from(kernel.has_kv) | (u32::from(kernel.has_rope) << 1);
    let total_heads = if kernel.has_kv {
        num_q_heads + n_kvh * 2
    } else {
        num_q_heads
    };
    let func = command_buffer.context().kernels.attention_prepare;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_qkvg = qkvg.as_ptr() as *const c_void;
        let mut p_q = queries.as_ptr() as *mut c_void;
        let mut p_k = keys_ptr.as_ptr() as *mut c_void;
        let mut p_v = values_ptr.as_ptr() as *mut c_void;
        let mut p_cos = cos_ptr.as_ptr() as *const c_void;
        let mut p_sin = sin_ptr.as_ptr() as *const c_void;
        let mut p_nqh = num_q_heads;
        let mut p_nkvh = n_kvh;
        let mut p_hd = head_dim;
        let mut p_rd = r_dim;
        let mut p_kvoff = kv_off;
        let mut p_irs = input_row_stride;
        let mut p_bd = batch_dim;
        let mut p_dt = dtype;
        let mut p_fl = flags;
        let mut params: [*mut c_void; 15] = [
            (&mut p_qkvg as *mut *const c_void).cast(),
            (&mut p_q as *mut *mut c_void).cast(),
            (&mut p_k as *mut *mut c_void).cast(),
            (&mut p_v as *mut *mut c_void).cast(),
            (&mut p_cos as *mut *const c_void).cast(),
            (&mut p_sin as *mut *const c_void).cast(),
            (&mut p_nqh as *mut u32).cast(),
            (&mut p_nkvh as *mut u32).cast(),
            (&mut p_hd as *mut u32).cast(),
            (&mut p_rd as *mut u32).cast(),
            (&mut p_kvoff as *mut u32).cast(),
            (&mut p_irs as *mut u32).cast(),
            (&mut p_bd as *mut u32).cast(),
            (&mut p_dt as *mut u32).cast(),
            (&mut p_fl as *mut u32).cast(),
        ];
        let total = batch_dim.saturating_mul(total_heads).saturating_mul(head_dim);
        let block = 128u32;
        let grid = total.div_ceil(block);
        stream
            .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
            .expect("rocm_attention_prepare failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_attention_single_pass(
    kernel: &AttentionSinglePassRocmKernel,
    queries: SendPtr<u8>,
    keys: SendPtr<u8>,
    values: SendPtr<u8>,
    out: SendPtrMut<u8>,
    gqa_factor: u32,
    sequence_length: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    v_head_stride: u32,
    v_seq_stride: u32,
    ring_params: Option<RingParams>,
    scale: f32,
    trie: Option<SendPtr<u8>>,
    sliding_window_size: Option<u32>,
    sinks: Option<SendPtr<u8>>,
    num_heads: u32,
    suffix_length: u32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let head_dim = kernel.HEAD_DIM;
    if head_dim > 512 {
        return false;
    }
    let (ring_offset, ring_length) = ring_params.map(|r| (r.ring_offset, r.ring_length)).unwrap_or((0, 0));
    let trie_ptr = trie.unwrap_or(SendPtr(std::ptr::null()));
    let sinks_ptr = sinks.unwrap_or(SendPtr(std::ptr::null()));
    let sw_size = sliding_window_size.unwrap_or(0);
    let flags = u32::from(kernel.has_sinks)
        | (u32::from(kernel.is_kv_cache_ring) << 1)
        | (u32::from(kernel.is_causal) << 2)
        | (u32::from(kernel.is_trie) << 3)
        | (u32::from(kernel.is_sliding_window) << 4);
    let func = command_buffer.context().kernels.attention_single_pass;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_q = queries.as_ptr() as *const c_void;
        let mut p_k = keys.as_ptr() as *const c_void;
        let mut p_v = values.as_ptr() as *const c_void;
        let mut p_out = out.as_ptr() as *mut c_void;
        let mut p_trie = trie_ptr.as_ptr() as *const c_void;
        let mut p_sinks = sinks_ptr.as_ptr() as *const c_void;
        let mut p_gqa = gqa_factor;
        let mut p_seq = sequence_length;
        let mut p_kh = k_head_stride;
        let mut p_ks = k_seq_stride;
        let mut p_vh = v_head_stride;
        let mut p_vs = v_seq_stride;
        let mut p_roff = ring_offset;
        let mut p_rlen = ring_length;
        let mut p_scale = scale;
        let mut p_sw = sw_size;
        let mut p_nh = num_heads;
        let mut p_suflen = suffix_length;
        let mut p_hdim = head_dim;
        let mut p_dtype = dtype;
        let mut p_flags = flags;
        let mut params: [*mut c_void; 21] = [
            (&mut p_q as *mut *const c_void).cast(),
            (&mut p_k as *mut *const c_void).cast(),
            (&mut p_v as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_trie as *mut *const c_void).cast(),
            (&mut p_sinks as *mut *const c_void).cast(),
            (&mut p_gqa as *mut u32).cast(),
            (&mut p_seq as *mut u32).cast(),
            (&mut p_kh as *mut u32).cast(),
            (&mut p_ks as *mut u32).cast(),
            (&mut p_vh as *mut u32).cast(),
            (&mut p_vs as *mut u32).cast(),
            (&mut p_roff as *mut u32).cast(),
            (&mut p_rlen as *mut u32).cast(),
            (&mut p_scale as *mut f32).cast(),
            (&mut p_sw as *mut u32).cast(),
            (&mut p_nh as *mut u32).cast(),
            (&mut p_suflen as *mut u32).cast(),
            (&mut p_hdim as *mut u32).cast(),
            (&mut p_dtype as *mut u32).cast(),
            (&mut p_flags as *mut u32).cast(),
        ];
        let total = num_heads.saturating_mul(suffix_length);
        let block = 64u32;
        let grid = total.div_ceil(block);
        stream
            .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
            .expect("rocm_attention_single_pass failed");
    });
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_encode_input_embedding_lookup(
    kernel: &InputEmbeddingLookupRocmKernel,
    token_ids: SendPtr<u8>,
    values: SendPtr<u8>,
    scales: Option<SendPtr<u8>>,
    zero_points: Option<SendPtr<u8>>,
    biases: Option<SendPtr<u8>>,
    hadamard_factors: Option<SendPtr<u8>>,
    _ladder_indices: Option<SendPtr<u8>>,
    _ladder: Option<SendPtr<u8>>,
    _codebook: Option<SendPtr<u8>>,
    output: SendPtrMut<u8>,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
    command_buffer: &mut RocmCommandBufferEncoding,
) -> bool {
    let Some(dtype) = dtype_code(kernel.T) else {
        return false;
    };
    let use_hadamard = kernel.use_hadamard;
    let had_ptr = hadamard_factors.unwrap_or(SendPtr(std::ptr::null()));

    if kernel.table_kind == EmbeddingTableKind::Quantized {
        let (Some(group_size), Some(mode), Some(method), Some(scales_ptr)) =
            (kernel.group_size, kernel.quantization_mode, kernel.quantization_method, scales)
        else {
            return false;
        };
        let quant_mode = match mode {
            QuantizationMode::U4 => 0u32,
            QuantizationMode::I8 => 1u32,
            QuantizationMode::U8 => 2u32,
        };
        let quant_method = match method {
            QuantizationMethod::ScaleBias => 0u32,
            QuantizationMethod::ScaleZeroPoint => 1u32,
            QuantizationMethod::ScaleSymmetric => 2u32,
        };
        let zp_ptr = zero_points.unwrap_or(SendPtr(std::ptr::null()));
        let bias_ptr = biases.unwrap_or(SendPtr(std::ptr::null()));
        let func = command_buffer.context().kernels.input_embedding_lookup_quant;

        command_buffer.push_gpu_command(move |stream| unsafe {
            let mut p_toks = token_ids.as_ptr() as *const c_void;
            let mut p_vals = values.as_ptr() as *const c_void;
            let mut p_sc = scales_ptr.as_ptr() as *const c_void;
            let mut p_zp = zp_ptr.as_ptr() as *const c_void;
            let mut p_bi = bias_ptr.as_ptr() as *const c_void;
            let mut p_had = had_ptr.as_ptr() as *const c_void;
            let mut p_out = output.as_ptr() as *mut c_void;
            let mut p_bs = batch_size;
            let mut p_vs = vocab_size;
            let mut p_md = model_dim;
            let mut p_gs = group_size;
            let mut p_qm = quant_mode;
            let mut p_qmeth = quant_method;
            let mut p_scale = input_scale;
            let mut p_dtype = dtype;
            let mut p_uhad = u32::from(use_hadamard);
            let mut params: [*mut c_void; 16] = [
                (&mut p_toks as *mut *const c_void).cast(),
                (&mut p_vals as *mut *const c_void).cast(),
                (&mut p_sc as *mut *const c_void).cast(),
                (&mut p_zp as *mut *const c_void).cast(),
                (&mut p_bi as *mut *const c_void).cast(),
                (&mut p_had as *mut *const c_void).cast(),
                (&mut p_out as *mut *mut c_void).cast(),
                (&mut p_bs as *mut u32).cast(),
                (&mut p_vs as *mut u32).cast(),
                (&mut p_md as *mut u32).cast(),
                (&mut p_gs as *mut u32).cast(),
                (&mut p_qm as *mut u32).cast(),
                (&mut p_qmeth as *mut u32).cast(),
                (&mut p_scale as *mut f32).cast(),
                (&mut p_dtype as *mut u32).cast(),
                (&mut p_uhad as *mut u32).cast(),
            ];
            let total = if use_hadamard {
                batch_size.saturating_mul(model_dim / 32)
            } else {
                batch_size.saturating_mul(model_dim)
            };
            let block = 128u32;
            let grid = total.div_ceil(block);
            stream
                .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                .expect("rocm_input_embedding_lookup_quant failed");
        });
        return true;
    }

    if kernel.table_kind != EmbeddingTableKind::Dense {
        return false;
    }
    let func = command_buffer.context().kernels.input_embedding_lookup_dense;

    command_buffer.push_gpu_command(move |stream| unsafe {
        let mut p_toks = token_ids.as_ptr() as *const c_void;
        let mut p_vals = values.as_ptr() as *const c_void;
        let mut p_had = had_ptr.as_ptr() as *const c_void;
        let mut p_out = output.as_ptr() as *mut c_void;
        let mut p_bs = batch_size;
        let mut p_vs = vocab_size;
        let mut p_md = model_dim;
        let mut p_scale = input_scale;
        let mut p_dtype = dtype;
        let mut p_uhad = u32::from(use_hadamard);
        let mut params: [*mut c_void; 10] = [
            (&mut p_toks as *mut *const c_void).cast(),
            (&mut p_vals as *mut *const c_void).cast(),
            (&mut p_had as *mut *const c_void).cast(),
            (&mut p_out as *mut *mut c_void).cast(),
            (&mut p_bs as *mut u32).cast(),
            (&mut p_vs as *mut u32).cast(),
            (&mut p_md as *mut u32).cast(),
            (&mut p_scale as *mut f32).cast(),
            (&mut p_dtype as *mut u32).cast(),
            (&mut p_uhad as *mut u32).cast(),
        ];
        let total = if use_hadamard {
            batch_size.saturating_mul(model_dim / 32)
        } else {
            batch_size.saturating_mul(model_dim)
        };
        let block = 128u32;
        let grid = total.div_ceil(block);
        stream
            .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
            .expect("rocm_input_embedding_lookup_dense failed");
    });
    true
}
