#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// DataType encoding:
// 0 = F32, 1 = F16, 2 = BF16

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffU) > 0x7f800000U) {
        return uint16_t((u >> 16) | 0x0040U);
    }
    uint32_t rounding_bias = 0x00007fffU + ((u >> 16) & 1U);
    return uint16_t((u + rounding_bias) >> 16);
}

__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    _Float16 v;
    __builtin_memcpy(&v, &h, sizeof(v));
    return float(v);
}

__device__ __forceinline__ uint16_t f32_to_f16(float f) {
    _Float16 v = (_Float16)f;
    uint16_t h;
    __builtin_memcpy(&h, &v, sizeof(h));
    return h;
}

__device__ __forceinline__ float read_elem(const void* base, uint32_t dtype, size_t idx) {
    if (dtype == 0) {
        return static_cast<const float*>(base)[idx];
    } else if (dtype == 1) {
        return f16_to_f32(static_cast<const uint16_t*>(base)[idx]);
    } else {
        return bf16_to_f32(static_cast<const uint16_t*>(base)[idx]);
    }
}

__device__ __forceinline__ void write_elem(void* base, uint32_t dtype, size_t idx, float val) {
    if (dtype == 0) {
        static_cast<float*>(base)[idx] = val;
    } else if (dtype == 1) {
        static_cast<uint16_t*>(base)[idx] = f32_to_f16(val);
    } else {
        static_cast<uint16_t*>(base)[idx] = f32_to_bf16(val);
    }
}

__device__ __forceinline__ float round_to_dtype(float val, uint32_t dtype) {
    if (dtype == 0) {
        return val;
    } else if (dtype == 1) {
        return f16_to_f32(f32_to_f16(val));
    } else {
        return bf16_to_f32(f32_to_bf16(val));
    }
}

__device__ __forceinline__ size_t int8_code_index(size_t index, uint32_t code_layout) {
    if (code_layout == 0) {
        return index;
    }
    const size_t NIBBLES_PER_BYTE = 2;
    const size_t CODES_PER_WORD = 4 * NIBBLES_PER_BYTE;
    const size_t NIBBLE_GROUP_SIZE = CODES_PER_WORD / NIBBLES_PER_BYTE;
    size_t offset = index % CODES_PER_WORD;
    return index - offset + (offset % NIBBLES_PER_BYTE) * NIBBLE_GROUP_SIZE + offset / NIBBLES_PER_BYTE;
}

__device__ __forceinline__ float read_a_val(
    const uint8_t* a,
    const float* a_scales,
    uint32_t input_dtype,
    uint32_t a_mode,
    uint32_t a_group_size,
    uint32_t a_code_layout,
    size_t row,
    size_t k_u,
    size_t inner
) {
    if (a_mode == 0) {
        return read_elem(a, input_dtype, row * k_u + inner);
    } else {
        size_t groups = (k_u + a_group_size - 1) / a_group_size;
        size_t group = inner / a_group_size;
        size_t code_idx = int8_code_index(inner, a_code_layout);
        float q = float(reinterpret_cast<const int8_t*>(a)[row * k_u + code_idx]);
        float scale = a_scales[row * groups + group];
        return q * scale;
    }
}

extern "C" __global__ void rocm_matmul_fp(
    const uint8_t* a,
    const float* a_scales,
    const uint8_t* b,
    const uint8_t* bias,
    const uint32_t* gather,
    uint8_t* d,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t b_ld,
    uint32_t b_transpose,
    uint32_t input_dtype,
    uint32_t weights_dtype,
    uint32_t output_dtype,
    uint32_t a_mode,
    uint32_t a_group_size,
    uint32_t a_code_layout,
    float output_scale,
    float soft_cap,
    uint32_t has_soft_cap
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = m * n;
    if (idx >= total) return;

    size_t row = idx / n;
    size_t col = idx % n;
    size_t k_u = k;
    size_t ld = b_ld;
    size_t b_col = (gather != nullptr) ? size_t(gather[row * size_t(n) + col]) : col;

    float accumulator = 0.0f;
    for (size_t inner = 0; inner < k_u; ++inner) {
        float a_val = read_a_val(a, a_scales, input_dtype, a_mode, a_group_size, a_code_layout, row, k_u, inner);
        size_t b_idx = (b_transpose != 0) ? (b_col * ld + inner) : (inner * ld + b_col);
        float b_val = read_elem(b, weights_dtype, b_idx);
        accumulator += a_val * b_val;
    }

    float value = output_scale * accumulator;
    if (bias != nullptr) {
        value += read_elem(bias, weights_dtype, col);
    }
    if (has_soft_cap != 0) {
        value = soft_cap * tanhf(value / soft_cap);
    }
    write_elem(d, output_dtype, row * size_t(n) + col, value);
}

extern "C" __global__ void rocm_matmul_quant(
    const uint8_t* a,
    const float* a_scales,
    const uint32_t* b_weights,
    const uint8_t* b_scales,
    const uint8_t* b_zp,
    const uint8_t* b_biases,
    const uint8_t* output_bias,
    const uint32_t* gather,
    uint8_t* d,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t input_dtype,
    uint32_t weights_dtype,
    uint32_t output_dtype,
    uint32_t a_mode,
    uint32_t a_group_size,
    uint32_t a_code_layout,
    uint32_t b_bits,
    uint32_t b_group_size,
    uint32_t b_signed_codes,
    uint32_t scale_out_stride,
    uint32_t scale_grp_stride,
    uint32_t zp_out_stride,
    uint32_t zp_grp_stride,
    float output_scale,
    float soft_cap,
    uint32_t has_soft_cap
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = m * n;
    if (idx >= total) return;

    size_t row = idx / n;
    size_t col = idx % n;
    size_t k_u = k;
    size_t b_col = (gather != nullptr) ? size_t(gather[row * size_t(n) + col]) : col;
    size_t pack_factor = 32 / b_bits;
    uint32_t code_mask = (1u << b_bits) - 1u;
    float midpoint = float(1u << (b_bits - 1u));

    float accumulator = 0.0f;
    for (size_t inner = 0; inner < k_u; ++inner) {
        float a_val = read_a_val(a, a_scales, input_dtype, a_mode, a_group_size, a_code_layout, row, k_u, inner);

        size_t weight_linear_index = b_col * k_u + inner;
        size_t word_index = weight_linear_index / pack_factor;
        size_t code_index_in_word = weight_linear_index % pack_factor;
        uint32_t bit_offset = uint32_t(code_index_in_word) * b_bits;
        uint32_t word = b_weights[word_index];
        uint8_t weight_code = uint8_t((word >> bit_offset) & code_mask);
        if (b_signed_codes != 0) {
            weight_code ^= uint8_t(1u << (b_bits - 1u));
        }
        float quantized_value = float(weight_code);

        size_t group_index = inner / b_group_size;
        size_t metadata_index = b_col * size_t(scale_out_stride) + group_index * size_t(scale_grp_stride);
        float scale = read_elem(b_scales, weights_dtype, metadata_index);

        float bias_term;
        if (b_zp != nullptr) {
            size_t zp_index = b_col * size_t(zp_out_stride) + group_index * size_t(zp_grp_stride);
            float zp_val;
            if (b_bits == 4) {
                uint8_t byte_val = b_zp[zp_index / 2];
                if ((zp_index & 1) == 0) {
                    zp_val = float(byte_val & 0x0Fu);
                } else {
                    zp_val = float((byte_val >> 4) & 0x0Fu);
                }
            } else {
                zp_val = float(b_zp[zp_index]);
            }
            bias_term = -scale * zp_val;
        } else if (b_biases != nullptr) {
            bias_term = read_elem(b_biases, weights_dtype, metadata_index);
        } else {
            bias_term = -scale * midpoint;
        }

        float b_val = scale * quantized_value + bias_term;
        accumulator += a_val * b_val;
    }

    float value = output_scale * accumulator;
    if (output_bias != nullptr) {
        value += read_elem(output_bias, weights_dtype, col);
    }
    if (has_soft_cap != 0) {
        value = soft_cap * tanhf(value / soft_cap);
    }
    write_elem(d, output_dtype, row * size_t(n) + col, value);
}

__device__ __forceinline__ uint16_t trellis_decode_state(
    const uint8_t* row,
    uint32_t columns,
    uint32_t column,
    uint32_t vector_width,
    uint32_t transition_bits,
    uint32_t restart_columns
) {
    uint32_t block_columns = (restart_columns != 0) ? restart_columns : columns;
    uint32_t block_bits = 16u + (block_columns / vector_width - 1u) * transition_bits;
    uint32_t start_bit = (column / block_columns + 1u) * block_bits - 16u - (column % block_columns) / vector_width * transition_bits;
    uint32_t byte_idx = start_bit / 8u;
    uint32_t bit_offset = start_bit % 8u;
    uint32_t packed = uint32_t(row[byte_idx])
        | (uint32_t(row[byte_idx + 1u]) << 8u)
        | (uint32_t(row[byte_idx + 2u]) << 16u);
    return uint16_t((packed >> bit_offset) & 0xFFFFu);
}

__device__ __forceinline__ int32_t trellis_decode_level(
    const uint8_t* row,
    uint32_t columns,
    uint32_t column,
    uint32_t vector_width,
    uint32_t transition_bits,
    uint32_t restart_columns
) {
    uint32_t state = uint32_t(trellis_decode_state(row, columns, column, vector_width, transition_bits, restart_columns));
    uint32_t hash = state * 0xCFCCB83Fu + 0x584B4AA3u;
    hash = (hash ^ (hash >> 16u)) * 0x85EBCA6Bu;
    uint32_t hash_byte = ((hash ^ (hash >> 16u)) >> (8u * (column % vector_width))) & 0xFFu;
    uint32_t field_sum = (hash_byte & 3u) + ((hash_byte >> 2u) & 3u) + ((hash_byte >> 4u) & 3u) + ((hash_byte >> 6u) & 3u);
    return int32_t(8u * field_sum + ((3u * (hash_byte & 15u)) & 15u)) - 54;
}

extern "C" __global__ void rocm_matmul_trellis(
    const int8_t* activations,
    const float* column_group_sums,
    const float* activation_scales,
    const uint8_t* codes,
    const float* row_scales,
    const float* codebook,
    uint16_t* output,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t output_stride,
    uint32_t row_bytes,
    uint32_t vector_width,
    uint32_t transition_bits,
    uint32_t restart_columns
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = m * n;
    if (idx >= total) return;

    uint32_t token = idx / n;
    uint32_t row = idx % n;
    const int8_t* act_row = activations + size_t(token) * k;
    const uint8_t* code_row = codes + size_t(row) * row_bytes;

    int32_t level_dot = 0;
    for (uint32_t col = 0; col < k; ++col) {
        int32_t level = trellis_decode_level(code_row, k, col, vector_width, transition_bits, restart_columns);
        level_dot += level * int32_t(act_row[col]);
    }

    const float* sums = column_group_sums + size_t(token) * 4u;
    float offsets_dot = (((sums[0] * codebook[1]) + sums[1] * codebook[2]) + sums[2] * codebook[3]) + sums[3] * codebook[4];
    float dot = fmaf(float(level_dot), codebook[0], offsets_dot);
    float val = dot * row_scales[row] * activation_scales[token];
    output[size_t(token) * output_stride + row] = f32_to_bf16(val);
}

__device__ __forceinline__ float apply_activation(float x_float, uint32_t act_type, float alpha) {
    if (act_type == 0) { // SILU
        return x_float / (1.0f + expf(-alpha * x_float));
    } else if (act_type == 1) { // GELUApprox
        float tan_arg = 0.7978846f * (x_float + 0.044715f * x_float * x_float * x_float);
        return 0.5f * x_float * (1.0f + tanhf(tan_arg));
    } else if (act_type == 2) { // GELUExact
        return 0.5f * x_float * (1.0f + erff(x_float * 0.70710678118654752440f));
    } else if (act_type == 3) { // IDENTITY
        return x_float;
    } else { // SOFTPLUS
        if (x_float > 20.0f) return x_float;
        return logf(1.0f + expf(x_float));
    }
}

extern "C" __global__ void rocm_activation(
    const uint8_t* input,
    uint8_t* output,
    uint32_t n,
    uint32_t dtype,
    uint32_t act_type
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = read_elem(input, dtype, i);
    float y = apply_activation(x, act_type, 1.0f);
    write_elem(output, dtype, i, y);
}

extern "C" __global__ void rocm_tensor_add_bias(
    const uint8_t* input,
    const uint8_t* bias,
    uint8_t* output,
    uint32_t num_cols,
    uint32_t length,
    uint32_t t_dtype,
    uint32_t bias_dtype
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    uint32_t col = i % num_cols;
    float v = read_elem(input, t_dtype, i);
    float b = read_elem(bias, bias_dtype, col);
    write_elem(output, t_dtype, i, v + b);
}

extern "C" __global__ void rocm_tensor_add_scale(
    const uint8_t* input,
    const uint8_t* bias,
    uint8_t* output,
    uint32_t num_cols,
    uint32_t length,
    float scale,
    uint32_t dtype
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    uint32_t col = i % num_cols;
    float v = read_elem(input, dtype, i);
    float b = read_elem(bias, dtype, col);
    write_elem(output, dtype, i, (v + b) * scale);
}

extern "C" __global__ void rocm_logit_transform(
    uint8_t* logits,
    uint32_t length,
    float scale,
    float soft_cap,
    uint32_t has_soft_cap,
    uint32_t dtype
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    float v = read_elem(logits, dtype, i) * scale;
    if (has_soft_cap != 0) {
        v = tanhf(v / soft_cap) * soft_cap;
    }
    write_elem(logits, dtype, i, v);
}

extern "C" __global__ void rocm_sigmoid_gate(
    const uint8_t* gate,
    uint8_t* output,
    uint32_t gate_dim,
    uint32_t batch_dim,
    uint32_t gate_row_stride,
    uint32_t dtype
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch_dim * gate_dim;
    if (idx >= total) return;
    uint32_t batch_idx = idx / gate_dim;
    uint32_t col_idx = idx % gate_dim;
    float g = read_elem(gate, dtype, size_t(batch_idx) * gate_row_stride + col_idx);
    float sigmoid = 1.0f / (1.0f + expf(-g));
    float out = read_elem(output, dtype, idx);
    write_elem(output, dtype, idx, out * sigmoid);
}

extern "C" __global__ void rocm_pooling_mean(
    const uint8_t* input,
    uint8_t* output,
    uint32_t seq_len,
    uint32_t hidden_dim,
    uint32_t batch_size,
    uint32_t dtype
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch_size * hidden_dim;
    if (idx >= total) return;
    uint32_t batch_idx = idx / hidden_dim;
    uint32_t dim_idx = idx % hidden_dim;
    float sum = 0.0f;
    for (uint32_t s = 0; s < seq_len; ++s) {
        size_t pos = (size_t(batch_idx) * seq_len + s) * hidden_dim + dim_idx;
        sum += read_elem(input, dtype, pos);
    }
    write_elem(output, dtype, idx, sum / float(seq_len));
}

extern "C" __global__ void rocm_softmax(
    uint8_t* values,
    const uint8_t* sinks,
    uint32_t row_length,
    uint32_t outer_dim,
    uint32_t batch_dim,
    uint32_t has_sinks,
    uint32_t dtype
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_rows = outer_dim * batch_dim;
    if (row_idx >= total_rows) return;

    uint32_t outer_index = row_idx / batch_dim;
    size_t row_offset = size_t(row_idx) * row_length;
    float sink = (has_sinks != 0) ? read_elem(sinks, dtype, outer_index) : -INFINITY;
    float max_value = (has_sinks != 0) ? sink : -INFINITY;

    for (uint32_t i = 0; i < row_length; ++i) {
        float v = read_elem(values, dtype, row_offset + i);
        if (v > max_value) max_value = v;
    }

    float sum_exp = (has_sinks != 0) ? expf(sink - max_value) : 0.0f;
    for (uint32_t i = 0; i < row_length; ++i) {
        float p = expf(read_elem(values, dtype, row_offset + i) - max_value);
        sum_exp += p;
        write_elem(values, dtype, row_offset + i, p);
    }

    float inv = 1.0f / sum_exp;
    for (uint32_t i = 0; i < row_length; ++i) {
        float p = read_elem(values, dtype, row_offset + i);
        write_elem(values, dtype, row_offset + i, p * inv);
    }
}

__device__ __forceinline__ void hadamard_block_32(float* values) {
    for (uint32_t stride = 1; stride < 32; stride <<= 1) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
            if ((lane & stride) == 0) {
                float a = values[lane];
                float b = values[lane | stride];
                values[lane] = a + b;
                values[lane | stride] = a - b;
            }
        }
    }
    const float scale = 0.1767766952966368811f; // 1 / sqrt(32)
    for (uint32_t i = 0; i < 32; ++i) {
        values[i] *= scale;
    }
}

extern "C" __global__ void rocm_normalization(
    const uint8_t* input,
    const uint8_t* scales,
    const uint8_t* biases,
    uint8_t* output,
    uint8_t* shortcut,
    const int32_t* hadamard_factors,
    uint32_t batch_size,
    uint32_t element_count,
    float epsilon,
    float scale_offset,
    float post_layer_scalar,
    uint32_t in_dtype,
    uint32_t affine_dtype,
    uint32_t out_dtype,
    uint32_t flags
) {
    uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batch_size) return;

    bool subtract_mean = (flags & (1u << 1)) != 0;
    bool full_layer = (flags & (1u << 2)) != 0;
    bool copy_to_shortcut = (flags & (1u << 3)) != 0;
    bool residual_add = (flags & (1u << 4)) != 0;
    bool use_hadamard = (flags & (1u << 5)) != 0;
    bool scale_residual_sum = (flags & (1u << 6)) != 0;
    bool scale_output = (flags & (1u << 7)) != 0;
    bool has_biases = (flags & (1u << 8)) != 0;
    bool has_scales = (flags & (1u << 9)) != 0;

    size_t batch_offset = size_t(batch) * element_count;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t i = 0; i < element_count; ++i) {
        float val = read_elem(input, in_dtype, batch_offset + i);
        if (copy_to_shortcut) {
            if (residual_add) {
                float skip_val = read_elem(shortcut, in_dtype, batch_offset + i);
                val = round_to_dtype(val + skip_val, in_dtype);
                if (scale_residual_sum) {
                    val = round_to_dtype(val * post_layer_scalar, in_dtype);
                }
            }
            write_elem(shortcut, in_dtype, batch_offset + i, val);
        }
        if (subtract_mean) {
            sum += val;
        }
        sum_sq += val * val;
    }

    float count_f = float(element_count);
    float mean = subtract_mean ? (sum / count_f) : 0.0f;
    float variance = sum_sq / count_f - mean * mean;
    float rms_inv = 1.0f / sqrtf(variance + epsilon);
    float post_scalar_rounded = round_to_dtype(post_layer_scalar, out_dtype);

    if (!use_hadamard) {
        for (uint32_t i = 0; i < element_count; ++i) {
            float in_val = residual_add
                ? read_elem(shortcut, in_dtype, batch_offset + i)
                : read_elem(input, in_dtype, batch_offset + i);
            float normalized = (in_val - mean) * rms_inv;
            float res;
            if (has_scales) {
                float sc = read_elem(scales, affine_dtype, i);
                if (full_layer) {
                    res = round_to_dtype(normalized * (sc + scale_offset), out_dtype);
                } else {
                    float norm_out = round_to_dtype(normalized, out_dtype);
                    float sc_out = round_to_dtype(sc + scale_offset, out_dtype);
                    res = round_to_dtype(norm_out * sc_out, out_dtype);
                }
            } else {
                res = round_to_dtype(normalized, out_dtype);
            }
            if (has_biases) {
                float b = read_elem(biases, affine_dtype, i);
                res = round_to_dtype(res + b, out_dtype);
            }
            if (scale_output) {
                res = round_to_dtype(res * post_scalar_rounded, out_dtype);
            }
            write_elem(output, out_dtype, batch_offset + i, res);
        }
    } else {
        for (uint32_t block_start = 0; block_start < element_count; block_start += 32) {
            float block[32];
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t i = block_start + lane;
                float in_val = residual_add
                    ? read_elem(shortcut, in_dtype, batch_offset + i)
                    : read_elem(input, in_dtype, batch_offset + i);
                float normalized = (in_val - mean) * rms_inv;
                float res;
                if (has_scales) {
                    float sc = read_elem(scales, affine_dtype, i);
                    if (full_layer) {
                        res = round_to_dtype(normalized * (sc + scale_offset), out_dtype);
                    } else {
                        float norm_out = round_to_dtype(normalized, out_dtype);
                        float sc_out = round_to_dtype(sc + scale_offset, out_dtype);
                        res = round_to_dtype(norm_out * sc_out, out_dtype);
                    }
                } else {
                    res = round_to_dtype(normalized, out_dtype);
                }
                if (has_biases) {
                    float b = read_elem(biases, affine_dtype, i);
                    res = round_to_dtype(res + b, out_dtype);
                }
                block[lane] = res * float(hadamard_factors[i]);
            }
            hadamard_block_32(block);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t i = block_start + lane;
                float res = round_to_dtype(block[lane], out_dtype);
                if (scale_output) {
                    res = round_to_dtype(res * post_scalar_rounded, out_dtype);
                }
                write_elem(output, out_dtype, batch_offset + i, res);
            }
        }
    }
}

extern "C" __global__ void rocm_qkv_norm(
    const uint8_t* qkvg_input,
    const uint8_t* scales,
    uint8_t* qkvg_output,
    uint32_t batch_size,
    uint32_t input_row_stride,
    uint32_t head_dim,
    float epsilon,
    float scale_offset,
    uint32_t head_offset,
    uint32_t head_count,
    uint32_t in_dtype,
    uint32_t scale_dtype,
    uint32_t out_dtype,
    uint32_t flags
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch_size * head_count;
    if (idx >= total) return;

    uint32_t batch = idx / head_count;
    uint32_t head = idx % head_count;
    bool full_layer = (flags & 1u) != 0;
    bool has_scales = (flags & 2u) != 0;

    size_t offset = size_t(batch) * input_row_stride + size_t(head_offset + head) * head_dim;
    float total_sum = 0.0f;
    for (uint32_t i = 0; i < head_dim; ++i) {
        float val = read_elem(qkvg_input, in_dtype, offset + i);
        total_sum += val * val;
    }
    float mean_square = total_sum / float(head_dim);
    float rms_norm = 1.0f / sqrtf(mean_square + epsilon);

    for (uint32_t i = 0; i < head_dim; ++i) {
        float val = read_elem(qkvg_input, in_dtype, offset + i);
        float normalized = val * rms_norm;
        float res;
        if (!has_scales) {
            res = round_to_dtype(normalized, out_dtype);
        } else {
            float sc = read_elem(scales, scale_dtype, i);
            if (full_layer) {
                res = round_to_dtype(normalized * (sc + scale_offset), out_dtype);
            } else {
                float norm_low = round_to_dtype(normalized, out_dtype);
                float sc_low = round_to_dtype(sc + scale_offset, out_dtype);
                res = round_to_dtype(norm_low * sc_low, out_dtype);
            }
        }
        write_elem(qkvg_output, out_dtype, offset + i, res);
    }
}

extern "C" __global__ void rocm_kv_cache_update(
    uint8_t* in_place_keys,
    uint8_t* in_place_values,
    const uint32_t* copies,
    uint32_t copy_count,
    uint32_t element_dim,
    uint32_t dtype
) {
    uint32_t elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= element_dim) return;
    for (uint32_t i = 0; i < copy_count; ++i) {
        uint32_t src = copies[i * 2];
        uint32_t dst = copies[i * 2 + 1];
        size_t src_idx = size_t(src) * element_dim + elem_idx;
        size_t dst_idx = size_t(dst) * element_dim + elem_idx;
        if (dtype == 0) {
            reinterpret_cast<float*>(in_place_keys)[dst_idx] = reinterpret_cast<const float*>(in_place_keys)[src_idx];
            reinterpret_cast<float*>(in_place_values)[dst_idx] = reinterpret_cast<const float*>(in_place_values)[src_idx];
        } else {
            reinterpret_cast<uint16_t*>(in_place_keys)[dst_idx] = reinterpret_cast<const uint16_t*>(in_place_keys)[src_idx];
            reinterpret_cast<uint16_t*>(in_place_values)[dst_idx] = reinterpret_cast<const uint16_t*>(in_place_values)[src_idx];
        }
    }
}

extern "C" __global__ void rocm_gated_act_mul_fp(
    const uint8_t* act_operand,
    const uint8_t* value_operand,
    uint8_t* fp_out,
    const int32_t* hadamard_factors,
    uint32_t gated_dim,
    uint32_t batch_dim,
    uint32_t value_offset,
    uint32_t value_row_stride,
    uint32_t act_type,
    float activation_alpha,
    float gate_clip_min,
    float gate_clip_max,
    float value_clip_min,
    float value_clip_max,
    uint32_t dtype,
    uint32_t flags
) {
    bool interleaved = (flags & 1u) != 0;
    bool use_hadamard = (flags & 2u) != 0;
    bool custom_alpha = (flags & 4u) != 0;
    bool clip_gate = (flags & 8u) != 0;
    bool clip_value = (flags & 16u) != 0;

    if (!use_hadamard) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t total = batch_dim * gated_dim;
        if (idx >= total) return;
        uint32_t batch = idx / gated_dim;
        uint32_t gated = idx % gated_dim;

        size_t act_idx;
        float val;
        if (interleaved) {
            size_t base = size_t(batch) * 2u * gated_dim;
            act_idx = base + gated_dim + gated;
            val = read_elem(act_operand, dtype, base + gated);
        } else {
            size_t value_idx = size_t(batch) * value_row_stride + value_offset + gated;
            act_idx = size_t(batch) * gated_dim + gated;
            val = read_elem(value_operand, dtype, value_idx);
        }

        float gate = read_elem(act_operand, dtype, act_idx);
        if (clip_gate) {
            gate = fminf(fmaxf(gate, gate_clip_min), gate_clip_max);
            gate = round_to_dtype(gate, dtype);
        }
        float alpha = (custom_alpha && act_type == 0) ? activation_alpha : 1.0f;
        float activated = round_to_dtype(apply_activation(gate, act_type, alpha), dtype);
        if (clip_value) {
            val = fminf(fmaxf(val, value_clip_min), value_clip_max);
            val = round_to_dtype(val, dtype);
        }
        float res = round_to_dtype(val * activated, dtype);
        write_elem(fp_out, dtype, idx, res);
    } else {
        uint32_t stripe_idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t stripes_per_row = gated_dim / 32u;
        uint32_t total_stripes = batch_dim * stripes_per_row;
        if (stripe_idx >= total_stripes) return;

        uint32_t batch = stripe_idx / stripes_per_row;
        uint32_t stripe_start = (stripe_idx % stripes_per_row) * 32u;
        float stripe[32];

        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t gated = stripe_start + lane;
            size_t act_idx;
            float val;
            if (interleaved) {
                size_t base = size_t(batch) * 2u * gated_dim;
                act_idx = base + gated_dim + gated;
                val = read_elem(act_operand, dtype, base + gated);
            } else {
                size_t value_idx = size_t(batch) * value_row_stride + value_offset + gated;
                act_idx = size_t(batch) * gated_dim + gated;
                val = read_elem(value_operand, dtype, value_idx);
            }
            float gate = read_elem(act_operand, dtype, act_idx);
            if (clip_gate) {
                gate = fminf(fmaxf(gate, gate_clip_min), gate_clip_max);
                gate = round_to_dtype(gate, dtype);
            }
            float alpha = (custom_alpha && act_type == 0) ? activation_alpha : 1.0f;
            float activated = round_to_dtype(apply_activation(gate, act_type, alpha), dtype);
            if (clip_value) {
                val = fminf(fmaxf(val, value_clip_min), value_clip_max);
                val = round_to_dtype(val, dtype);
            }
            float res = round_to_dtype(val * activated, dtype);
            stripe[lane] = res * float(hadamard_factors[gated]);
        }
        hadamard_block_32(stripe);
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t gated = stripe_start + lane;
            write_elem(fp_out, dtype, size_t(batch) * gated_dim + gated, stripe[lane]);
        }
    }
}

__device__ __forceinline__ bool rocm_should_use_key(
    bool is_ring,
    uint32_t ring_offset,
    uint32_t ring_length,
    bool is_trie,
    const uint32_t* trie,
    bool is_sliding_window,
    uint32_t sliding_window_size,
    uint32_t q_seq_idx,
    uint32_t prefix_length,
    uint32_t suffix_position,
    uint32_t query_position,
    uint32_t i,
    bool is_causal
) {
    bool use_key = true;
    uint32_t key_position;

    if (i >= prefix_length) {
        uint32_t key_position_in_suffix = i - prefix_length;
        if (is_trie) {
            uint32_t trie_start = trie[key_position_in_suffix * 3 + 0];
            uint32_t trie_end = trie[key_position_in_suffix * 3 + 1];
            uint32_t height = trie[key_position_in_suffix * 3 + 2];
            key_position = suffix_position + height;
            if (is_causal) {
                use_key = use_key && (q_seq_idx >= trie_start && q_seq_idx <= trie_end);
            }
        } else {
            key_position = suffix_position + key_position_in_suffix;
            if (is_causal) {
                use_key = use_key && (key_position_in_suffix <= q_seq_idx);
            }
        }
    } else {
        if (is_ring) {
            key_position = (prefix_length + i - ring_offset) % prefix_length;
            use_key = use_key && (key_position < ring_length);
        } else {
            key_position = i;
        }
    }

    if (is_sliding_window) {
        if (is_causal) {
            use_key = use_key && (key_position <= query_position && (query_position - key_position) < sliding_window_size);
        } else if (key_position <= query_position) {
            use_key = use_key && ((query_position - key_position) <= sliding_window_size / 2u);
        } else {
            use_key = use_key && ((key_position - query_position) <= sliding_window_size / 2u);
        }
    }
    return use_key;
}

extern "C" __global__ void rocm_attention_single_pass(
    const uint8_t* queries,
    const uint8_t* keys,
    const uint8_t* values,
    uint8_t* out,
    const uint32_t* trie,
    const uint8_t* sinks,
    uint32_t gqa_factor,
    uint32_t sequence_length,
    uint32_t k_head_stride,
    uint32_t k_seq_stride,
    uint32_t v_head_stride,
    uint32_t v_seq_stride,
    uint32_t ring_offset,
    uint32_t ring_length,
    float scale,
    uint32_t sliding_window_size,
    uint32_t num_heads,
    uint32_t suffix_length,
    uint32_t head_dim,
    uint32_t dtype,
    uint32_t flags
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = num_heads * suffix_length;
    if (idx >= total || head_dim > 512) return;

    uint32_t head_idx = idx / suffix_length;
    uint32_t q_seq_idx = idx % suffix_length;

    bool has_sinks = (flags & 1u) != 0;
    bool is_ring = (flags & 2u) != 0;
    bool is_causal = (flags & 4u) != 0;
    bool is_trie = (flags & 8u) != 0;
    bool is_sliding_window = (flags & 16u) != 0;

    uint32_t prefix_length = sequence_length - suffix_length;
    uint32_t suffix_position = is_ring ? ring_length : prefix_length;
    uint32_t query_position = is_trie ? (suffix_position + trie[q_seq_idx * 3 + 2]) : (suffix_position + q_seq_idx);

    uint32_t kv_head_idx = head_idx / gqa_factor;
    size_t o_offset = (size_t(q_seq_idx) * num_heads + head_idx) * head_dim;
    size_t q_offset = (size_t(head_idx) * suffix_length + q_seq_idx) * head_dim;
    size_t k_base = size_t(kv_head_idx) * k_head_stride;
    size_t v_base = size_t(kv_head_idx) * v_head_stride;

    float q[512];
    float o[512];
    for (uint32_t j = 0; j < head_dim; ++j) {
        q[j] = scale * read_elem(queries, dtype, q_offset + j);
        o[j] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    if (has_sinks) {
        max_score = read_elem(sinks, dtype, head_idx % num_heads);
        sum_exp_score = 1.0f;
    }

    for (uint32_t i = 0; i < sequence_length; ++i) {
        if (rocm_should_use_key(
                is_ring, ring_offset, ring_length, is_trie, trie, is_sliding_window, sliding_window_size,
                q_seq_idx, prefix_length, suffix_position, query_position, i, is_causal)) {
            size_t k_off = k_base + size_t(i) * k_seq_stride;
            float score = 0.0f;
            for (uint32_t j = 0; j < head_dim; ++j) {
                score += q[j] * read_elem(keys, dtype, k_off + j);
            }
            float new_max = fmaxf(max_score, score);
            float factor = expf(max_score - new_max);
            float exp_score = expf(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            size_t v_off = v_base + size_t(i) * v_seq_stride;
            for (uint32_t j = 0; j < head_dim; ++j) {
                o[j] = o[j] * factor + exp_score * read_elem(values, dtype, v_off + j);
            }
        }
    }

    for (uint32_t j = 0; j < head_dim; ++j) {
        write_elem(out, dtype, o_offset + j, o[j] / sum_exp_score);
    }
}

extern "C" __global__ void rocm_input_embedding_lookup_dense(
    const uint32_t* token_ids,
    const uint8_t* values,
    const int32_t* hadamard_factors,
    uint8_t* output,
    uint32_t batch_size,
    uint32_t vocab_size,
    uint32_t model_dim,
    float input_scale,
    uint32_t dtype,
    uint32_t use_hadamard
) {
    float dense_scale = round_to_dtype(input_scale, dtype);
    if (use_hadamard == 0) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t total = batch_size * model_dim;
        if (idx >= total) return;
        uint32_t batch = idx / model_dim;
        uint32_t col = idx % model_dim;
        uint32_t token = token_ids[batch];
        if (token >= vocab_size) {
            write_elem(output, dtype, idx, 0.0f);
        } else {
            float v = read_elem(values, dtype, size_t(token) * model_dim + col);
            write_elem(output, dtype, idx, round_to_dtype(v * dense_scale, dtype));
        }
    } else {
        uint32_t stripe_idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t stripes_per_row = model_dim / 32u;
        uint32_t total_stripes = batch_size * stripes_per_row;
        if (stripe_idx >= total_stripes) return;
        uint32_t batch = stripe_idx / stripes_per_row;
        uint32_t block_start = (stripe_idx % stripes_per_row) * 32u;
        uint32_t token = token_ids[batch];
        if (token >= vocab_size) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                write_elem(output, dtype, size_t(batch) * model_dim + block_start + lane, 0.0f);
            }
        } else {
            float block[32];
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t col = block_start + lane;
                float v = read_elem(values, dtype, size_t(token) * model_dim + col);
                block[lane] = round_to_dtype(v * dense_scale, dtype);
            }
            hadamard_block_32(block);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t col = block_start + lane;
                write_elem(output, dtype, size_t(batch) * model_dim + col, block[lane] * float(hadamard_factors[col]));
            }
        }
    }
}
