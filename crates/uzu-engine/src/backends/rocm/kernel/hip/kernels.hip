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
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid & 7u;
    uint32_t idx = (blockIdx.x * blockDim.x + tid) >> 3u;
    uint32_t total = m * n;
    bool valid = (idx < total);

    __shared__ float s_acc[128];
    float accumulator = 0.0f;

    size_t row = valid ? (idx / n) : 0;
    size_t col = valid ? (idx % n) : 0;
    size_t k_u = k;
    size_t b_col = (valid && gather != nullptr) ? size_t(gather[row * size_t(n) + col]) : col;
    size_t pack_factor = 32 / b_bits;
    uint32_t code_mask = (1u << b_bits) - 1u;
    float midpoint = float(1u << (b_bits - 1u));

    if (valid) {
        if (b_group_size > 0 && (k_u % b_group_size) == 0 && (b_group_size % pack_factor) == 0) {
            size_t num_b_groups = k_u / b_group_size;
            size_t words_per_group = b_group_size / pack_factor;
            size_t a_groups = (a_mode != 0 && a_group_size > 0) ? ((k_u + a_group_size - 1) / a_group_size) : 0;
            const int8_t* a_row_i8 = reinterpret_cast<const int8_t*>(a) + row * k_u;
            for (size_t group_index = 0; group_index < num_b_groups; ++group_index) {
                size_t metadata_index = b_col * size_t(scale_out_stride) + group_index * size_t(scale_grp_stride);
                float scale = read_elem(b_scales, weights_dtype, metadata_index);
                float bias_term;
                if (b_zp != nullptr) {
                    size_t zp_index = b_col * size_t(zp_out_stride) + group_index * size_t(zp_grp_stride);
                    float zp_val;
                    if (b_bits == 4) {
                        uint8_t byte_val = b_zp[zp_index / 2];
                        zp_val = ((zp_index & 1) == 0) ? float(byte_val & 0x0Fu) : float((byte_val >> 4) & 0x0Fu);
                    } else {
                        zp_val = float(b_zp[zp_index]);
                    }
                    bias_term = -scale * zp_val;
                } else if (b_biases != nullptr) {
                    bias_term = read_elem(b_biases, weights_dtype, metadata_index);
                } else {
                    bias_term = -scale * midpoint;
                }

                size_t group_start = group_index * b_group_size;
                for (size_t w = lane; w < words_per_group; w += 8u) {
                    size_t inner_base = group_start + w * pack_factor;
                    uint32_t word = b_weights[(b_col * k_u + inner_base) / pack_factor];
                    for (size_t c = 0; c < pack_factor; ++c) {
                        size_t inner = inner_base + c;
                        float a_val;
                        if (a_mode == 0) {
                            a_val = read_elem(a, input_dtype, row * k_u + inner);
                        } else {
                            size_t a_grp = inner / a_group_size;
                            size_t code_idx = int8_code_index(inner, a_code_layout);
                            a_val = float(a_row_i8[code_idx]) * a_scales[row * a_groups + a_grp];
                        }
                        uint8_t weight_code = uint8_t((word >> (uint32_t(c) * b_bits)) & code_mask);
                        if (b_signed_codes != 0) {
                            weight_code ^= uint8_t(1u << (b_bits - 1u));
                        }
                        float b_val = scale * float(weight_code) + bias_term;
                        accumulator += a_val * b_val;
                    }
                }
            }
        } else if (lane == 0) {
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
        }
    }

    s_acc[tid] = accumulator;
    __syncthreads();
    if (lane < 4u) s_acc[tid] += s_acc[tid + 4u];
    __syncthreads();
    if (lane < 2u) s_acc[tid] += s_acc[tid + 2u];
    __syncthreads();

    if (valid && lane == 0u) {
        float sum = s_acc[tid] + s_acc[tid + 1u];
        float value = output_scale * sum;
        if (output_bias != nullptr) {
            value += read_elem(output_bias, weights_dtype, col);
        }
        if (has_soft_cap != 0) {
            value = soft_cap * tanhf(value / soft_cap);
        }
        write_elem(d, output_dtype, row * size_t(n) + col, value);
    }
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

__device__ __forceinline__ uint8_t rocm_read_u4(const uint8_t* values, size_t nibble) {
    return (values[nibble / 2u] >> (4u * (nibble & 1u))) & 0x0Fu;
}

__device__ __forceinline__ float rocm_load_quant_emb(
    const uint8_t* values,
    const uint8_t* scales,
    const uint8_t* zero_points,
    const uint8_t* biases,
    size_t token,
    uint32_t column,
    uint32_t dim,
    uint32_t group_size,
    uint32_t quant_mode, // 0=U4, 1=I8, 2=U8
    uint32_t quant_method, // 0=ScaleBias, 1=ScaleZeroPoint, 2=ScaleSymmetric
    float input_scale,
    uint32_t dtype
) {
    uint32_t num_groups = (dim + group_size - 1u) / group_size;
    uint32_t pack_div = (quant_mode == 0u) ? 2u : 1u;
    size_t weights_stride = dim / pack_div;
    uint32_t group = column / group_size;
    size_t index = token * num_groups + group;
    float scale = read_elem(scales, dtype, index);
    size_t offset = token * weights_stride;
    float code;
    if (quant_mode == 0u) {
        code = float(rocm_read_u4(values, 2u * offset + column));
    } else if (quant_mode == 1u) {
        code = float(reinterpret_cast<const int8_t*>(values)[offset + column]);
    } else {
        code = float(values[offset + column]);
    }
    float bias;
    if (quant_method == 0u) {
        bias = read_elem(biases, dtype, index);
    } else if (quant_method == 1u) {
        uint8_t zp;
        if (quant_mode == 0u) {
            size_t zp_groups_padded = (num_groups + 1u) / 2u;
            zp = rocm_read_u4(zero_points, 2u * token * zp_groups_padded + group);
        } else {
            zp = zero_points[index];
        }
        bias = -scale * float(zp);
    } else {
        float midpoint = (quant_mode == 0u) ? 8.0f : 128.0f;
        bias = -scale * midpoint;
    }
    return round_to_dtype((scale * code + bias) * input_scale, dtype);
}

extern "C" __global__ void rocm_input_embedding_lookup_quant(
    const uint32_t* token_ids,
    const uint8_t* values,
    const uint8_t* scales,
    const uint8_t* zero_points,
    const uint8_t* biases,
    const int32_t* hadamard_factors,
    uint8_t* output,
    uint32_t batch_size,
    uint32_t vocab_size,
    uint32_t model_dim,
    uint32_t group_size,
    uint32_t quant_mode,
    uint32_t quant_method,
    float input_scale,
    uint32_t dtype,
    uint32_t use_hadamard
) {
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
            float v = rocm_load_quant_emb(
                values, scales, zero_points, biases, token, col, model_dim,
                group_size, quant_mode, quant_method, input_scale, dtype
            );
            write_elem(output, dtype, idx, v);
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
                block[lane] = rocm_load_quant_emb(
                    values, scales, zero_points, biases, token, col, model_dim,
                    group_size, quant_mode, quant_method, input_scale, dtype
                );
            }
            hadamard_block_32(block);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t col = block_start + lane;
                write_elem(output, dtype, size_t(batch) * model_dim + col, block[lane] * float(hadamard_factors[col]));
            }
        }
    }
}

__device__ __forceinline__ float min_max_symmetric_divisor_device(const float* values, uint32_t len) {
    float min_v = INFINITY;
    float max_v = -INFINITY;
    for (uint32_t i = 0; i < len; ++i) {
        float v = values[i];
        min_v = fminf(min_v, v);
        max_v = fmaxf(max_v, v);
    }
    float magnitude = fmaxf(fabsf(min_v), fabsf(max_v));
    if (isfinite(magnitude) && magnitude > 0.0f) {
        return magnitude / 127.0f;
    }
    return 1.0f;
}

__device__ __forceinline__ int8_t quantize_symmetric_i8_device(float value, float divisor) {
    float q = roundf(value / divisor);
    if (isnan(q)) return 0;
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return int8_t(int32_t(q));
}

extern "C" __global__ void rocm_activation_transform(
    const uint8_t* input,
    uint8_t* fp_out,
    const uint8_t* bias,
    int8_t* q_out,
    float* scales_out,
    int32_t* group_sums_out,
    const int32_t* rht_factors,
    uint32_t batch_size,
    uint32_t element_count,
    uint32_t scale_group_size,
    uint32_t sum_group_size,
    uint32_t t_dtype,
    uint32_t bias_dtype,
    uint32_t flags
) {
    bool input_rht = (flags & 1u) != 0;
    bool quantize = (flags & 2u) != 0;
    bool emit_group_sums = (flags & 4u) != 0;
    uint32_t code_layout = ((flags & 8u) != 0) ? 1u : 0u;
    bool has_bias = (flags & 16u) != 0;

    if (!quantize) {
        uint32_t stripe_idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t stripes_per_row = element_count / 32u;
        uint32_t total_stripes = batch_size * stripes_per_row;
        if (stripe_idx >= total_stripes) return;

        uint32_t row = stripe_idx / stripes_per_row;
        uint32_t stripe_start = (stripe_idx % stripes_per_row) * 32u;
        size_t row_offset = size_t(row) * element_count;

        float stripe[32];
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t index = stripe_start + lane;
            float value = read_elem(input, t_dtype, row_offset + index);
            float factor = float(rht_factors[index]);
            stripe[lane] = input_rht ? (value * factor) : value;
        }
        hadamard_block_32(stripe);
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t index = stripe_start + lane;
            float factor = float(rht_factors[index]);
            float value = input_rht ? stripe[lane] : (stripe[lane] * factor);
            if (has_bias) {
                value = round_to_dtype(value, t_dtype) + read_elem(bias, bias_dtype, index);
            }
            write_elem(fp_out, t_dtype, row_offset + index, value);
        }
    } else {
        uint32_t grp_idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t groups_per_row = element_count / scale_group_size;
        uint32_t total_groups = batch_size * groups_per_row;
        if (grp_idx >= total_groups || scale_group_size > 256u) return;

        uint32_t row = grp_idx / groups_per_row;
        uint32_t scale_group_index = grp_idx % groups_per_row;
        size_t row_offset = size_t(row) * element_count;
        uint32_t group_col_start = scale_group_index * scale_group_size;

        float transformed[256];
        for (uint32_t sub = 0; sub < scale_group_size; sub += 32u) {
            float stripe[32];
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t index = group_col_start + sub + lane;
                float value = read_elem(input, t_dtype, row_offset + index);
                float factor = float(rht_factors[index]);
                stripe[lane] = input_rht ? (value * factor) : value;
            }
            hadamard_block_32(stripe);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                uint32_t index = group_col_start + sub + lane;
                float factor = float(rht_factors[index]);
                transformed[sub + lane] = input_rht ? stripe[lane] : (stripe[lane] * factor);
            }
        }

        float scale = min_max_symmetric_divisor_device(transformed, scale_group_size);
        scales_out[size_t(row) * groups_per_row + scale_group_index] = scale;

        uint32_t sums_per_row = emit_group_sums ? (element_count / sum_group_size) : 0u;
        uint32_t sum_grp_start = emit_group_sums ? (group_col_start / sum_group_size) : 0u;
        uint32_t sum_grp_count = emit_group_sums ? (scale_group_size / sum_group_size) : 0u;
        for (uint32_t s = 0; s < sum_grp_count; ++s) {
            group_sums_out[size_t(row) * sums_per_row + sum_grp_start + s] = 0;
        }

        int8_t* q_row = q_out + row_offset;
        for (uint32_t i = 0; i < scale_group_size; ++i) {
            int8_t code = quantize_symmetric_i8_device(transformed[i], scale);
            size_t abs_idx = group_col_start + i;
            size_t out_idx = int8_code_index(abs_idx, code_layout);
            q_row[out_idx] = code;
            if (emit_group_sums) {
                group_sums_out[size_t(row) * sums_per_row + (abs_idx / sum_group_size)] += int32_t(code);
            }
        }
    }
}

extern "C" __global__ void rocm_gated_act_mul_quant(
    const uint8_t* act_operand,
    const uint8_t* value_operand,
    int8_t* q_out,
    float* scales_out,
    int32_t* group_sums_out,
    const int32_t* hadamard_factors,
    uint32_t gated_dim,
    uint32_t batch_dim,
    uint32_t value_offset,
    uint32_t value_row_stride,
    uint32_t scale_group_size,
    uint32_t sum_group_size,
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
    bool custom_alpha = (flags & 4u) != 0;
    bool clip_gate = (flags & 8u) != 0;
    bool clip_value = (flags & 16u) != 0;
    bool emit_group_sums = (flags & 32u) != 0;
    uint32_t code_layout = ((flags & 64u) != 0) ? 1u : 0u;

    uint32_t grp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t groups_per_row = gated_dim / scale_group_size;
    uint32_t total_groups = batch_dim * groups_per_row;
    if (grp_idx >= total_groups || scale_group_size > 256u) return;

    uint32_t batch = grp_idx / groups_per_row;
    uint32_t scale_group_index = grp_idx % groups_per_row;
    uint32_t group_col_start = scale_group_index * scale_group_size;

    float transformed[256];
    for (uint32_t sub = 0; sub < scale_group_size; sub += 32u) {
        float stripe[32];
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t gated = group_col_start + sub + lane;
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
            transformed[sub + lane] = stripe[lane];
        }
    }

    float scale = min_max_symmetric_divisor_device(transformed, scale_group_size);
    scales_out[size_t(batch) * groups_per_row + scale_group_index] = scale;

    uint32_t sums_per_row = emit_group_sums ? (gated_dim / sum_group_size) : 0u;
    uint32_t sum_grp_start = emit_group_sums ? (group_col_start / sum_group_size) : 0u;
    uint32_t sum_grp_count = emit_group_sums ? (scale_group_size / sum_group_size) : 0u;
    for (uint32_t s = 0; s < sum_grp_count; ++s) {
        group_sums_out[size_t(batch) * sums_per_row + sum_grp_start + s] = 0;
    }

    int8_t* q_row = q_out + size_t(batch) * gated_dim;
    for (uint32_t i = 0; i < scale_group_size; ++i) {
        int8_t code = quantize_symmetric_i8_device(transformed[i], scale);
        size_t abs_idx = group_col_start + i;
        size_t out_idx = int8_code_index(abs_idx, code_layout);
        q_row[out_idx] = code;
        if (emit_group_sums) {
            group_sums_out[size_t(batch) * sums_per_row + (abs_idx / sum_group_size)] += int32_t(code);
        }
    }
}

extern "C" __global__ void rocm_short_conv_pack(
    const uint8_t* state_in,
    const uint8_t* in_proj,
    uint8_t* padded,
    uint32_t state_stride,
    uint32_t suffix_len,
    uint32_t in_proj_stride,
    uint32_t model_dim,
    uint32_t dtype
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_rows = state_stride + suffix_len;
    uint32_t total = total_rows * model_dim;
    if (idx >= total) return;

    uint32_t row_idx = idx / model_dim;
    uint32_t channel_idx = idx % model_dim;
    size_t padded_offset = size_t(row_idx) * model_dim + channel_idx;

    if (row_idx < state_stride) {
        size_t state_idx = size_t(channel_idx) * state_stride + row_idx;
        float val = read_elem(state_in, dtype, state_idx);
        write_elem(padded, dtype, padded_offset, val);
    } else {
        uint32_t token = row_idx - state_stride;
        size_t in_proj_idx = size_t(token) * in_proj_stride + channel_idx;
        float pre_gate = read_elem(in_proj, dtype, in_proj_idx);
        float x_in = read_elem(in_proj, dtype, in_proj_idx + 2u * model_dim);
        write_elem(padded, dtype, padded_offset, x_in * pre_gate);
    }
}

extern "C" __global__ void rocm_short_conv_prefill(
    const uint8_t* padded,
    const uint8_t* in_proj,
    const float* w,
    const float* b,
    uint8_t* out,
    uint8_t* state_out,
    uint32_t suffix_len,
    uint32_t kernel_size,
    uint32_t in_proj_stride,
    uint32_t state_stride,
    uint32_t model_dim,
    uint32_t dtype,
    uint32_t has_bias
) {
    uint32_t tap_count = (kernel_size > 0u) ? (kernel_size - 1u) : 0u;
    uint32_t total_rows = suffix_len + tap_count;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = total_rows * model_dim;
    if (idx >= total) return;

    uint32_t token_idx = idx / model_dim;
    uint32_t channel_idx = idx % model_dim;
    const float* w_row = w + size_t(channel_idx) * kernel_size;

    if (token_idx < suffix_len) {
        float acc = (has_bias != 0) ? b[channel_idx] : 0.0f;
        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            size_t padded_offset = size_t(token_idx + tap) * model_dim + channel_idx;
            float sample = read_elem(padded, dtype, padded_offset);
            acc += w_row[tap] * sample;
        }
        size_t in_proj_idx = size_t(token_idx) * in_proj_stride + channel_idx;
        float post_conv_gate = read_elem(in_proj, dtype, in_proj_idx + model_dim);
        write_elem(out, dtype, size_t(token_idx) * model_dim + channel_idx, acc * post_conv_gate);
    } else if (tap_count > 0u) {
        uint32_t tap = token_idx - suffix_len;
        if (tap < tap_count) {
            size_t padded_offset = size_t(suffix_len + tap) * model_dim + channel_idx;
            size_t state_idx = size_t(channel_idx) * state_stride + tap;
            write_elem(state_out, dtype, state_idx, read_elem(padded, dtype, padded_offset));
        }
    }
}

extern "C" __global__ void rocm_short_conv_decode(
    const uint8_t* in_proj,
    const float* w,
    const float* b,
    const uint8_t* state_in,
    uint8_t* out,
    uint8_t* next_state,
    uint32_t suffix_len,
    uint32_t kernel_size,
    uint32_t in_proj_stride,
    uint32_t state_stride,
    uint32_t model_dim,
    uint32_t dtype,
    uint32_t flags
) {
    uint32_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel_idx >= model_dim) return;

    bool has_bias = (flags & 1u) != 0;
    bool state_in_place = (flags & 2u) != 0;
    uint32_t tap_count = (kernel_size > 0u) ? (kernel_size - 1u) : 0u;
    size_t state_offset = size_t(channel_idx) * state_stride;
    const float* w_row = w + size_t(channel_idx) * kernel_size;
    const uint8_t* cur_state = state_in_place ? next_state : state_in;

    for (uint32_t token_idx = 0; token_idx < suffix_len; ++token_idx) {
        size_t in_proj_idx = size_t(token_idx) * in_proj_stride + channel_idx;
        float pre_conv_gate = read_elem(in_proj, dtype, in_proj_idx);
        float post_conv_gate = read_elem(in_proj, dtype, in_proj_idx + model_dim);
        float x_in = read_elem(in_proj, dtype, in_proj_idx + 2u * model_dim);
        float x = x_in * pre_conv_gate;

        float acc = has_bias ? b[channel_idx] : 0.0f;
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            float sample = read_elem(cur_state, dtype, state_offset + tap);
            acc += w_row[tap] * sample;
        }
        acc += w_row[tap_count] * x;

        float gated_output = acc * post_conv_gate;
        write_elem(out, dtype, size_t(token_idx) * model_dim + channel_idx, gated_output);

        if (tap_count > 0u) {
            for (uint32_t tap = 0; tap + 1u < tap_count; ++tap) {
                float s = read_elem(cur_state, dtype, state_offset + tap + 1u);
                write_elem(next_state, dtype, state_offset + tap, s);
            }
            write_elem(next_state, dtype, state_offset + tap_count - 1u, x);
        }
        cur_state = next_state;
    }
}

extern "C" __global__ void rocm_attention_prepare(
    const uint8_t* qkvg,
    uint8_t* queries,
    uint8_t* keys,
    uint8_t* values,
    const float* cosines,
    const float* sines,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t kv_token_offset,
    uint32_t input_row_stride,
    uint32_t batch_dim,
    uint32_t dtype,
    uint32_t flags
) {
    bool has_kv = (flags & 1u) != 0;
    bool has_rope = (flags & 2u) != 0;
    uint32_t total_heads = has_kv ? (num_q_heads + num_kv_heads * 2u) : num_q_heads;
    uint32_t total = batch_dim * total_heads * head_dim;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    uint32_t head_dim_idx = idx % head_dim;
    uint32_t temp = idx / head_dim;
    uint32_t head_idx = temp % total_heads;
    uint32_t batch_idx = temp / total_heads;

    size_t qkvg_head_off = size_t(batch_idx) * input_row_stride + size_t(head_idx) * head_dim;
    bool is_query = !has_kv || (head_idx < num_q_heads);
    bool is_key = has_kv && (head_idx >= num_q_heads) && (head_idx < num_q_heads + num_kv_heads);

    float element = read_elem(qkvg, dtype, qkvg_head_off + head_dim_idx);
    if (has_rope && head_dim_idx < rope_dim && (is_query || is_key)) {
        uint32_t half_rope_dim = rope_dim / 2u;
        uint32_t paired_idx = (head_dim_idx < half_rope_dim) ? (head_dim_idx + half_rope_dim) : (head_dim_idx - half_rope_dim);
        float paired = read_elem(qkvg, dtype, qkvg_head_off + paired_idx);
        float signed_paired = (head_dim_idx < half_rope_dim) ? -paired : paired;
        float cos_val = cosines[size_t(batch_idx) * rope_dim + head_dim_idx];
        float sin_val = sines[size_t(batch_idx) * rope_dim + head_dim_idx];
        element = round_to_dtype(element * cos_val + signed_paired * sin_val, dtype);
    }

    if (is_query) {
        size_t query_offset = (size_t(head_idx) * batch_dim + batch_idx) * head_dim + head_dim_idx;
        write_elem(queries, dtype, query_offset, element);
    } else if (is_key) {
        size_t key_offset = (size_t(kv_token_offset + batch_idx) * num_kv_heads + (head_idx - num_q_heads)) * head_dim + head_dim_idx;
        write_elem(keys, dtype, key_offset, element);
    } else {
        size_t value_offset = (size_t(kv_token_offset + batch_idx) * num_kv_heads + (head_idx - num_q_heads - num_kv_heads)) * head_dim + head_dim_idx;
        write_elem(values, dtype, value_offset, element);
    }
}
