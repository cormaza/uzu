#include <metal_stdlib>

#include "../attention/attention_gate.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/integral_constant.h"
#include "../generated/gemm.h"
#include "../matmul/gemv/common/arguments.h"
#include "../matmul/gemv/common/epilogue.h"
#include "../matmul/gemv/common/full_precision_b_source.h"
#include "../matmul/gemv/common/quant_b_source.h"
#include "../matmul/gemv/common/reduce.h"
#include "../matmul/gemv/common/tile.h"

using namespace uzu;
using namespace uzu::gemm;

#define MOE_ROUTER_NUM_SIMDGROUPS 8
#define MOE_ROUTER_MAX_PER_THREAD 2
#define MOE_GEMV_NUM_SIMDGROUPS 8
#define MOE_GEMV_OUTPUT_ROW_TILE (32)

inline bool is_better_candidate(float v_new, uint id_new, float v_cur, uint id_cur, uint num_experts) {
  if (id_new >= num_experts)
    return false;
  if (id_cur >= num_experts)
    return true;
  return (v_new > v_cur) || (v_new == v_cur && id_new < id_cur);
}

template <typename T, uint MAX_TOP_K>
VARIANTS(T, float)
VARIANTS(MAX_TOP_K, 8)
PUBLIC KERNEL(MoeRouterTopK)(
    const device T* logits,
    const device T* biases OPTIONAL(has_biases),
    device uint* expert_ids,
    device float* expert_weights,
    const constant uint& num_experts,
    const constant uint& top_k,
    const constant uint& batch_size,
    const constant uint& logit_stride,
    const bool has_biases SPECIALIZE,
    threadgroup float shared_best_vals[MOE_ROUTER_NUM_SIMDGROUPS],
    threadgroup uint shared_best_idxs[MOE_ROUTER_NUM_SIMDGROUPS],
    const uint token_idx GROUPS(batch_size),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_ROUTER_NUM_SIMDGROUPS)
) {
  const uint tid = lane + simd_group * METAL_SIMD_SIZE;
  const device T* row = logits + token_idx * logit_stride;

  const uint slots = (num_experts + 255) / 256;
  float my_vals[MOE_ROUTER_MAX_PER_THREAD];
  uint my_ids[MOE_ROUTER_MAX_PER_THREAD];
  for (uint s = 0; s < MOE_ROUTER_MAX_PER_THREAD; ++s) {
    const uint e = tid + s * 256;
    if (e < num_experts) {
      float v = float(row[e]);
      if (has_biases) {
        v += float(biases[e]);
      }
      my_vals[s] = isnan(v) ? -INFINITY : v;
      my_ids[s] = e;
    } else {
      my_vals[s] = -INFINITY;
      my_ids[s] = 0xFFFFFFFFu;
    }
  }

  float my_chosen_val = -INFINITY;
  uint my_chosen_id = 0xFFFFFFFFu;

  for (uint k = 0; k < top_k; ++k) {
    float thread_best_val = my_vals[0];
    uint thread_best_idx = my_ids[0];
    for (uint s = 1; s < MOE_ROUTER_MAX_PER_THREAD; ++s) {
      if (s < slots) {
        const float v = my_vals[s];
        const uint id = my_ids[s];
        if (is_better_candidate(v, id, thread_best_val, thread_best_idx, num_experts)) {
          thread_best_val = v;
          thread_best_idx = id;
        }
      }
    }

    for (ushort offset = METAL_SIMD_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = simd_shuffle_xor(thread_best_val, offset);
      const uint other_idx = simd_shuffle_xor(thread_best_idx, offset);
      if (is_better_candidate(other_val, other_idx, thread_best_val, thread_best_idx, num_experts)) {
        thread_best_val = other_val;
        thread_best_idx = other_idx;
      }
    }

    if (lane == 0) {
      shared_best_vals[simd_group] = thread_best_val;
      shared_best_idxs[simd_group] = thread_best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
      float v = (lane < MOE_ROUTER_NUM_SIMDGROUPS) ? shared_best_vals[lane] : -INFINITY;
      uint id = (lane < MOE_ROUTER_NUM_SIMDGROUPS) ? shared_best_idxs[lane] : 0xFFFFFFFFu;
      for (ushort offset = 4; offset > 0; offset >>= 1) {
        const float other_val = simd_shuffle_xor(v, offset);
        const uint other_id = simd_shuffle_xor(id, offset);
        if (is_better_candidate(other_val, other_id, v, id, num_experts)) {
          v = other_val;
          id = other_id;
        }
      }
      if (lane == k) {
        my_chosen_val = (id < num_experts) ? v : -INFINITY;
        my_chosen_id = (id < num_experts) ? id : min(k, num_experts - 1);
      }
      if (lane == 0) {
        shared_best_idxs[0] = (id < num_experts) ? id : min(k, num_experts - 1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint winner_id = shared_best_idxs[0];
    if (winner_id % 256 == tid) {
      const uint s = winner_id / 256;
      if (s < MOE_ROUTER_MAX_PER_THREAD) {
        my_vals[s] = -INFINITY;
        my_ids[s] = 0xFFFFFFFFu;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_group == 0) {
    const float max_chosen = simd_shuffle(my_chosen_val, 0);

    const float e_val =
        (lane < top_k) ? ((max_chosen == -INFINITY) ? (1.0f / float(top_k)) : fast::exp(my_chosen_val - max_chosen))
                       : 0.0f;

    float sum_exp = 0.0f;
    for (uint i = 0; i < top_k; ++i) {
      sum_exp += simd_shuffle(e_val, i);
    }

    if (lane < top_k) {
      const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
      device uint* out_ids = expert_ids + token_idx * top_k;
      device float* out_weights = expert_weights + token_idx * top_k;
      out_ids[lane] = my_chosen_id;
      out_weights[lane] = e_val * inv_sum;
    }
  }
}

template <
    typename T,
    uint GROUP_SIZE,
    uint BITS>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 0, 32, 64)
VARIANTS(BITS, 0, 4, 8)
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
PUBLIC KERNEL(MoeGemv)(
    const device uint8_t* b,
    const device T* scales OPTIONAL(b_prologue != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points OPTIONAL(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* biases OPTIONAL(b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device uint8_t* shared_b OPTIONAL(has_shared),
    const device T* shared_scales OPTIONAL(has_shared && b_prologue != GemmBPrologueKind::FullPrecision),
    const device uint8_t* shared_zero_points OPTIONAL(has_shared && b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* shared_biases OPTIONAL(has_shared && b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device T* a,
    const device uint* expert_ids,
    device T* d,
    const device T* output_bias OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* shared_output_bias OPTIONAL(has_shared && output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* shared_hadamard_factors OPTIONAL(has_shared && output_transform.contains(GemmDTransform::RHT)),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& num_routes,
    const constant uint& active_experts_per_token,
    const constant uint& group_count_x,
    const constant uint& b_expert_stride_bytes,
    const constant uint& scale_expert_stride,
    const constant uint& zero_point_expert_stride,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant uint& zero_point_output_stride,
    const constant uint& zero_point_group_stride,
    const constant uint& num_routed_routes,
    const GemmBPrologueKind b_prologue SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const bool input_aligned SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool full_tile SPECIALIZE,
    const bool a_per_token SPECIALIZE,
    const bool has_shared SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint route_idx GROUPS(num_routes),
    const uint output_tile_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_GEMV_NUM_SIMDGROUPS)
) {
  const bool is_shared = has_shared && (route_idx >= num_routed_routes);
  const uint expert_id = is_shared ? 0u : expert_ids[route_idx];
  const uint a_row_idx =
      a_per_token ? (is_shared ? (route_idx - num_routed_routes) : (route_idx / active_experts_per_token)) : route_idx;

  const device uint32_t* expert_b =
      is_shared ? reinterpret_cast<const device uint32_t*>(shared_b)
                : reinterpret_cast<const device uint32_t*>(b + ulong(expert_id) * b_expert_stride_bytes);
  const device T* expert_scales = (b_prologue != GemmBPrologueKind::FullPrecision)
                                      ? (is_shared ? shared_scales : (scales + ulong(expert_id) * scale_expert_stride))
                                      : nullptr;
  const device uint8_t* expert_zero_points =
      (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)
          ? (is_shared ? shared_zero_points : (zero_points + ulong(expert_id) * zero_point_expert_stride))
          : nullptr;
  const device T* expert_biases = (b_prologue == GemmBPrologueKind::ScaleBiasDequant)
                                      ? (is_shared ? shared_biases : (biases + ulong(expert_id) * scale_expert_stride))
                                      : nullptr;
  const device T* route_a = a + ulong(a_row_idx) * in_vec_size;
  device T* route_d = d + ulong(route_idx) * out_vec_size;
  const device T* expert_output_bias =
      output_transform.contains(GemmDTransform::BIAS)
          ? (is_shared ? shared_output_bias : (output_bias + ulong(expert_id) * out_vec_size))
          : nullptr;
  const device int32_t* expert_hadamard =
      output_transform.contains(GemmDTransform::RHT)
          ? (is_shared ? shared_hadamard_factors : (hadamard_factors + ulong(expert_id) * out_vec_size))
          : nullptr;

  using Ops = GemvOperands<T, T, T>;
  const Ops ops = {
      expert_b,
      expert_scales,
      expert_zero_points,
      expert_biases,
      route_a,
      route_d,
      expert_output_bias,
      expert_hadamard,
      nullptr
  };
  const GemvParams params = {
      in_vec_size,
      out_vec_size,
      scale_output_stride,
      scale_group_stride,
      zero_point_output_stride,
      zero_point_group_stride,
      /* batch_size = */ 1u,
      /* ab_scale = */ 1.0f,
      /* soft_cap = */ 0.0f,
      output_transform,
      /* gathered = */ false,
      signed_codes,
      GemmBLayout::RowMajor
  };

  constexpr uint GROUP_LANES =
      (BITS == 0) ? 1 : ((BITS == 4) ? (GROUP_SIZE == 32 ? 2 : 4) : (GROUP_SIZE == 32 ? 4 : 8));
  using Tile = GemvTile<
      /* INPUT_ROW_TILE = */ 1,
      /* OUTPUT_ROW_TILE = */ 32,
      /* REDUCTION_LANES = */ 32,
      GROUP_LANES,
      /* NUM_SIMDGROUPS = */ MOE_GEMV_NUM_SIMDGROUPS,
      /* K_SPLIT = */ 1>;

  dispatch_bool(full_tile, [&](auto full_tile_constant) {
    constexpr bool FullTile = decltype(full_tile_constant)::value;
    dispatch_bool(input_aligned, [&](auto input_aligned_constant) {
      constexpr bool InputAligned = decltype(input_aligned_constant)::value;
      const OutputTile<Tile, FullTile> tile = OutputTile<Tile, FullTile>::make(
          output_tile_idx,
          /* input_tile_idx = */ 0u,
          simd_group,
          simd_lane,
          out_vec_size
      );
      thread float result[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

      if constexpr (BITS == 0) {
        FullPrecisionBSource<Tile, T, T, T, InputAligned, FullTile>::accumulate(result, ops, params, tile);
      } else if constexpr (BITS == 4) {
        if (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant) {
          QuantBSource<
              Tile,
              T,
              T,
              T,
              GemmBPrologueKind::ScaleZeroPointDequant,
              GROUP_SIZE,
              BITS,
              InputAligned,
              FullTile>::accumulate(result, ops, params, tile);
        } else {
          QuantBSource<Tile, T, T, T, GemmBPrologueKind::ScaleBiasDequant, GROUP_SIZE, BITS, InputAligned, FullTile>::
              accumulate(result, ops, params, tile);
        }
      } else {
        QuantBSource<
            Tile,
            T,
            T,
            T,
            GemmBPrologueKind::ScaleSymmetricDequant,
            GROUP_SIZE,
            BITS,
            InputAligned,
            FullTile>::accumulate(result, ops, params, tile);
      }

      Reduce<Tile, FullTile>::run(result, shared_results, tile);
      Epilogue<Tile, T, T, T, FullTile>::store(result, ops, params, tile, shared_results);
    });
  });
}

template <
    typename T,
    uint GROUP_SIZE,
    uint BITS>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 0, 32, 64)
VARIANTS(BITS, 0, 4, 8)
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
PUBLIC KERNEL(MoeDownCombine)(
    const device uint8_t* b,
    const device T* scales OPTIONAL(b_prologue != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points OPTIONAL(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* biases OPTIONAL(b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device uint8_t* shared_b OPTIONAL(has_shared),
    const device T* shared_scales OPTIONAL(has_shared && b_prologue != GemmBPrologueKind::FullPrecision),
    const device uint8_t* shared_zero_points OPTIONAL(has_shared && b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* shared_biases OPTIONAL(has_shared && b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device T* a,
    const device uint* expert_ids,
    const device float* expert_weights,
    const device float* shared_gate OPTIONAL(has_gate),
    device T* output,
    const device T* output_bias OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* shared_output_bias OPTIONAL(has_shared && output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* shared_hadamard_factors OPTIONAL(has_shared && output_transform.contains(GemmDTransform::RHT)),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& active_experts_per_token,
    const constant uint& group_count_x,
    const constant uint& b_expert_stride_bytes,
    const constant uint& scale_expert_stride,
    const constant uint& zero_point_expert_stride,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant uint& zero_point_output_stride,
    const constant uint& zero_point_group_stride,
    const constant uint& batch_size,
    const constant uint& gate_stride,
    const GemmBPrologueKind b_prologue SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const bool input_aligned SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool full_tile SPECIALIZE,
    const bool has_shared SPECIALIZE,
    const bool has_gate SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint output_tile_idx GROUPS(group_count_x),
    const uint token_idx GROUPS(batch_size),
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_GEMV_NUM_SIMDGROUPS)
) {
  constexpr uint GROUP_LANES =
      (BITS == 0) ? 1 : ((BITS == 4) ? (GROUP_SIZE == 32 ? 2 : 4) : (GROUP_SIZE == 32 ? 4 : 8));
  using Tile = GemvTile<
      /* INPUT_ROW_TILE = */ 1,
      /* OUTPUT_ROW_TILE = */ MOE_GEMV_OUTPUT_ROW_TILE,
      /* REDUCTION_LANES = */ 32,
      GROUP_LANES,
      /* NUM_SIMDGROUPS = */ MOE_GEMV_NUM_SIMDGROUPS,
      /* K_SPLIT = */ 1>;

  dispatch_bool(full_tile, [&](auto full_tile_constant) {
    constexpr bool FullTile = decltype(full_tile_constant)::value;
    dispatch_bool(input_aligned, [&](auto input_aligned_constant) {
      constexpr bool InputAligned = decltype(input_aligned_constant)::value;
      const OutputTile<Tile, FullTile> tile = OutputTile<Tile, FullTile>::make(
          output_tile_idx,
          /* input_tile_idx = */ 0u,
          simd_group,
          simd_lane,
          out_vec_size
      );
      thread float combined[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

      const uint base_routed_idx = token_idx * active_experts_per_token;
      const device float* weights_row = expert_weights + ulong(base_routed_idx);

      for (uint k = 0; k < active_experts_per_token; ++k) {
        const float weight = weights_row[k];
        const uint route_idx = base_routed_idx + k;
        const uint expert_id = expert_ids[route_idx];

        const device uint32_t* expert_b =
            reinterpret_cast<const device uint32_t*>(b + ulong(expert_id) * b_expert_stride_bytes);
        const device T* expert_scales = (b_prologue != GemmBPrologueKind::FullPrecision)
                                            ? (scales + ulong(expert_id) * scale_expert_stride)
                                            : nullptr;
        const device uint8_t* expert_zero_points = (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)
                                                       ? (zero_points + ulong(expert_id) * zero_point_expert_stride)
                                                       : nullptr;
        const device T* expert_biases = (b_prologue == GemmBPrologueKind::ScaleBiasDequant)
                                            ? (biases + ulong(expert_id) * scale_expert_stride)
                                            : nullptr;
        const device T* route_a = a + ulong(route_idx) * in_vec_size;

        using Ops = GemvOperands<T, T, T>;
        const Ops ops =
            {expert_b, expert_scales, expert_zero_points, expert_biases, route_a, nullptr, nullptr, nullptr, nullptr};
        const GemvParams params = {
            in_vec_size,
            out_vec_size,
            scale_output_stride,
            scale_group_stride,
            zero_point_output_stride,
            zero_point_group_stride,
            /* batch_size = */ 1u,
            /* ab_scale = */ 1.0f,
            /* soft_cap = */ 0.0f,
            GemmDTransform(),
            /* gathered = */ false,
            signed_codes,
            GemmBLayout::RowMajor
        };

        thread float route_res[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

        if constexpr (BITS == 0) {
          FullPrecisionBSource<Tile, T, T, T, InputAligned, FullTile>::accumulate(route_res, ops, params, tile);
        } else if constexpr (BITS == 4) {
          if (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant) {
            QuantBSource<
                Tile,
                T,
                T,
                T,
                GemmBPrologueKind::ScaleZeroPointDequant,
                GROUP_SIZE,
                BITS,
                InputAligned,
                FullTile>::accumulate(route_res, ops, params, tile);
          } else {
            QuantBSource<Tile, T, T, T, GemmBPrologueKind::ScaleBiasDequant, GROUP_SIZE, BITS, InputAligned, FullTile>::
                accumulate(route_res, ops, params, tile);
          }
        } else {
          QuantBSource<
              Tile,
              T,
              T,
              T,
              GemmBPrologueKind::ScaleSymmetricDequant,
              GROUP_SIZE,
              BITS,
              InputAligned,
              FullTile>::accumulate(route_res, ops, params, tile);
        }

        Reduce<Tile, FullTile>::run(route_res, nullptr, tile);

        const device T* expert_output_bias =
            output_transform.contains(GemmDTransform::BIAS) ? (output_bias + ulong(expert_id) * out_vec_size) : nullptr;

        Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
          constexpr uint R = decltype(output_index)::value;
          const uint global_row = tile.row0 + R;
          if (FullTile || global_row < out_vec_size) {
            float val = route_res[0][R];
            if (output_transform.contains(GemmDTransform::BIAS)) {
              val += static_cast<float>(expert_output_bias[global_row]);
            }
            combined[0][R] = fma(weight, val, combined[0][R]);
          }
        });
      }

      if (has_shared) {
        const uint num_routed_routes = batch_size * active_experts_per_token;
        const uint shared_route_idx = num_routed_routes + token_idx;
        const device uint32_t* expert_b = reinterpret_cast<const device uint32_t*>(shared_b);
        const device T* expert_scales = (b_prologue != GemmBPrologueKind::FullPrecision) ? shared_scales : nullptr;
        const device uint8_t* expert_zero_points =
            (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant) ? shared_zero_points : nullptr;
        const device T* expert_biases = (b_prologue == GemmBPrologueKind::ScaleBiasDequant) ? shared_biases : nullptr;
        const device T* route_a = a + ulong(shared_route_idx) * in_vec_size;

        using Ops = GemvOperands<T, T, T>;
        const Ops ops =
            {expert_b, expert_scales, expert_zero_points, expert_biases, route_a, nullptr, nullptr, nullptr, nullptr};
        const GemvParams params = {
            in_vec_size,
            out_vec_size,
            scale_output_stride,
            scale_group_stride,
            zero_point_output_stride,
            zero_point_group_stride,
            /* batch_size = */ 1u,
            /* ab_scale = */ 1.0f,
            /* soft_cap = */ 0.0f,
            GemmDTransform(),
            /* gathered = */ false,
            signed_codes,
            GemmBLayout::RowMajor
        };

        thread float shared_res[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

        if constexpr (BITS == 0) {
          FullPrecisionBSource<Tile, T, T, T, InputAligned, FullTile>::accumulate(shared_res, ops, params, tile);
        } else if constexpr (BITS == 4) {
          if (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant) {
            QuantBSource<
                Tile,
                T,
                T,
                T,
                GemmBPrologueKind::ScaleZeroPointDequant,
                GROUP_SIZE,
                BITS,
                InputAligned,
                FullTile>::accumulate(shared_res, ops, params, tile);
          } else {
            QuantBSource<Tile, T, T, T, GemmBPrologueKind::ScaleBiasDequant, GROUP_SIZE, BITS, InputAligned, FullTile>::
                accumulate(shared_res, ops, params, tile);
          }
        } else {
          QuantBSource<
              Tile,
              T,
              T,
              T,
              GemmBPrologueKind::ScaleSymmetricDequant,
              GROUP_SIZE,
              BITS,
              InputAligned,
              FullTile>::accumulate(shared_res, ops, params, tile);
        }

        Reduce<Tile, FullTile>::run(shared_res, nullptr, tile);

        const device T* expert_output_bias =
            output_transform.contains(GemmDTransform::BIAS) ? (shared_output_bias + ulong(0) * out_vec_size) : nullptr;

        float shared_gate_val = 0.0f;
        if (has_gate) {
          shared_gate_val = shared_gate[ulong(token_idx) * gate_stride];
        }

        Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
          constexpr uint R = decltype(output_index)::value;
          const uint global_row = tile.row0 + R;
          if (FullTile || global_row < out_vec_size) {
            float val = shared_res[0][R];
            if (output_transform.contains(GemmDTransform::BIAS)) {
              val += static_cast<float>(expert_output_bias[global_row]);
            }
            if (has_gate) {
              val = attention_apply_gate(val, shared_gate_val);
            }
            combined[0][R] += val;
          }
        });
      }

      const device int32_t* token_hadamard =
          output_transform.contains(GemmDTransform::RHT) ? hadamard_factors : nullptr;
      using Ops = GemvOperands<T, T, T>;
      const Ops ops_out = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          output + ulong(token_idx) * out_vec_size,
          nullptr,
          token_hadamard,
          nullptr
      };
      const GemvParams params_out = {
          in_vec_size,
          out_vec_size,
          scale_output_stride,
          scale_group_stride,
          zero_point_output_stride,
          zero_point_group_stride,
          /* batch_size = */ 1u,
          /* ab_scale = */ 1.0f,
          /* soft_cap = */ 0.0f,
          GemmDTransform(output_transform.bits() & ~GemmDTransform::BIAS),
          /* gathered = */ false,
          signed_codes,
          GemmBLayout::RowMajor
      };

      Epilogue<Tile, T, T, T, FullTile>::store(combined, ops_out, params_out, tile, shared_results);
    });
  });
}

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeCombine)(
    const device T* expert_outputs,
    const device float* expert_weights,
    const device T* shared_output OPTIONAL(has_shared),
    const device float* shared_gate OPTIONAL(has_gate),
    device T* output,
    const constant uint& model_dim,
    const constant uint& top_k,
    const constant uint& batch_size,
    const constant uint& gate_stride,
    const bool has_shared SPECIALIZE,
    const bool has_gate SPECIALIZE,
    const uint col AXIS(model_dim, 256),
    const uint token_idx AXIS(batch_size, 1)
) {
  float acc = 0.0f;
  const device float* weights_row = expert_weights + token_idx * top_k;
  const uint base_route = token_idx * top_k;
  for (uint k = 0; k < top_k; ++k) {
    const float w = weights_row[k];
    const float val = float(expert_outputs[(base_route + k) * model_dim + col]);
    acc = fma(w, val, acc);
  }
  if (has_shared) {
    float s_val = float(shared_output[token_idx * model_dim + col]);
    if (has_gate) {
      s_val = attention_apply_gate(s_val, float(shared_gate[token_idx * gate_stride]));
    }
    acc += s_val;
  }
  output[token_idx * model_dim + col] = T(acc);
}
