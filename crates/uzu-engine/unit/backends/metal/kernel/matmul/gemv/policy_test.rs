use metal::MTLGPUFamily;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::{
        common::{
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::matmul::{MatmulShape, QuantParamsLayout},
        },
        metal::{
            gpu_types::gemm::{GemmBLayout, GemmTiling},
            kernel::matmul::{MatmulDispatch, MatmulMetalKernel},
        },
    },
    data_type::DataType,
};

fn shape(
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: if bits == 8 {
            GemmBPrologueKind::ScaleSymmetricDequant
        } else {
            GemmBPrologueKind::ScaleZeroPointDequant
        },
        b_is_trellis: false,
        b_bits: Some(bits),
        b_group_size: Some(group),
        signed_codes: bits == 8,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::GroupOutput),
        d_transform: GemmDTransform::empty(),
    }
}

#[uzu_test]
fn w4_w8_lane_policy() {
    let qtile = |sg, rows| GemvTile::quantized(sg, rows, 1, 32, 2);
    for (cores, family, m, n, k, bits, expected) in [
        (30, MTLGPUFamily::Apple9, 1, 256, 1536, 4, qtile(2, 1)),
        (29, MTLGPUFamily::Apple9, 1, 256, 1536, 4, qtile(4, 2)),
        (30, MTLGPUFamily::Apple9, 1, 262144, 1536, 4, qtile(8, 4)),
        (29, MTLGPUFamily::Apple9, 1, 1536, 256, 4, qtile(4, 4)),
        (29, MTLGPUFamily::Apple10, 1, 1536, 256, 4, qtile(8, 4)),
        (29, MTLGPUFamily::Apple8, 1, 2048, 1536, 4, qtile(8, 2)),
        (29, MTLGPUFamily::Apple7, 1, 256, 1536, 4, qtile(8, 2)),
        (29, MTLGPUFamily::Apple7, 1, 1536, 256, 4, qtile(4, 8)),
        (30, MTLGPUFamily::Apple9, 2, 2048, 1536, 4, qtile(8, 4)),
        (30, MTLGPUFamily::Apple9, 4, 5120, 5120, 4, qtile(8, 4)),
        (30, MTLGPUFamily::Apple9, 1, 2048, 1536, 8, GemvTile::quantized(8, 4, 1, 32, 4)),
    ] {
        assert_eq!(quantized_tile(cores, family, bits, 32, m, n, k, true), Some(expected));
    }
    assert_eq!(
        quantized_tile(30, MTLGPUFamily::Apple10, 4, 64, 4, 8192, 4100, true),
        Some(GemvTile::quantized(8, 4, 1, 32, 4))
    );
    assert_eq!(quantized_tile(30, MTLGPUFamily::Apple10, 4, 32, 9, 8192, 4096, true), None);
    assert_eq!(gathered_tile(4, 32, 1, 8192), Some(qtile(8, 4)));
}

#[uzu_test]
fn qmv_route_boundaries() {
    for (supports_mxu, m, n, k, bits, group, gemm_route) in [
        (false, 2, 72, 1024, 4, 64, false),
        (false, 4, 72, 1024, 4, 64, true),
        (false, 4, 72, 256, 8, 64, true),
        (true, 4, 72, 1024, 4, 64, false),
        (true, 5, 72, 1024, 4, 32, true),
        (true, 5, 72, 256, 8, 64, true),
        (true, 5, 72, 768, 8, 64, true),
        (true, 4, 72, 1024, 8, 64, false),
        (true, 8, 72, 1024, 8, 64, true),
        (true, 3, 72, 1024, 4, 32, false),
    ] {
        let family = if supports_mxu {
            MTLGPUFamily::Apple10
        } else {
            MTLGPUFamily::Apple9
        };
        let problem = shape(m, n, k, bits, group);
        let actual = MatmulMetalKernel::choose_dispatch(
            &problem,
            GemmBLayout::RowMajor,
            40,
            family,
            supports_mxu,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
        );
        if gemm_route {
            let expected_tiling = if bits == 4 {
                GemmTiling::Tile8x32x64_Simdgroups1x2x4
            } else {
                GemmTiling::Tile8x32x128_Simdgroups1x2x4
            };
            assert!(
                matches!(actual, MatmulDispatch::Gemm(plan) if plan.tiling == expected_tiling),
                "M={m} W{bits} G{group}, MXU={supports_mxu}"
            );
        } else {
            assert!(matches!(actual, MatmulDispatch::Gemv(_)), "M={m} W{bits} G{group}, MXU={supports_mxu}");
        }
    }

    let mut soft_cap = shape(8, 72, 1024, 8, 64);
    soft_cap.d_transform = GemmDTransform::SOFT_CAP;
    let fallback = MatmulMetalKernel::choose_dispatch(
        &soft_cap,
        GemmBLayout::RowMajor,
        40,
        MTLGPUFamily::Apple10,
        true,
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
    );
    assert!(
        matches!(fallback, MatmulDispatch::Gemm(plan) if plan.tiling == GemmTiling::Tile8x32x128_Simdgroups1x2x4),
        "W8 G64 soft-cap uses the register-quant GEMM route"
    );
}
