use std::ffi::c_void;

use crate::{
    backends::{
        common::{
            Backend, BufferCpuAccessible, BufferMut, BufferRef, CommandBufferEncoding, Kernels,
            gpu_types::quantization::QuantizationMode,
            kernel::{
                ActivationTransform, TensorAddBiasKernel,
                matmul::{Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulKernel, TRELLIS_CODES_TAIL_BYTES},
            },
        },
        rocm::{
            Rocm,
            buffer::RocmBufferExt,
            command_buffer::RocmCommandBufferEncoding,
            context::RocmContext,
            kernel::{hip_ops::dtype_code, matmul::MatmulRocmError},
        },
    },
    data_type::DataType,
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct MatmulRocmKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    output_rht: ActivationTransform<Rocm>,
    bias_add: <<Rocm as Backend>::Kernels as Kernels>::TensorAddBiasKernel,
}

impl MatmulKernel for MatmulRocmKernel {
    type Backend = Rocm;
    type Error = MatmulRocmError;
    type WeightsLayout = ();

    fn new(
        context: &RocmContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Self::Error> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
                return Err(MatmulRocmError::UnsupportedDataType(data_type));
            }
        }
        let output_rht = ActivationTransform::output_rht(context, output_data_type, None, true)?;
        let bias_add = <<Rocm as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(
            context,
            output_data_type,
            weights_data_type,
            true,
        )?;
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            output_rht,
            bias_add,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Rocm,
            impl BufferRef<Backend = Rocm>,
            impl BufferRef<Backend = Rocm>,
            impl BufferMut<Backend = Rocm>,
            impl BufferRef<Backend = Rocm>,
        >,
        command_buffer: &mut RocmCommandBufferEncoding,
    ) -> Result<(), Self::Error> {
        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            output,
            m,
            n,
            k,
            gather_indices,
            ..
        } = arguments;

        if let (
            MatmulA::Trellis {
                values: activations,
                column_group_sums,
                scales: activation_scales,
            },
            MatmulB::Trellis {
                codes,
                row_scales,
                codebook,
                format,
            },
        ) = (&a, &b)
        {
            let output_stride = output.row_stride.unwrap_or(n);
            let row_bytes = format.code_row_bytes(k);
            if self.output_data_type != DataType::BF16
                || !output.ops.mask().is_empty()
                || !b_transpose
                || b_leading_dimension.is_some()
                || output_stride < n
                || codes.size() < (n * row_bytes) as usize + TRELLIS_CODES_TAIL_BYTES
            {
                return Err(MatmulRocmError::UnsupportedLayout {
                    path: "RocmMatmulTrellis",
                });
            }

            let act_ptr = buffer_ref_dev_ptr(activations);
            let sums_ptr = buffer_ref_dev_ptr(column_group_sums);
            let ascales_ptr = buffer_ref_dev_ptr(activation_scales);
            let codes_ptr = buffer_ref_dev_ptr(codes);
            let rscales_ptr = buffer_ref_dev_ptr(row_scales);
            let cbook_ptr = buffer_ref_dev_ptr(codebook);
            let (out_buf, out_range) = output.values.parts();
            let out_ptr =
                SendPtrMut(unsafe { out_buf.device_address().as_ptr().cast::<u8>().byte_add(out_range.start) });

            let vector_width = format.vector_width;
            let transition_bits = format.transition_bits;
            let restart_columns = format.restart_columns.unwrap_or(0);
            let func = command_buffer.context().kernels.matmul_trellis;

            command_buffer.push_gpu_command(move |stream| unsafe {
                let mut p_act = act_ptr.as_ptr() as *const c_void;
                let mut p_sums = sums_ptr.as_ptr() as *const c_void;
                let mut p_asc = ascales_ptr.as_ptr() as *const c_void;
                let mut p_codes = codes_ptr.as_ptr() as *const c_void;
                let mut p_rsc = rscales_ptr.as_ptr() as *const c_void;
                let mut p_cb = cbook_ptr.as_ptr() as *const c_void;
                let mut p_out = out_ptr.as_ptr() as *mut c_void;
                let mut p_m = m;
                let mut p_n = n;
                let mut p_k = k;
                let mut p_stride = output_stride;
                let mut p_rbytes = row_bytes;
                let mut p_vw = vector_width;
                let mut p_tb = transition_bits;
                let mut p_rc = restart_columns;
                let mut params: [*mut c_void; 15] = [
                    (&mut p_act as *mut *const c_void).cast(),
                    (&mut p_sums as *mut *const c_void).cast(),
                    (&mut p_asc as *mut *const c_void).cast(),
                    (&mut p_codes as *mut *const c_void).cast(),
                    (&mut p_rsc as *mut *const c_void).cast(),
                    (&mut p_cb as *mut *const c_void).cast(),
                    (&mut p_out as *mut *mut c_void).cast(),
                    (&mut p_m as *mut u32).cast(),
                    (&mut p_n as *mut u32).cast(),
                    (&mut p_k as *mut u32).cast(),
                    (&mut p_stride as *mut u32).cast(),
                    (&mut p_rbytes as *mut u32).cast(),
                    (&mut p_vw as *mut u32).cast(),
                    (&mut p_tb as *mut u32).cast(),
                    (&mut p_rc as *mut u32).cast(),
                ];
                let total = m.saturating_mul(n);
                let block = 128u32;
                let grid = total.div_ceil(block);
                stream
                    .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .expect("rocm_matmul_trellis failed");
            });
            return Ok(());
        }

        let (mut d, ops) = output.into_contiguous(n, "RocmMatmul")?;
        let output_scale = ops.ab_scale;
        let bias_buffer = ops.bias;
        let post_rht = ops.rht_factors;
        let soft_cap = ops.soft_cap;

        let input_dtype = dtype_code(self.input_data_type).unwrap();
        let weights_dtype = dtype_code(self.weights_data_type).unwrap();
        let output_dtype = dtype_code(self.output_data_type).unwrap();

        let (a_ptr, a_scales_ptr, a_mode, a_group_size, a_code_layout) = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => {
                let (values, range) = values.parts();
                let byte_offset = range.start + offset * self.input_data_type.size_in_bytes();
                let ptr = SendPtr(values.device_address().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(byte_offset));
                (ptr, SendPtr(std::ptr::null()), 0u32, 0u32, 0u32)
            },
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums: _,
                scale_group_size: a_group_size,
                code_layout,
            } => {
                let compatible = matches!(a_group_size, 32 | 64 | 128)
                    && k.is_multiple_of(a_group_size)
                    && matches!(b.group_size(), Some(32 | 64 | 128))
                    && b.quantized()
                        .is_some_and(|quantized| matches!(quantized.mode, QuantizationMode::U4 | QuantizationMode::U8));
                if !compatible {
                    return Err(MatmulRocmError::IncompatibleA {
                        path: "RocmMatmul",
                        reason: "symmetric int8 activations require a supported 32/64/128 activation and weight group",
                    });
                }
                let val_ptr = buffer_ref_dev_ptr(&values);
                let sc_ptr = buffer_ref_dev_ptr(&scales);
                let layout_code = match code_layout {
                    Int8CodeLayout::Sequential => 0u32,
                    Int8CodeLayout::GroupedByNibble => 1u32,
                };
                (val_ptr, sc_ptr, 1u32, a_group_size, layout_code)
            },
            MatmulA::Trellis {
                ..
            } => {
                return Err(MatmulRocmError::UnsupportedLayout {
                    path: "RocmMatmul",
                });
            },
        };

        let bias_after_rht = post_rht.is_some();
        let inline_bias_ptr = if !bias_after_rht {
            bias_buffer
                .map(|bias| SendPtr(bias.device_address().as_ptr().cast::<u8>().cast_const()))
                .unwrap_or(SendPtr(std::ptr::null()))
        } else {
            SendPtr(std::ptr::null())
        };
        let gather_ptr = gather_indices
            .as_ref()
            .map(buffer_ref_dev_ptr)
            .unwrap_or(SendPtr(std::ptr::null()));
        let (d_buffer, d_range) = d.reborrow().parts();
        let d_ptr = SendPtrMut(d_buffer.device_address().as_ptr().cast::<u8>().wrapping_byte_add(d_range.start));
        let (soft_cap_val, has_soft_cap) = match soft_cap {
            Some(cap) => (cap, 1u32),
            None => (0.0f32, 0u32),
        };

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let b_ptr = buffer_ref_dev_ptr(&weights);
                let b_ld = b_leading_dimension.unwrap_or(if b_transpose {
                    k
                } else {
                    n
                });
                let b_trans = u32::from(b_transpose);
                let func = command_buffer.context().kernels.matmul_fp;

                command_buffer.push_gpu_command(move |stream| unsafe {
                    let mut p_a = a_ptr.as_ptr() as *const c_void;
                    let mut p_asc = a_scales_ptr.as_ptr() as *const c_void;
                    let mut p_b = b_ptr.as_ptr() as *const c_void;
                    let mut p_bias = inline_bias_ptr.as_ptr() as *const c_void;
                    let mut p_gath = gather_ptr.as_ptr() as *const c_void;
                    let mut p_d = d_ptr.as_ptr() as *mut c_void;
                    let mut p_m = m;
                    let mut p_n = n;
                    let mut p_k = k;
                    let mut p_bld = b_ld;
                    let mut p_btr = b_trans;
                    let mut p_idt = input_dtype;
                    let mut p_wdt = weights_dtype;
                    let mut p_odt = output_dtype;
                    let mut p_am = a_mode;
                    let mut p_ags = a_group_size;
                    let mut p_acl = a_code_layout;
                    let mut p_osc = output_scale;
                    let mut p_sc = soft_cap_val;
                    let mut p_hsc = has_soft_cap;
                    let mut params: [*mut c_void; 20] = [
                        (&mut p_a as *mut *const c_void).cast(),
                        (&mut p_asc as *mut *const c_void).cast(),
                        (&mut p_b as *mut *const c_void).cast(),
                        (&mut p_bias as *mut *const c_void).cast(),
                        (&mut p_gath as *mut *const c_void).cast(),
                        (&mut p_d as *mut *mut c_void).cast(),
                        (&mut p_m as *mut u32).cast(),
                        (&mut p_n as *mut u32).cast(),
                        (&mut p_k as *mut u32).cast(),
                        (&mut p_bld as *mut u32).cast(),
                        (&mut p_btr as *mut u32).cast(),
                        (&mut p_idt as *mut u32).cast(),
                        (&mut p_wdt as *mut u32).cast(),
                        (&mut p_odt as *mut u32).cast(),
                        (&mut p_am as *mut u32).cast(),
                        (&mut p_ags as *mut u32).cast(),
                        (&mut p_acl as *mut u32).cast(),
                        (&mut p_osc as *mut f32).cast(),
                        (&mut p_sc as *mut f32).cast(),
                        (&mut p_hsc as *mut u32).cast(),
                    ];
                    let total = m.saturating_mul(n);
                    let block = 128u32;
                    let grid = total.div_ceil(block);
                    stream.launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params).expect("rocm_matmul_fp failed");
                });
            },
            MatmulB::Quantized(quantized) => {
                let b_bits = quantized.bits();
                let b_group_size = quantized.group_size;
                let b_signed_codes = u32::from(quantized.signed_codes);
                let scale_strides = quantized.params.scale_strides();
                let zp_strides = quantized.zero_point_strides();
                let b_weights_ptr = buffer_ref_dev_ptr(&quantized.codes);
                let b_scales_ptr = buffer_ref_dev_ptr(&quantized.scales);
                let b_zp_ptr = quantized
                    .zero_points()
                    .as_ref()
                    .map(buffer_ref_dev_ptr)
                    .unwrap_or(SendPtr(std::ptr::null()));
                let b_biases_ptr = quantized
                    .biases()
                    .as_ref()
                    .map(buffer_ref_dev_ptr)
                    .unwrap_or(SendPtr(std::ptr::null()));
                let func = command_buffer.context().kernels.matmul_quant;

                command_buffer.push_gpu_command(move |stream| unsafe {
                    let mut p_a = a_ptr.as_ptr() as *const c_void;
                    let mut p_asc = a_scales_ptr.as_ptr() as *const c_void;
                    let mut p_bw = b_weights_ptr.as_ptr() as *const c_void;
                    let mut p_bs = b_scales_ptr.as_ptr() as *const c_void;
                    let mut p_bzp = b_zp_ptr.as_ptr() as *const c_void;
                    let mut p_bbi = b_biases_ptr.as_ptr() as *const c_void;
                    let mut p_obias = inline_bias_ptr.as_ptr() as *const c_void;
                    let mut p_gath = gather_ptr.as_ptr() as *const c_void;
                    let mut p_d = d_ptr.as_ptr() as *mut c_void;
                    let mut p_m = m;
                    let mut p_n = n;
                    let mut p_k = k;
                    let mut p_idt = input_dtype;
                    let mut p_wdt = weights_dtype;
                    let mut p_odt = output_dtype;
                    let mut p_am = a_mode;
                    let mut p_ags = a_group_size;
                    let mut p_acl = a_code_layout;
                    let mut p_bbits = b_bits;
                    let mut p_bgs = b_group_size;
                    let mut p_bsign = b_signed_codes;
                    let mut p_sos = scale_strides.output_stride;
                    let mut p_sgs = scale_strides.group_stride;
                    let mut p_zpos = zp_strides.output_stride;
                    let mut p_zpgs = zp_strides.group_stride;
                    let mut p_osc = output_scale;
                    let mut p_sc = soft_cap_val;
                    let mut p_hsc = has_soft_cap;
                    let mut params: [*mut c_void; 28] = [
                        (&mut p_a as *mut *const c_void).cast(),
                        (&mut p_asc as *mut *const c_void).cast(),
                        (&mut p_bw as *mut *const c_void).cast(),
                        (&mut p_bs as *mut *const c_void).cast(),
                        (&mut p_bzp as *mut *const c_void).cast(),
                        (&mut p_bbi as *mut *const c_void).cast(),
                        (&mut p_obias as *mut *const c_void).cast(),
                        (&mut p_gath as *mut *const c_void).cast(),
                        (&mut p_d as *mut *mut c_void).cast(),
                        (&mut p_m as *mut u32).cast(),
                        (&mut p_n as *mut u32).cast(),
                        (&mut p_k as *mut u32).cast(),
                        (&mut p_idt as *mut u32).cast(),
                        (&mut p_wdt as *mut u32).cast(),
                        (&mut p_odt as *mut u32).cast(),
                        (&mut p_am as *mut u32).cast(),
                        (&mut p_ags as *mut u32).cast(),
                        (&mut p_acl as *mut u32).cast(),
                        (&mut p_bbits as *mut u32).cast(),
                        (&mut p_bgs as *mut u32).cast(),
                        (&mut p_bsign as *mut u32).cast(),
                        (&mut p_sos as *mut u32).cast(),
                        (&mut p_sgs as *mut u32).cast(),
                        (&mut p_zpos as *mut u32).cast(),
                        (&mut p_zpgs as *mut u32).cast(),
                        (&mut p_osc as *mut f32).cast(),
                        (&mut p_sc as *mut f32).cast(),
                        (&mut p_hsc as *mut u32).cast(),
                    ];
                    let total = m.saturating_mul(n).saturating_mul(8);
                    let block = 128u32;
                    let grid = total.div_ceil(block);
                    stream
                        .launch_kernel(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                        .expect("rocm_matmul_quant failed");
                });
            },
            MatmulB::Trellis {
                ..
            } => {
                return Err(MatmulRocmError::UnsupportedLayout {
                    path: "RocmMatmul",
                });
            },
        }

        if let Some(factors) = post_rht {
            self.output_rht.encode_fp_in_place(d.reborrow(), factors, None, m, n, command_buffer);
            if let Some(bias) = bias_buffer {
                let output_length = m.checked_mul(n).expect("matmul output length must fit in u32");
                let _ = bias.cpu_ptr();
                self.bias_add.encode(
                    None::<&<Rocm as Backend>::ScratchBuffer>,
                    bias,
                    d,
                    n,
                    output_length,
                    command_buffer,
                );
            }
        }

        Ok(())
    }
}

fn buffer_ref_dev_ptr(view: &impl BufferRef<Backend = Rocm>) -> SendPtr<u8> {
    let (buffer, range) = view.parts();
    SendPtr(buffer.device_address().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(range.start))
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    use uzu_engine_macros::uzu_test;

    use crate::{
        backends::{
            common::{
                Backend, Context, Kernels,
                kernel::matmul::{
                    MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput,
                    TRELLIS_CODES_TAIL_BYTES, TrellisFormat,
                },
            },
            rocm::Rocm,
        },
        data_type::DataType,
        tests::helpers::{buffer_to_vec, create_buffer_with_data, submit_command_buffer},
    };

    fn decode_state(
        format: TrellisFormat,
        row: &[u8],
        columns: usize,
        column: usize,
    ) -> u16 {
        let (width, transition_bits) = (format.vector_width as usize, format.transition_bits as usize);
        let block_columns = format.restart_columns.map_or(columns, |restart| restart as usize);
        let block_bits = 16 + (block_columns / width - 1) * transition_bits;
        let start_bit =
            (column / block_columns + 1) * block_bits - 16 - column % block_columns / width * transition_bits;
        (start_bit..start_bit + 16)
            .rev()
            .fold(0u32, |state, bit| (state << 1) | u32::from(row[bit / 8] >> (bit % 8) & 1)) as u16
    }

    fn decode_level(
        format: TrellisFormat,
        row: &[u8],
        columns: usize,
        column: usize,
    ) -> i32 {
        let state = u32::from(decode_state(format, row, columns, column));
        let hash = state.wrapping_mul(0xCFCC_B83F).wrapping_add(0x584B_4AA3);
        let hash = (hash ^ hash >> 16).wrapping_mul(0x85EB_CA6B);
        let hash_byte = ((hash ^ hash >> 16) >> (8 * (column % format.vector_width as usize))) & 0xFF;
        let field_sum: u32 = (0..4).map(|field| (hash_byte >> (2 * field)) & 3).sum();
        (8 * field_sum + ((3 * (hash_byte & 15)) & 15)) as i32 - 54
    }

    #[uzu_test]
    fn rocm_trellis_projection_matches_cpu_reference() {
        let context = futures::executor::block_on(<Rocm as Backend>::Context::new()).expect("ROCm context");
        assert!(
            context.device_name().is_some_and(|n| !n.is_empty()),
            "Expected non-empty ROCm device name"
        );
        let formats = [
            TrellisFormat {
                vector_width: 4,
                transition_bits: 8,
                restart_columns: Some(64),
            },
            TrellisFormat {
                vector_width: 2,
                transition_bits: 6,
                restart_columns: None,
            },
            TrellisFormat {
                vector_width: 4,
                transition_bits: 6,
                restart_columns: Some(64),
            },
        ];
        let codebook_host: [f32; 5] = [0.0123, 0.3, -0.25, 0.125, -0.0625];
        let padding = 24usize;
        let sentinel = 0x7F7Fu16;

        for format in formats {
            for (m, n, k) in [(1usize, 80usize, 128usize), (7, 80, 128), (16, 128, 64)] {
                let mut rng = SmallRng::seed_from_u64((m * k + n) as u64);
                let row_bytes = format.code_row_bytes(k as u32) as usize;
                let stored: Vec<u8> = (0..n * row_bytes).map(|_| rng.random()).collect();
                let row_scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.001..0.01)).collect();
                let values: Vec<i8> = (0..m * k).map(|_| rng.random_range(-127..=127)).collect();
                let group_sums_host: Vec<f32> = values
                    .chunks_exact(k)
                    .flat_map(|row| {
                        (0..4).map(|group| {
                            row.iter().skip(group).step_by(4).fold(0.0, |sum, &value| sum + f32::from(value))
                        })
                    })
                    .collect();
                let activation_scales_host: Vec<f32> = (0..m).map(|_| rng.random_range(0.001..0.01)).collect();

                let activations = create_buffer_with_data::<Rocm, i8>(&context, &values);
                let group_sums = create_buffer_with_data::<Rocm, f32>(&context, &group_sums_host);
                let activation_scales = create_buffer_with_data::<Rocm, f32>(&context, &activation_scales_host);
                let mut codes_with_tail = stored.clone();
                codes_with_tail.resize(stored.len() + TRELLIS_CODES_TAIL_BYTES, 0xA5);
                let codes = create_buffer_with_data::<Rocm, u8>(&context, &codes_with_tail);
                let row_scales_buffer = create_buffer_with_data::<Rocm, f32>(&context, &row_scales);
                let codebook = create_buffer_with_data::<Rocm, f32>(&context, &codebook_host);
                let mut output = create_buffer_with_data::<Rocm, u16>(&context, &vec![sentinel; m * (n + padding)]);

                let arguments = MatmulArguments {
                    a: MatmulA::Trellis {
                        values: &activations,
                        column_group_sums: &group_sums,
                        scales: &activation_scales,
                    },
                    b: MatmulB::Trellis {
                        codes: &codes,
                        row_scales: &row_scales_buffer,
                        codebook: &codebook,
                        format,
                    },
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput {
                        values: &mut output,
                        row_stride: Some((n + padding) as u32),
                        ops: MatmulDOps::none(),
                    },
                    gather_indices: None::<&<Rocm as Backend>::GlobalBuffer>,
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                };

                let mut command_buffer = context.create_command_buffer(None, None).unwrap();
                let mut matmul = <<<Rocm as Backend>::Kernels as Kernels>::MatmulKernel as MatmulKernel>::new(
                    &context,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                )
                .unwrap();
                matmul.encode(arguments, &mut command_buffer).unwrap();
                submit_command_buffer(command_buffer);

                let decoded_levels: Vec<i32> = stored
                    .chunks_exact(row_bytes)
                    .flat_map(|codes| (0..k).map(|column| decode_level(format, codes, k, column)))
                    .collect();
                let actual = buffer_to_vec::<Rocm, u16>(&output);
                let stride = n + padding;
                for (index, &got) in actual.iter().enumerate() {
                    let token = index / stride;
                    let row = index % stride;
                    if row < n {
                        let row_values = &values[token * k..(token + 1) * k];
                        let levels = &decoded_levels[row * k..(row + 1) * k];
                        let level_dot: i32 =
                            levels.iter().zip(row_values).map(|(&level, &value)| level * i32::from(value)).sum();
                        let sums = &group_sums_host[token * 4..token * 4 + 4];
                        let offsets_dot = (((sums[0] * codebook_host[1]) + sums[1] * codebook_host[2])
                            + sums[2] * codebook_host[3])
                            + sums[3] * codebook_host[4];
                        let dot = (level_dot as f32).mul_add(codebook_host[0], offsets_dot);
                        let want = bf16::from_f32(dot * row_scales[row] * activation_scales_host[token]).to_f32();
                        let got = bf16::from_bits(got).to_f32();
                        assert!(
                            (got - want).abs() <= 1e-2 * want.abs(),
                            "{format:?} ({m}, {n}, {k}) token {token} row {row}: got {got}, want {want}"
                        );
                    } else {
                        assert_eq!(got, sentinel);
                    }
                }
            }
        }
    }
}
