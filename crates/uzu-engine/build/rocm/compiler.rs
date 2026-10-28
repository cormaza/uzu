use std::{collections::HashMap, env, fs, path::PathBuf, process::Command};

use anyhow::{Context, bail};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Expr, FnArg, GenericArgument, GenericParam, Ident, Item, ItemFn, PathArguments, Type, punctuated::Punctuated,
    token::Comma, visit_mut::VisitMut,
};
use walkdir::WalkDir;

use crate::common::{
    codegen::write_tokens,
    compiler::Compiler,
    constraints::Constraints,
    gpu_type_paths::GpuTypePaths,
    gpu_types::GpuTypes,
    identifiers::{ArgumentName, KernelName, KernelPath},
    kernel::{Kernel, KernelArgument, KernelArgumentType, KernelBufferAccess, KernelParameter, KernelParameterType},
};

#[derive(PartialEq, Debug)]
pub enum FunctionArgumentType {
    Buffer(KernelBufferAccess),
    Constant(Type, Option<Expr>),
    Scalar(Type),
    Specialization(Type),
}

#[derive(PartialEq, Debug)]
pub struct FunctionArgument {
    pub name: Ident,
    pub conditional: Option<Expr>,
    pub ty: FunctionArgumentType,
}

impl FunctionArgument {
    fn to_kernel_argument(
        &self,
        gpu_type_paths: &GpuTypePaths,
    ) -> Option<KernelArgument> {
        Some(KernelArgument {
            name: ArgumentName::from(self.name.to_string()),
            conditional: self.conditional.is_some(),
            ty: match &self.ty {
                FunctionArgumentType::Buffer(access) => KernelArgumentType::Buffer(access.clone()),
                FunctionArgumentType::Constant(ty, None) => KernelArgumentType::Constant(
                    format!("&[{}]", canonicalize_type_text(ty, gpu_type_paths)).into_boxed_str(),
                ),
                FunctionArgumentType::Constant(ty, Some(size)) => KernelArgumentType::Constant(
                    format!("&[{}; {}]", canonicalize_type_text(ty, gpu_type_paths), size.to_token_stream())
                        .into_boxed_str(),
                ),
                FunctionArgumentType::Scalar(ty) => {
                    KernelArgumentType::Constant(canonicalize_type_text(ty, gpu_type_paths).into_boxed_str())
                },
                FunctionArgumentType::Specialization(_) => {
                    return None;
                },
            },
        })
    }

    fn to_kernel_parameter(
        &self,
        gpu_type_paths: &GpuTypePaths,
    ) -> Option<KernelParameter> {
        Some(KernelParameter {
            name: self.name.to_string().into_boxed_str(),
            ty: match &self.ty {
                FunctionArgumentType::Specialization(ty) => {
                    KernelParameterType::Value(canonicalize_type_text(ty, gpu_type_paths).into_boxed_str())
                },
                _ => {
                    return None;
                },
            },
        })
    }
}

fn canonicalize_type_text(
    ty: &Type,
    gpu_type_paths: &GpuTypePaths,
) -> String {
    struct TypeNames<'a>(&'a GpuTypePaths);

    impl VisitMut for TypeNames<'_> {
        fn visit_type_path_mut(
            &mut self,
            ty: &mut syn::TypePath,
        ) {
            if ty.qself.is_none()
                && let Some(segment) = ty.path.segments.last()
                && matches!(segment.arguments, PathArguments::None)
                && self.0.full_path_for(&segment.ident.to_string()).is_some()
            {
                let ident = &segment.ident;
                ty.path = syn::parse_quote! { #ident };
            }
            syn::visit_mut::visit_type_path_mut(self, ty);
        }
    }

    let mut canonicalized = ty.clone();
    TypeNames(gpu_type_paths).visit_type_mut(&mut canonicalized);
    canonicalized.to_token_stream().to_string().replace(" :: ", "::")
}

#[derive(PartialEq, Debug)]
pub enum FunctionParameterType {
    Type,
    Value(Type),
}

#[derive(PartialEq, Debug)]
pub struct FunctionParameter {
    pub name: Ident,
    pub ty: FunctionParameterType,
}

impl FunctionParameter {
    fn to_kernel_parameter(
        &self,
        gpu_type_paths: &GpuTypePaths,
    ) -> KernelParameter {
        KernelParameter {
            name: self.name.to_string().into_boxed_str(),
            ty: match &self.ty {
                FunctionParameterType::Type => KernelParameterType::Type,
                FunctionParameterType::Value(ty) => {
                    KernelParameterType::Value(canonicalize_type_text(ty, gpu_type_paths).into_boxed_str())
                },
            },
        }
    }
}

pub struct RocmCompiler {
    cpu_src_dir: PathBuf,
    hip_src_path: PathBuf,
    build_dir: PathBuf,
    dsl_dir: PathBuf,
    hsaco_path: PathBuf,
}

fn detect_rocm_arch() -> String {
    if let Ok(arch) = env::var("UZU_ROCM_ARCH")
        && !arch.trim().is_empty()
    {
        return arch.trim().to_string();
    }
    for bin in ["rocminfo", "/opt/rocm/bin/rocminfo"] {
        if let Ok(output) = Command::new(bin).output()
            && output.status.success()
            && let Ok(text) = String::from_utf8(output.stdout)
        {
            for line in text.lines() {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("Name:") {
                    let name = rest.trim();
                    if name.starts_with("gfx") && !name.contains("generic") {
                        return name.to_string();
                    }
                }
            }
        }
    }
    "gfx1200".to_string()
}

impl RocmCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
        let cpu_src_dir = manifest_dir.join("src/backends/cpu/kernel");
        let hip_src_path = manifest_dir.join("src/backends/rocm/kernel/hip/kernels.hip");
        println!("cargo::rerun-if-changed={}", cpu_src_dir.display());
        println!("cargo::rerun-if-changed={}", hip_src_path.display());

        let out_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);
        let build_dir = out_dir.join("rocm");
        let dsl_dir = out_dir.join("rocm_dsl");
        let hsaco_path = out_dir.join("rocm_kernels.hsaco");
        fs::create_dir_all(&build_dir).with_context(|| format!("cannot create {}", build_dir.display()))?;
        fs::create_dir_all(&dsl_dir).with_context(|| format!("cannot create {}", dsl_dir.display()))?;

        Ok(Self {
            cpu_src_dir,
            hip_src_path,
            build_dir,
            dsl_dir,
            hsaco_path,
        })
    }

    fn compile_hip(&self) -> anyhow::Result<()> {
        let arch = detect_rocm_arch();
        let hipcc = env::var("HIPCC").unwrap_or_else(|_| {
            if PathBuf::from("/opt/rocm/bin/hipcc").exists() {
                "/opt/rocm/bin/hipcc".to_string()
            } else {
                "hipcc".to_string()
            }
        });

        let status = Command::new(&hipcc)
            .arg("--genco")
            .arg("-O3")
            .arg(format!("--offload-arch={arch}"))
            .arg(&self.hip_src_path)
            .arg("-o")
            .arg(&self.hsaco_path)
            .status()
            .with_context(|| format!("failed to invoke {hipcc} for {}", self.hip_src_path.display()))?;

        if !status.success() {
            bail!("hipcc failed with status {status}");
        }
        Ok(())
    }

    fn compile(
        &self,
        source_path: PathBuf,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<(KernelPath, Box<[Kernel]>)> {
        let rel_path = source_path.strip_prefix(&self.cpu_src_dir).context("source is not in cpu_src_dir")?;
        let src_rel_path: KernelPath =
            rel_path.with_extension("").as_os_str().to_str().unwrap().split("/").map(|s| s.to_string()).collect();

        let source_contents = fs::read_to_string(&source_path).context("cannot read the source file")?;
        let mut source_ast = syn::parse_file(&source_contents).context("cannot parse ast")?;

        let mut kernels = Vec::new();
        let mut transformed_items = Vec::new();

        for item in source_ast.items {
            match item {
                Item::Use(u) => {
                    let s = u.to_token_stream().to_string();
                    if s.contains("uzu_engine_macros") && s.contains("kernel") {
                        continue;
                    }
                    transformed_items.push(Item::Use(u));
                },
                Item::Fn(ifn) if ifn.attrs.iter().any(|attr| attr.path().is_ident("kernel")) => {
                    let mut stripped = ifn.clone();
                    stripped.attrs.retain(|attr| {
                        !attr.path().is_ident("kernel")
                            && !attr.path().is_ident("variants")
                            && !attr.path().is_ident("constraint")
                    });
                    for arg in &mut stripped.sig.inputs {
                        if let FnArg::Typed(pat) = arg {
                            pat.attrs.retain(|attr| {
                                let p = attr.path();
                                !(p.is_ident("specialize") || p.is_ident("optional"))
                            });
                        }
                    }
                    let kernel = self.compile_kernel(ifn, gpu_type_paths)?;
                    let include_file = format!("/rocm/{}.rs", kernel.name);
                    let include_item: Item = syn::parse_quote! {
                        include!(concat!(env!("OUT_DIR"), #include_file));
                    };
                    transformed_items.push(Item::Fn(stripped));
                    transformed_items.push(include_item);
                    kernels.push(kernel);
                },
                other => transformed_items.push(other),
            }
        }

        // Mirror non-manual DSL files into $OUT_DIR/rocm_dsl/<rel_path>
        let rel_str = rel_path.to_str().unwrap_or("");
        if rel_str != "mod.rs" && rel_str != "radix_top_k_small.rs" && !rel_str.starts_with("matmul/") {
            let dst_path = self.dsl_dir.join(rel_path);
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            source_ast.items = transformed_items;
            let mut formatted = prettyplease::unparse(&source_ast).replace("cpu::kernel::", "rocm::kernel::");
            if rel_str == "attention/mod.rs" {
                formatted = formatted
                    .replace("cpu::", "rocm::")
                    .replace("Cpu,", "Rocm,")
                    .replace("AttentionCpuKernel", "AttentionRocmKernel")
                    .replace("AttentionSinglePassCpuKernel", "AttentionSinglePassRocmKernel")
                    .replace("SigmoidGateCpuKernel", "SigmoidGateRocmKernel")
                    .replace("CpuCommandBufferEncoding", "RocmCommandBufferEncoding")
                    .replace("CpuContext", "RocmContext")
                    .replace("CpuError", "RocmError")
                    .replace("<Cpu as Backend>", "<Rocm as Backend>")
                    .replace("<Cpu>", "<Rocm>")
                    .replace("= Cpu", "= Rocm");
            }
            fs::write(&dst_path, formatted)?;
        }

        Ok((src_rel_path, kernels.into_boxed_slice()))
    }

    fn compile_kernel(
        &self,
        ifn: ItemFn,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<Kernel> {
        let mut kernel_ident = None;
        let mut function_variants = Vec::new();
        let mut function_constraints: Vec<Expr> = Vec::new();

        for attr in ifn.attrs {
            match attr.path().get_ident().map(|i| i.to_string()).as_ref().map(|s| s.as_ref()) {
                Some("kernel") => {
                    if kernel_ident.is_some() {
                        bail!("Multiple kernel attributes!");
                    }
                    kernel_ident = Some(attr.parse_args::<Ident>().context("cannot parse kernel attribute arg")?);
                },
                Some("variants") => {
                    let mut args = attr
                        .parse_args_with(Punctuated::<Expr, Comma>::parse_terminated)
                        .context("cannot parse variants attribute args")?
                        .into_iter();

                    function_variants.push((
                        syn::parse2::<Ident>(args.next().context("variant must have a name")?.into_token_stream())
                            .unwrap(),
                        args.collect::<Box<[_]>>(),
                    ));
                },
                Some("constraint") => {
                    let expr = attr.parse_args::<Expr>().context("cannot parse constraint attribute")?;
                    function_constraints.push(expr);
                },
                _ => {},
            }
        }

        let Some(kernel_ident) = kernel_ident else {
            bail!("Not a kernel")
        };

        let function_ident = ifn.sig.ident;
        let function_parameters = ifn
            .sig
            .generics
            .params
            .into_iter()
            .map(|parameter| {
                Ok(match parameter {
                    GenericParam::Type(parameter) => FunctionParameter {
                        name: parameter.ident,
                        ty: FunctionParameterType::Type,
                    },
                    GenericParam::Const(parameter) => FunctionParameter {
                        name: parameter.ident,
                        ty: FunctionParameterType::Value(parameter.ty),
                    },
                    parameter => {
                        bail!("unsupported kernel parameter type: {parameter:?}")
                    },
                })
            })
            .collect::<anyhow::Result<Box<[FunctionParameter]>>>()?;
        let function_arguments = ifn
            .sig
            .inputs
            .into_iter()
            .map(|argument| {
                let FnArg::Typed(argument) = argument else {
                    bail!("self argument in a kernel is not supported");
                };

                let name = syn::parse2(argument.pat.into_token_stream()).context("cannot parse the argument name")?;
                let specialize = argument.attrs.iter().any(|attr| attr.path().is_ident("specialize"));
                let conditional = argument
                    .attrs
                    .iter()
                    .find_map(|attr| attr.path().is_ident("optional").then(|| attr.parse_args::<Expr>().unwrap()));

                let ty = if specialize {
                    FunctionArgumentType::Specialization(*argument.ty)
                } else if conditional.is_some() {
                    let Type::Path(ty) = *argument.ty else {
                        bail!("conditional argument must be a type path");
                    };
                    if ty.path.segments.len() != 1 {
                        bail!("conditional argument type path must have one segment");
                    }
                    let seg = &ty.path.segments[0];
                    if seg.ident != "Option" {
                        bail!("conditional argument type must be Option<...>");
                    }
                    let PathArguments::AngleBracketed(option_arguments) = &seg.arguments else {
                        bail!("conditional argument type must be angle bracketed");
                    };
                    if option_arguments.args.len() != 1 {
                        bail!("conditional argument type Option must have one generic argument");
                    }
                    let generic_argument = &option_arguments.args[0];
                    let GenericArgument::Type(inner_ty) = generic_argument else {
                        bail!("conditional argument type Option must have a type argument")
                    };
                    Self::parse_type(inner_ty.clone()).context("failed to parse conditional argument type")?
                } else {
                    Self::parse_type(*argument.ty).context("failed to parse argument type")?
                };

                Ok(FunctionArgument {
                    name,
                    conditional,
                    ty,
                })
            })
            .collect::<anyhow::Result<Box<[FunctionArgument]>>>()?;

        let kernel_parameters = function_parameters
            .iter()
            .map(|parameter| parameter.to_kernel_parameter(gpu_type_paths))
            .chain(function_arguments.iter().flat_map(|argument| argument.to_kernel_parameter(gpu_type_paths)))
            .collect::<Box<[KernelParameter]>>();

        let kernel_arguments = function_arguments
            .iter()
            .flat_map(|argument| argument.to_kernel_argument(gpu_type_paths))
            .collect::<Box<[KernelArgument]>>();

        if function_parameters.len() != function_variants.len() {
            bail!(
                "Kernel function has {} generics != {} #[variants(...)]!",
                function_parameters.len(),
                function_variants.len()
            );
        }

        for (parameter, (variant_name, _)) in std::iter::zip(function_parameters.iter(), function_variants.iter()) {
            if &parameter.name != variant_name {
                bail!("Parameter name doesn't match variant name: {} | {}", parameter.name, variant_name);
            }
        }

        let trait_ident = format_ident!("{kernel_ident}Kernel");
        let struct_ident = format_ident!("{kernel_ident}RocmKernel");

        let (struct_fields_defs, struct_fields_sets): (Vec<TokenStream>, Vec<TokenStream>) = function_parameters
            .iter()
            .map(|parameter| {
                let ident = &parameter.name;
                let ty = match &parameter.ty {
                    FunctionParameterType::Type => quote! { crate::data_type::DataType },
                    FunctionParameterType::Value(ty) => quote! { #ty },
                };
                (quote! { pub(crate) #ident: #ty }, quote! { #ident })
            })
            .chain(function_arguments.iter().flat_map(|argument| {
                let FunctionArgumentType::Specialization(ty) = &argument.ty else {
                    return None;
                };
                let ident = &argument.name;
                Some((quote! { pub(crate) #ident: #ty }, quote! { #ident }))
            }))
            .collect();

        let parameter_args: Vec<TokenStream> = kernel_parameters
            .iter()
            .map(|parameter| {
                let parameter_ident: Ident = syn::parse_str(parameter.name.as_ref())?;
                Ok(match &parameter.ty {
                    KernelParameterType::Type => {
                        quote! { #[allow(non_snake_case)] #parameter_ident: crate::data_type::DataType }
                    },
                    KernelParameterType::Value(ty) => {
                        let ty = gpu_type_paths.resolve_type(ty)?;
                        quote! { #[allow(non_snake_case)] #parameter_ident: #ty }
                    },
                })
            })
            .collect::<anyhow::Result<_>>()?;

        let mut encode_args_defs: Vec<_> = kernel_arguments
            .iter()
            .map(|argument| {
                let argument_ident: Ident = syn::parse_str(argument.name.as_ref()).context("cannot parse ident")?;

                let mut ty = match &argument.ty {
                    KernelArgumentType::Buffer(access) => match access {
                        KernelBufferAccess::Read => {
                            quote! { impl crate::backends::common::BufferRef<Backend = crate::backends::rocm::Rocm> }
                        },
                        KernelBufferAccess::ReadWrite => {
                            quote! { impl crate::backends::common::BufferMut<Backend = crate::backends::rocm::Rocm> }
                        },
                    },
                    KernelArgumentType::Constant(ty) => {
                        let ty = gpu_type_paths.resolve_type(ty)?;
                        quote! { #ty }
                    },
                };

                if argument.conditional {
                    ty = quote! { Option<#ty> };
                }

                Ok(quote! { #argument_ident: #ty })
            })
            .collect::<anyhow::Result<_>>()?;

        let argument_copies = function_arguments
            .iter()
            .flat_map(|argument| {
                let argument_ident = &argument.name;
                match &argument.ty {
                    FunctionArgumentType::Buffer(access) => {
                        let (buffer_ptr, buffer_ptr_wrapper) = match access {
                            KernelBufferAccess::Read => (
                                quote! { crate::backends::rocm::buffer::RocmBufferExt::cpu_address(__dsl_buffer).as_ptr().cast::<u8>().cast_const() },
                                quote! { crate::utils::pointers::SendPtr },
                            ),
                            KernelBufferAccess::ReadWrite => (
                                quote! { crate::backends::rocm::buffer::RocmBufferExt::cpu_address(__dsl_buffer).as_ptr().cast::<u8>() },
                                quote! { crate::utils::pointers::SendPtrMut },
                            ),
                        };

                        if argument.conditional.is_some() {
                            Some(quote! {
                                let #argument_ident = #argument_ident.map(|__dsl_buffer_impl| unsafe {
                                    let (__dsl_buffer, __dsl_range) = __dsl_buffer_impl.parts();
                                    #buffer_ptr_wrapper(#buffer_ptr.byte_add(__dsl_range.start))
                                });
                            })
                        } else {
                            Some(quote! {
                                let #argument_ident = unsafe {
                                    let (__dsl_buffer, __dsl_range) = #argument_ident.parts();
                                    #buffer_ptr_wrapper(#buffer_ptr.byte_add(__dsl_range.start))
                                };
                            })
                        }
                    },
                    FunctionArgumentType::Constant(_, None) => {
                        Some(quote! { let #argument_ident = #argument_ident.to_vec().into_boxed_slice(); })
                    },
                    FunctionArgumentType::Constant(_, Some(_)) => {
                        Some(quote! { let #argument_ident = Box::new(*#argument_ident); })
                    },
                    FunctionArgumentType::Scalar(_) => None,
                    FunctionArgumentType::Specialization(_) => {
                        Some(quote! { let #argument_ident = self.#argument_ident; })
                    },
                }
            })
            .collect::<Vec<_>>();

        let gpu_dispatch_call = Self::make_gpu_dispatch(&kernel_ident.to_string(), &function_arguments);

        let make_encode = |generics: TokenStream| -> TokenStream {
            let monomorphized_function = if !generics.is_empty() {
                quote! { self::#function_ident::<#generics> }
            } else {
                quote! { self::#function_ident }
            };

            let function_call_args = function_arguments.iter().map(|argument| {
                let argument_ident = &argument.name;
                match &argument.ty {
                    FunctionArgumentType::Buffer(_) => {
                        if argument.conditional.is_some() {
                            quote! { #argument_ident.map(|p| p.as_ptr() as _) }
                        } else {
                            quote! { #argument_ident.as_ptr() as _ }
                        }
                    },
                    FunctionArgumentType::Constant(_, _) => quote! { &*#argument_ident },
                    FunctionArgumentType::Scalar(_) | FunctionArgumentType::Specialization(_) => {
                        quote! { #argument_ident }
                    },
                }
            });

            let function_call_args_joined = function_call_args.fold(quote! {}, |a, b| {
                if !a.is_empty() && !b.is_empty() {
                    quote! {#a , #b}
                } else {
                    quote! {#a #b}
                }
            });

            quote! {
                command_buffer.push_command(move || #monomorphized_function(#function_call_args_joined));
            }
        };

        let encode_body = if !function_parameters.is_empty() {
            let mut parameter_idents = function_parameters.iter().map(|p| p.name.clone()).fold(quote! {}, |a, b| {
                if !a.is_empty() {
                    quote! {#a , self.#b}
                } else {
                    quote! {self.#b}
                }
            });

            if function_parameters.len() > 1 {
                parameter_idents = quote! { (#parameter_idents) };
            }

            let variant_value_strs: Vec<String> = function_variants
                .iter()
                .flat_map(|(_, variants)| variants.iter().map(|v| v.to_token_stream().to_string()))
                .collect();
            let constraints = (!function_constraints.is_empty()).then(|| {
                Constraints::new(
                    variant_value_strs.iter().map(|s| s.as_str()),
                    function_constraints.iter().map(|c| c.to_token_stream().to_string()),
                )
            });

            let match_arms = function_parameters
                .iter()
                .zip(function_variants.iter())
                .map(|(parameter, (_, variants))| {
                    variants.iter().map(|variant| {
                        (
                            match parameter.ty {
                                FunctionParameterType::Type => {
                                    let dtype =
                                        format_ident!("{}", variant.to_token_stream().to_string().to_uppercase());
                                    quote! { crate::data_type::DataType::#dtype }
                                },
                                FunctionParameterType::Value(_) => quote! { #variant },
                            },
                            quote! { #variant },
                        )
                    })
                })
                .multi_cartesian_product()
                .filter(|variants| {
                    let Some(constraints) = &constraints else {
                        return true;
                    };
                    constraints.satisfied(
                        function_parameters
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (p.name.to_string(), variants[i].1.to_string())),
                    )
                })
                .map(|variants| {
                    let (match_variants, generic_variants): (Vec<TokenStream>, Vec<TokenStream>) =
                        variants.into_iter().unzip();

                    let mut match_variant = match_variants.iter().fold(quote! {}, |a, b| {
                        if !a.is_empty() && !b.is_empty() {
                            quote! {#a , #b}
                        } else {
                            quote! {#a #b}
                        }
                    });

                    if match_variants.len() > 1 {
                        match_variant = quote! { (#match_variant) };
                    }

                    let generic_variant = generic_variants.iter().fold(quote! {}, |a, b| {
                        if !a.is_empty() && !b.is_empty() {
                            quote! {#a , #b}
                        } else {
                            quote! {#a #b}
                        }
                    });

                    let encode = make_encode(generic_variant);

                    quote! { #match_variant => { #encode } }
                })
                .collect::<Vec<_>>();

            quote! {
                match #parameter_idents {
                    #(#match_arms ,)*
                    __dsl_variant => unimplemented!("variant doesn't exist: {__dsl_variant:?}"),
                }
            }
        } else {
            make_encode(quote! {})
        };

        encode_args_defs.push(
            quote! { command_buffer: &mut <<crate::backends::rocm::Rocm as crate::backends::common::Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding },
        );

        let specialization_validation = function_arguments.iter().filter_map(|argument| {
            if !matches!(argument.ty, FunctionArgumentType::Specialization(_)) {
                return None;
            }
            let condition = argument.conditional.as_ref()?;
            let name = &argument.name;
            Some(quote! {
                assert_eq!(#name.is_some(), #condition, concat!("invalid presence for specialization ", stringify!(#name)));
            })
        });

        let tokens = quote! {
            #[allow(non_snake_case)]
            pub struct #struct_ident {
                #(#struct_fields_defs ,)*
            }

            #[allow(clippy::style, clippy::complexity, clippy::perf)]
            impl crate::backends::common::kernel::#trait_ident for #struct_ident {
                type Backend = crate::backends::rocm::Rocm;

                fn new(#[allow(unused)] context: &crate::backends::rocm::context::RocmContext #(, #parameter_args)*) -> Result<Self, crate::backends::rocm::error::RocmError> {
                    #(#specialization_validation)*
                    Ok(Self {
                        #(#struct_fields_sets ,)*
                    })
                }

                fn encode(&self, #(#encode_args_defs),*) {
                    #(#argument_copies)*
                    #gpu_dispatch_call
                    #encode_body
                }
            }
        };

        let out_path = self.build_dir.join(kernel_ident.to_string()).with_extension("rs");
        write_tokens(tokens, &out_path).context("cannot write rocm bindings")?;

        Ok(Kernel {
            name: KernelName::from(kernel_ident.to_string()),
            parameters: kernel_parameters,
            arguments: kernel_arguments,
        })
    }

    fn make_gpu_dispatch(
        kernel_name: &str,
        function_arguments: &[FunctionArgument],
    ) -> TokenStream {
        let hook_fn = match kernel_name {
            "Activation" => Some("try_encode_activation"),
            "TensorAddBias" => Some("try_encode_tensor_add_bias"),
            "TensorAddScale" => Some("try_encode_tensor_add_scale"),
            "LogitTransform" => Some("try_encode_logit_transform"),
            "SigmoidGate" => Some("try_encode_sigmoid_gate"),
            "PoolingMean" => Some("try_encode_pooling_mean"),
            "Softmax" => Some("try_encode_softmax"),
            "Normalization" => Some("try_encode_normalization"),
            "QKVNorm" => Some("try_encode_qkv_norm"),
            "KVCacheUpdate" => Some("try_encode_kv_cache_update"),
            "GatedActMul" => Some("try_encode_gated_act_mul"),
            "AttentionSinglePass" => Some("try_encode_attention_single_pass"),
            "InputEmbeddingLookup" => Some("try_encode_input_embedding_lookup"),
            _ => None,
        };
        let Some(hook_fn) = hook_fn else {
            return quote! {};
        };
        let hook_ident = format_ident!("{hook_fn}");
        let args: Vec<TokenStream> = function_arguments
            .iter()
            .filter_map(|arg| {
                if matches!(arg.ty, FunctionArgumentType::Specialization(_)) {
                    return None;
                }
                let name = &arg.name;
                match &arg.ty {
                    FunctionArgumentType::Constant(_, _) => Some(quote! { &#name }),
                    _ => Some(quote! { #name }),
                }
            })
            .collect();

        quote! {
            if crate::backends::rocm::kernel::hip_ops::#hook_ident(self #(, #args)*, command_buffer) {
                return;
            }
        }
    }

    fn parse_type(ty: Type) -> anyhow::Result<FunctionArgumentType> {
        Ok(match ty {
            Type::Ptr(ty) => FunctionArgumentType::Buffer(if matches!(ty.mutability, syn::PointerMutability::Mut(_)) {
                KernelBufferAccess::ReadWrite
            } else {
                KernelBufferAccess::Read
            }),
            Type::Reference(ty) => match *ty.elem {
                Type::Slice(ty) => FunctionArgumentType::Constant(*ty.elem, None),
                Type::Array(ty) => FunctionArgumentType::Constant(*ty.elem, Some(ty.len)),
                ty => bail!("unsupported reference type: {} ({:?})", ty.to_token_stream(), ty),
            },
            Type::Path(ty) => FunctionArgumentType::Scalar(Type::Path(ty)),
            ty => bail!("unsupported type: {} ({:?})", ty.to_token_stream(), ty),
        })
    }

    fn bindgen<'a>(
        &self,
        objects: impl IntoIterator<Item = &'a (KernelPath, Box<[Kernel]>)> + Clone,
    ) -> anyhow::Result<()> {
        let out_path = self.build_dir.with_extension("rs");

        let dsl_modules = [
            ("activation", false),
            ("activation_transform", true),
            ("attention", true),
            ("convolution", false),
            ("embedding", false),
            ("gated_act_mul", false),
            ("gdn", false),
            ("logit_transform", false),
            ("normalization", false),
            ("pooling", false),
            ("sampling", false),
            ("short_conv", false),
            ("softmax", false),
            ("tensor_add_bias", false),
            ("tensor_add_scale", false),
            ("weaver", false),
        ];

        let mod_decls = dsl_modules.iter().map(|(name, is_pub_crate)| {
            let mod_ident = format_ident!("{name}");
            let mod_path = self.dsl_dir.join(name).join("mod.rs").to_str().unwrap().to_string();
            if *is_pub_crate {
                quote! {
                    #[path = #mod_path]
                    pub(crate) mod #mod_ident;
                }
            } else {
                quote! {
                    #[path = #mod_path]
                    mod #mod_ident;
                }
            }
        });

        let associated_types = objects.into_iter().flat_map(|(file_path, kernels)| {
            kernels.iter().map(|kernel| {
                let file_path: TokenStream = file_path.iter().join("::").parse().unwrap();
                let kernel_trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
                let kernel_struct_name = format_ident!("{}RocmKernel", kernel.name.as_ref());
                quote! { type #kernel_trait_name = #file_path::#kernel_struct_name; }
            })
        });

        let tokens = quote! {
            #(#mod_decls)*

            macro_rules! autogen_kernels {
                () => {
                    #(#associated_types)*
                }
            }
        };

        write_tokens(tokens, &out_path).context("cannot write rocm dsl bindings")?;
        Ok(())
    }
}

impl Compiler for RocmCompiler {
    fn build(
        &self,
        _gpu_types: &GpuTypes,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        self.compile_hip().context("cannot compile HIP GPU kernels")?;

        let objects = WalkDir::new(&self.cpu_src_dir)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_type().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("rs")
            })
            .map(|entry| self.compile(entry.into_path(), gpu_type_paths))
            .collect::<anyhow::Result<Vec<(KernelPath, Box<[Kernel]>)>>>()
            .context("cannot compile rocm dsl sources")?;

        self.bindgen(&objects).context("cannot generate rocm bindings")?;

        Ok(objects.into_iter().collect())
    }
}
