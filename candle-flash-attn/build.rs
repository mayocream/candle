use anyhow::{Context, Result};
use candle_flash_attn_build::{cutlass_include_arg, fetch_cutlass};
use std::path::{Path, PathBuf};
use std::process::Command;

const CUTLASS_COMMIT: &str = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
const COMPUTE_CAP: usize = 80;
const KERNEL_FILE: &str = "kernels/flash_fwd_hdim128_sm80_driver.cu";
const PTX_FILE: &str = "flash_fwd_hdim128_sm80_driver.ptx";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={KERNEL_FILE}");
    println!("cargo::rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo::rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo::rerun-if-changed=kernels/flash.h");
    println!("cargo::rerun-if-changed=kernels/philox.cuh");
    println!("cargo::rerun-if-changed=kernels/softmax.h");
    println!("cargo::rerun-if-changed=kernels/utils.h");
    println!("cargo::rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo::rerun-if-changed=kernels/block_info.h");
    println!("cargo::rerun-if-changed=kernels/static_switch.h");
    println!("cargo::rerun-if-changed=kernels/hardware_info.h");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let cutlass_dir = fetch_cutlass(&out_dir, CUTLASS_COMMIT)?;
    let cutlass_include = cutlass_include_arg(&cutlass_dir);
    let ptx_path = out_dir.join(PTX_FILE);

    compile_ptx(&ptx_path, &cutlass_include)?;
    Ok(())
}

fn compile_ptx(ptx_path: &Path, cutlass_include: &str) -> Result<()> {
    let mut command = Command::new("nvcc");
    command
        .arg("-ptx")
        .arg("-std=c++17")
        .arg("-O3")
        .arg(format!("-arch=compute_{COMPUTE_CAP}"))
        .arg("-D_USE_MATH_DEFINES")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg(cutlass_include)
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-DFLASHATTENTION_DISABLE_DROPOUT")
        .arg("-DFLASHATTENTION_DISABLE_ALIBI")
        .arg("-DFLASHATTENTION_DISABLE_LOCAL")
        .arg("-DFLASHATTENTION_DISABLE_SOFTCAP")
        .arg("-DFLASHATTENTION_DISABLE_UNEVEN_K")
        .arg("-o")
        .arg(ptx_path)
        .arg(KERNEL_FILE);

    let status = command.status().context("failed to run nvcc")?;
    if !status.success() {
        anyhow::bail!("nvcc failed with status: {status}");
    }
    Ok(())
}
