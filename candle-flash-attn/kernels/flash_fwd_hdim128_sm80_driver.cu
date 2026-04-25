/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include "flash_fwd_launch_template.h"

template<typename T, int BlockN, bool IsEvenMN>
__device__ __forceinline__ void flash_fwd_hdim128_flux2(Flash_fwd_params const &params) {
    using KernelTraits = Flash_fwd_kernel_traits<128, 128, BlockN, 4, false, false, T>;
    flash::compute_attn<
        KernelTraits,
        false,      // Is_dropout
        false,      // Is_causal
        false,      // Is_local
        false,      // Has_alibi
        IsEvenMN,
        true,       // Is_even_K
        false,      // Is_softcap
        false       // Return_softmax
    >(params);
}

extern "C" __global__ void flash_fwd_hdim128_fp16_block32_even(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::half_t, 32, true>(params);
}

extern "C" __global__ void flash_fwd_hdim128_fp16_block32_uneven(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::half_t, 32, false>(params);
}

extern "C" __global__ void flash_fwd_hdim128_fp16_block64_even(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::half_t, 64, true>(params);
}

extern "C" __global__ void flash_fwd_hdim128_fp16_block64_uneven(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::half_t, 64, false>(params);
}

extern "C" __global__ void flash_fwd_hdim128_bf16_block32_even(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::bfloat16_t, 32, true>(params);
}

extern "C" __global__ void flash_fwd_hdim128_bf16_block32_uneven(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::bfloat16_t, 32, false>(params);
}

extern "C" __global__ void flash_fwd_hdim128_bf16_block64_even(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::bfloat16_t, 64, true>(params);
}

extern "C" __global__ void flash_fwd_hdim128_bf16_block64_uneven(
    KERNEL_PARAM_MODIFIER const Flash_fwd_params params
) {
    flash_fwd_hdim128_flux2<cutlass::bfloat16_t, 64, false>(params);
}
