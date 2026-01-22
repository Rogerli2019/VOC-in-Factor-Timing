#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code D (Optimized): AIPT Complexity Engine
- OPTIMIZATION 1: Implicit Broadcasting (Removes expensive memory tiling)
- OPTIMIZATION 2: TensorFloat32 (Enables faster A100 matrix math)
- OPTIMIZATION 3: Block GEMM (Unrolled loops)
- STRUCTURE: output_dir / feature_set / window / seed_X.joblib
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, vmap
import joblib
from tqdm.auto import tqdm

# Enable 64-bit precision (Keep this for financial accuracy)
jax.config.update("jax_enable_x64", True)

# OPTIMIZATION: Enable TensorFloat32 for A100 GPUs
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

# ======================================================
# 1. HELPER FUNCTIONS
# ======================================================
def generate_plist_numpy(trnwin, max_p):
    """Generates the list of P steps (tuple for JAX static args)."""
    def matlab_range(start, step, stop):
        if step == 0: step = 1
        return np.arange(start, stop + (step * 0.001), step).astype(int)

    seg0 = [2]
    step1 = int(np.floor(trnwin / 10))
    seg1 = matlab_range(5, step1, trnwin - 5)
    seg2 = matlab_range(trnwin - 4, 2, trnwin + 4)
    step3 = int(np.floor(trnwin / 2))
    seg3 = matlab_range(trnwin + 5, step3, 30 * trnwin)
    step4 = 10 * trnwin
    seg4 = matlab_range(31 * trnwin, step4, max_p - 1)
    seg5 = [max_p]

    raw_plist = np.concatenate([seg0, seg1, seg2, seg3, seg4, seg5])
    effective_plist = (np.floor(raw_plist / 2) * 2).astype(int)
    final_p = np.unique(effective_plist[effective_plist > 0])
    
    final_p = final_p[final_p <= max_p]
    return tuple(final_p.tolist())

def generate_shuffled_gammas(num_seeds, max_p, gamma_list, seed):
    """Distributes gamma values across features."""
    num_features = max_p // 2
    num_gammas = len(gamma_list)
    block_size = num_features // num_gammas
    
    fixed_part_list = []
    for g in gamma_list:
        fixed_part_list.append(np.full(block_size, g))
    
    fixed_part = np.concatenate(fixed_part_list)
    
    rng = np.random.default_rng(seed)
    all_gammas_shuffled = np.zeros((num_seeds, num_features))
    
    remainder_count = num_features - (block_size * num_gammas)
    gamma_values = np.array(gamma_list)

    for i in range(num_seeds):
        current_gammas = [fixed_part]
        if remainder_count > 0:
            unique_remainder = rng.choice(gamma_values, size=remainder_count, replace=True)
            current_gammas.append(unique_remainder)
        
        full_row = np.concatenate(current_gammas)
        rng.shuffle(full_row)
        all_gammas_shuffled[i] = full_row
        
    return jnp.array(all_gammas_shuffled, dtype=jnp.float64)

# ======================================================
# 2. JAX KERNEL (OPTIMIZED AIPT SOLVER)
# ======================================================
@jit
def core_sdf_solver(R_train, R_test, Z_train, Z_test, omega, gamma_vec, z_values, p_list_tuple):
    """
    Optimized Global AIPT Solver.
    """
    T, N = R_train.shape
    
    # --- A. Asset Kernels ---
    K_assets = jnp.dot(R_train, R_train.T) / N
    k_oos_assets = jnp.dot(R_test, R_train.T) / N

    # --- B. Feature Projection (RFF) ---
    proj_train = jnp.dot(Z_train, omega) * gamma_vec
    proj_test  = jnp.dot(Z_test, omega) * gamma_vec
    
    sin_train = jnp.sin(proj_train)
    cos_train = jnp.cos(proj_train)
    S_train = jnp.stack([sin_train, cos_train], axis=-1).reshape(T, -1)
    
    sin_test = jnp.sin(proj_test)
    cos_test = jnp.cos(proj_test)
    S_test = jnp.stack([sin_test, cos_test], axis=-1).reshape(1, -1)
    
    # Standardize Features
    feat_std = jnp.std(S_train, axis=0, ddof=1)
    feat_std = jnp.where(feat_std < 1e-12, 1.0, feat_std)
    S_train = S_train / feat_std
    S_test  = S_test / feat_std

    # --- C. Block Update Loop (GEMM) ---
    K_feat_cur = jnp.zeros((T, T), dtype=S_train.dtype)
    k_feat_cur = jnp.zeros((1, T), dtype=S_train.dtype)
    
    K_total_list = []
    k_total_list = []
    
    prev_p = 0
    for p in p_list_tuple:
        # Slice Feature Block
        block_S_train = S_train[:, prev_p:p]
        block_S_test  = S_test[:, prev_p:p]
        
        # Update Feature Kernel Accumulator
        K_feat_cur = K_feat_cur + jnp.dot(block_S_train, block_S_train.T)
        k_feat_cur = k_feat_cur + jnp.dot(block_S_test, block_S_train.T)
        
        # Combine: K_global = K_assets * (K_features / P)
        inv_p = 1.0 / p
        K_combined = K_assets * (K_feat_cur * inv_p)
        k_combined = k_oos_assets * (k_feat_cur * inv_p)
        
        K_total_list.append(K_combined)
        k_total_list.append(k_combined)
        
        prev_p = p

    # Stack Results -> Shape (Num_P, T, T)
    K_final = jnp.stack(K_total_list)
    k_final = jnp.stack(k_total_list)

    # --- D. Dual Metrics Solver ---
    def solve_single_p(Kmat, kvec):
        U, S_eig, Vh = jnp.linalg.svd(Kmat, full_matrices=False)
        
        ones_rot = jnp.dot(U.T, jnp.ones((T, 1))).flatten()
        k_rot = jnp.dot(kvec, U).flatten()
        
        def solve_z(z):
            denom = S_eig + z
            # 1. OOS SDF Return
            oos_ret = jnp.sum((k_rot * ones_rot) / denom)
            # 2. HJ Distance Squared
            hj_sq = jnp.sum((ones_rot**2 * S_eig) / (denom**2))
            return jnp.array([oos_ret, hj_sq])
            
        return vmap(solve_z)(z_values)

    return vmap(solve_single_p)(K_final, k_final)

# VECTORIZATION SETUP
batch_sdf_solver = vmap(core_sdf_solver, in_axes=(None, None, None, None, 0, 0, None, None))


# ======================================================
# 3. MAIN PIPELINE
# ======================================================
def run_pipeline_aipt(user_config):
    
    CONFIG = {
        'data_path': './data/processed/factors_50_cs_raw_fixed.pkl',
        'output_dir': './results/results_aipt',
        'windows': [120],          
        'max_p': 12000,         
        'num_seeds': 100,       
        'batch_size': 10,       
        'z_list': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        'gamma_list': [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'master_seed': 42,
        'demean_features': False 
    }
    CONFIG.update(user_config)
    
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    print(f"\n============================================================")
    print(f" AIPT ENGINE (SDF) LAUNCHED")
    print(f" DATA: {os.path.basename(CONFIG['data_path'])}")
    print(f" BATCH SIZE: {CONFIG['batch_size']}")
    print(f"============================================================")

    try:
        data_dict = joblib.load(CONFIG['data_path'])
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: File not found at {CONFIG['data_path']}")
        return

    factors = sorted(data_dict.keys())
    # Assumes balanced panel, extracts flat returns (Time x Assets)
    R_panel = np.stack([data_dict[f]['Y'].flatten() for f in factors], axis=1)
    dates = data_dict[factors[0]]['dates']
    mid_idx_global = data_dict[factors[0]]['mid_idx']
    
    # JAX Arrays (Pinned to GPU)
    R_panel_jax = jnp.array(R_panel, dtype=jnp.float64)
    Z_JAX = jnp.array(CONFIG['z_list'], dtype=jnp.float64)

    FEATURE_SETS = ['individual', 'common', 'all']

    for f_set in FEATURE_SETS:
        print(f"\n[CATEGORY]: {f_set.upper()}")
        
        # Define Feature Set Directory
        # e.g. results/results_aipt/common
        feature_set_dir = os.path.join(CONFIG['output_dir'], f_set)
        
        Z_panel = jnp.array(data_dict[factors[0]][f_set], dtype=jnp.float64)
        num_input_features = Z_panel.shape[1]

        for window in CONFIG['windows']:
            
            # Define Window Directory
            # e.g. results/results_aipt/common/120
            window_output_dir = os.path.join(feature_set_dir, str(window))
            if not os.path.exists(window_output_dir): os.makedirs(window_output_dir)

            # Generate Weights
            key = jax.random.PRNGKey(CONFIG['master_seed'])
            ALL_OMEGAS = jax.random.normal(key, (CONFIG['num_seeds'], num_input_features, CONFIG['max_p']//2))
            ALL_GAMMAS = generate_shuffled_gammas(CONFIG['num_seeds'], CONFIG['max_p'], CONFIG['gamma_list'], CONFIG['master_seed'])
            
            p_list_tuple = generate_plist_numpy(window, CONFIG['max_p'])
            p_array_np = np.array(p_list_tuple)

            start_t = max(window, mid_idx_global)
            num_oos = len(dates) - start_t
            num_batches = int(np.ceil(CONFIG['num_seeds'] / CONFIG['batch_size']))

            for b_idx in tqdm(range(num_batches), desc=f"  >> Win {window}"):
                s_start = b_idx * CONFIG['batch_size']
                s_end = min((b_idx+1)*CONFIG['batch_size'], CONFIG['num_seeds'])
                curr_batch = s_end - s_start
                
                batch_omegas = ALL_OMEGAS[s_start:s_end]
                batch_gammas = ALL_GAMMAS[s_start:s_end]

                if curr_batch < CONFIG['batch_size']:
                    pad_len = CONFIG['batch_size'] - curr_batch
                    pad_w = jnp.zeros((pad_len, batch_omegas.shape[1], batch_omegas.shape[2]))
                    batch_omegas_padded = jnp.concatenate([batch_omegas, pad_w], axis=0)
                    pad_g = jnp.zeros((pad_len, batch_gammas.shape[1]))
                    batch_gammas_padded = jnp.concatenate([batch_gammas, pad_g], axis=0)
                else:
                    batch_omegas_padded = batch_omegas
                    batch_gammas_padded = batch_gammas

                batch_results = np.zeros((CONFIG['batch_size'], num_oos, len(p_list_tuple), len(CONFIG['z_list']), 2))

                for t_step, t in enumerate(range(start_t, len(dates))):
                    r_tr = R_panel_jax[t-window : t]
                    r_te = R_panel_jax[t : t+1]
                    z_tr = Z_panel[t-window : t]
                    z_te = Z_panel[t : t+1]
                    
                    res = batch_sdf_solver(
                        r_tr, r_te, z_tr, z_te,
                        batch_omegas_padded, batch_gammas_padded,
                        Z_JAX, p_list_tuple
                    )
                    
                    batch_results[:, t_step, :, :, :] = res

                # SAVE RESULTS (Explicit Feature Set / Window structure)
                for k in range(curr_batch):
                    global_seed = s_start + k
                    seed_res = batch_results[k]
                    
                    # Construction: output/feature_set/window/seed_X.joblib
                    save_path = os.path.join(window_output_dir, f"seed_{global_seed}.joblib")
                    
                    joblib.dump({
                        'sdf_return': seed_res[..., 0],   
                        'HJ_distance': seed_res[..., 1],  
                        'dates': dates[start_t:],
                        'p_list': p_array_np,
                        'window': window,
                        'feature_set': f_set,
                        'z_list': CONFIG['z_list']
                    }, save_path, compress=3)

# ======================================================
# 4. CLI ENTRY POINT
# ======================================================
def get_cli_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/processed/factors_50_cs_raw_fixed.pkl')
    parser.add_argument('--output', type=str, default='./results/results_aipt_factors_50_cs_raw_fixed')
    parser.add_argument('--windows', type=int, nargs='+', default=[120])
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_p', type=int, default=12000)
    args = parser.parse_args()
    return {
        'data_path': args.data, 'output_dir': args.output, 'windows': args.windows,
        'batch_size': args.batch_size, 'max_p': args.max_p
    }

if __name__ == "__main__":
    run_pipeline_aipt(get_cli_config())