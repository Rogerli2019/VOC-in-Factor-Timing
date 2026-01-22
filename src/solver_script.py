#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code C: Complexity Engine (Universal) - JAX Optimized
- Dual-Mode: Can be run as a script OR imported as a function.
- Features: JAX Scan, Dynamic Slicing, Feature Demeaning, Vmap batching.
- OPTIMIZED: Uses Block-Matrix updates (Masking) for massive A100 speedup.
"""

import os
import re
import argparse
import time
import joblib
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
from tqdm import tqdm

# Enable 64-bit precision (Financial Standard)
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. JAX KERNEL (THE MATH)
# ==========================================

def generate_plist(trnwin, max_p):
    """Generates the segmented list of P (complexity) steps."""
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
    return jnp.array(final_p[final_p <= max_p], dtype=jnp.int64)

@partial(jit, static_argnames=['demean'])
def core_solver_fast(X_train, Y_train, X_test, omega, z_values, gamma_vec, demean, p_list):
    """
    Optimized RFF Ridge Solver using Masked Block Updates.
    Significantly faster on A100 than the sequential loop.
    """
    T, K = X_train.shape
    
    # 1. Projection
    proj_train = jnp.dot(X_train, omega) * gamma_vec
    proj_test  = jnp.dot(X_test, omega) * gamma_vec
    
    # 2. Random Fourier Features
    sin_train = jnp.sin(proj_train)
    cos_train = jnp.cos(proj_train)
    S_train = jnp.stack([sin_train, cos_train], axis=-1).reshape(T, -1)
    
    sin_test = jnp.sin(proj_test)
    cos_test = jnp.cos(proj_test)
    S_test = jnp.stack([sin_test, cos_test], axis=-1).reshape(1, -1)
    
    # 3. Standardization
    feat_std = jnp.std(S_train, axis=0, ddof=1)
    feat_std = jnp.where(feat_std < 1e-12, 1.0, feat_std)
    
    if demean:
        feat_mean = jnp.mean(S_train, axis=0)
        S_train = (S_train - feat_mean) / feat_std
        S_test  = (S_test - feat_mean) / feat_std
    else:
        S_train = S_train / feat_std
        S_test  = S_test / feat_std
    
    # ==========================================================
    # BLOCK UPDATE ALGORITHM (Masking)
    # ==========================================================
    
    # Pre-calculate global column indices for masking [0, 1, 2, ..., F-1]
    col_indices = jnp.arange(S_train.shape[1])

    # Initial State for Scan: (K_accumulated, k_accumulated, start_index)
    K_init = jnp.zeros((T, T), dtype=S_train.dtype)
    k_init = jnp.zeros((1, T), dtype=S_train.dtype)
    init_state = (K_init, k_init, 0)

    # The Scan Function: Moves from one P-value to the next
    def scan_block(carry, target_p):
        K_acc, k_acc, s_idx = carry
        
        # Create a mask for features in the range [s_idx, target_p)
        # Using floating point mask (1.0 or 0.0) allows fast matrix multiplication
        mask = (col_indices >= s_idx) & (col_indices < target_p)
        mask = mask.astype(S_train.dtype)
        
        # Apply mask (Zero out features not in this block)
        S_masked = S_train * mask[None, :]
        s_test_masked = S_test * mask[None, :]
        
        # Block Matrix Multiplication
        # Delta_K = S_block @ S_block.T
        delta_K = jnp.dot(S_masked, S_masked.T)
        delta_k = jnp.dot(s_test_masked, S_masked.T) 
        
        # Accumulate
        K_new = K_acc + delta_K
        k_new = k_acc + delta_k
        
        # Return state for next step (new accumulators, new start_idx is current target_p)
        # Output to stack is (K_new, k_new)
        return (K_new, k_new, target_p), (K_new, k_new)

    # Execute Scan over the p_list
    _, (K_stack, k_stack) = lax.scan(scan_block, init_state, p_list)
    
    # ==========================================================
    # RIDGE REGRESSION SOLVER
    # ==========================================================
    invT = 1.0 / T
    reg_base = K_stack * invT 
    k_vecs   = k_stack * invT

    def solve_single_p(Kmat, kvec):
        U, S, Vh = jnp.linalg.svd(Kmat, full_matrices=False, compute_uv=True)
        y_rot = jnp.dot(U.T, Y_train)        
        k_rot = jnp.dot(kvec, U)              
        
        def solve_z(z):
            damped_inv_S = 1.0 / (S + z)
            alpha_rot = y_rot * damped_inv_S.reshape(-1, 1)
            pred = jnp.dot(k_rot, alpha_rot)[0, 0]
            
            term = jnp.sum((alpha_rot.flatten() ** 2) * S)
            norm_sq = term * invT 
            return jnp.array([pred, jnp.sqrt(norm_sq)], dtype=Kmat.dtype)
            
        return vmap(solve_z)(z_values)

    return vmap(solve_single_p)(reg_base, k_vecs)

# Vectorize over seeds
# ARGUMENTS MAPPING:
# 0: X_train (None - Shared)
# 1: Y_train (None - Shared)
# 2: X_test (None - Shared)
# 3: omega (0 - Batched)
# 4: z_values (None - Shared)
# 5: gamma_vec (0 - Batched) <--- FIXED THIS (Was None)
# 6: demean (None - Shared)
# 7: p_list (None - Shared)
batch_core_solver = vmap(core_solver_fast, in_axes=(None, None, None, 0, None, 0, None, None))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def generate_shuffled_gammas(num_seeds, max_p, gamma_list, seed):
    """Distributes gamma values (scales) across features and shuffles them."""
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

# ==========================================
# 3. RUN PIPELINE
# ==========================================

def run_pipeline(config):
    """
    Main execution function.
    """
    defaults = {
        'factors': 'auto',
        'cases': ['individual', 'common', 'all'],
        'max_p': 12000,
        'num_seeds': 100,
        'batch_size': 100, # Optimized default for A100
        'z_list': [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        'master_seed': 42,
        'gamma_list': [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'demean_features': False
    }
    for k, v in defaults.items():
        if k not in config: config[k] = v

    # Path Setup
    if not os.path.exists(config['data_path']):
        print(f"CRITICAL ERROR: File not found at {config['data_path']}")
        return

    # Use the output directory exactly as passed (prevents double nesting)
    base_output_dir = config['output_dir']
    if not os.path.exists(base_output_dir): 
        os.makedirs(base_output_dir)
        
    dataset_name_log = os.path.splitext(os.path.basename(config['data_path']))[0]

    print(f"\n{'='*60}")
    print(f" LOADING DATASET: {dataset_name_log}")
    print(f" SAVING TO: {base_output_dir}")
    print(f" A100 OPTIMIZED MODE: ON (Batch Size: {config['batch_size']})")
    print(f"{'='*60}")

    data_dict = joblib.load(config['data_path'])
    Z_JAX = jnp.array(config['z_list'], dtype=jnp.float64)
    
    # Factor Detection
    if config['factors'] == 'auto':
        all_keys = [k for k in data_dict.keys() if isinstance(k, str)]
        is_pc_dataset = any(k.startswith('PC') and k[2:].isdigit() for k in all_keys)
        if is_pc_dataset:
            try: all_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
            except: all_keys.sort()
        else: all_keys.sort()
        factors_to_process = all_keys
    else: factors_to_process = config['factors']

    # LOOP
    for factor_name in factors_to_process:
        print(f"\n[Processing Factor]: {factor_name}")
        if factor_name not in data_dict: continue
        
        factor_data = data_dict[factor_name]
        raw_Y = factor_data['Y']
        dates = factor_data['dates']
        mid_idx = factor_data['mid_idx'] 
        if mid_idx >= len(dates): continue

        for case in config['cases']:
            print(f"  >> Case: {case}")
            if case == 'individual': raw_X = factor_data['individual']
            elif case == 'common':   raw_X = factor_data['common']
            else:                    raw_X = factor_data['all']
            
            X_jax = jnp.array(raw_X, dtype=jnp.float64)
            Y_jax = jnp.array(raw_Y, dtype=jnp.float64)
            
            key_master = jax.random.PRNGKey(config['master_seed'])
            ALL_OMEGAS = jax.random.normal(key_master, (config['num_seeds'], X_jax.shape[1], config['max_p'] // 2), dtype=jnp.float64)
            ALL_GAMMAS = generate_shuffled_gammas(config['num_seeds'], config['max_p'], config['gamma_list'], config['master_seed'])

            for window in config['windows']:
                save_dir = os.path.join(base_output_dir, factor_name, case, str(window))
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                if mid_idx < window: continue

                # Generate P-List and convert to JAX Array
                p_list_np = generate_plist(window, config['max_p'])
                p_list_jax = jnp.array(p_list_np, dtype=jnp.int32)

                # Use int64 for scan indices (x64 requirement)
                scan_indices = jnp.arange(mid_idx, len(Y_jax), dtype=jnp.int64)
                
                num_batches = int(np.ceil(config['num_seeds'] / config['batch_size']))

                for b_idx in tqdm(range(num_batches), desc=f"Seeds (Win={window})"):
                    start_seed = b_idx * config['batch_size']
                    end_seed = min((b_idx + 1) * config['batch_size'], config['num_seeds'])
                    curr_batch_size = end_seed - start_seed
                    
                    batch_omegas = ALL_OMEGAS[start_seed:end_seed]
                    batch_gammas = ALL_GAMMAS[start_seed:end_seed]
                    
                    # Padding for JIT stability
                    if curr_batch_size < config['batch_size']:
                        pad_len = config['batch_size'] - curr_batch_size
                        pad_w = jnp.zeros((pad_len, batch_omegas.shape[1], batch_omegas.shape[2]))
                        batch_omegas_padded = jnp.concatenate([batch_omegas, pad_w], axis=0)
                        pad_g = jnp.zeros((pad_len, batch_gammas.shape[1]))
                        batch_gammas_padded = jnp.concatenate([batch_gammas, pad_g], axis=0)
                    else:
                        batch_omegas_padded = batch_omegas
                        batch_gammas_padded = batch_gammas

                    # The Scan Body Function
                    def scan_body_fun(carry, t):
                        start_idx = t - window
                        X_train = lax.dynamic_slice(X_jax, (start_idx, 0), (window, X_jax.shape[1]))
                        Y_train = lax.dynamic_slice(Y_jax, (start_idx, 0), (window, 1))
                        X_test  = lax.dynamic_slice(X_jax, (t, 0), (1, X_jax.shape[1]))
                        
                        # NEW SOLVER CALL
                        res = batch_core_solver(
                            X_train, Y_train, X_test, 
                            batch_omegas_padded, Z_JAX, batch_gammas_padded,
                            config['demean_features'],
                            p_list_jax # <--- Passing the P-list map
                        )
                        # SPLIT RESULT into Preds and Norms
                        return None, (res[..., 0], res[..., 1])

                    # Execute Scan
                    _, (batch_preds_stacked, batch_norms_stacked) = jax.lax.scan(
                        scan_body_fun, None, scan_indices
                    )
                    
                    # Save results
                    batch_preds_np = np.array(jnp.swapaxes(batch_preds_stacked, 0, 1))
                    batch_norms_np = np.array(jnp.swapaxes(batch_norms_stacked, 0, 1))

                    for k in range(curr_batch_size):
                        global_seed = start_seed + k
                        filename = os.path.join(save_dir, f"seed_{global_seed}.joblib")
                        joblib.dump({
                            'preds': batch_preds_np[k], 'norms': batch_norms_np[k],
                            'seed': global_seed, 'oos_dates': dates[mid_idx:] 
                        }, filename, compress=3)

# ==========================================
# 4. CLI ENTRY POINT
# ==========================================
def get_cli_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/processed/factors_50_raw.pkl')
    parser.add_argument('--output', type=str, default='./results/results_pc_multiscale_paper_exact_fixed_full')
    parser.add_argument('--windows', type=int, nargs='+', default=[12, 60, 120])
    parser.add_argument('--batch_size', type=int, default=100) # Default for A100
    parser.add_argument('--max_p', type=int, default=12000)
    parser.add_argument('--demean', action='store_true')
    args = parser.parse_args()
    return {
        'data_path': args.data, 'output_dir': args.output, 'windows': args.windows,
        'batch_size': args.batch_size, 'max_p': args.max_p, 'demean_features': args.demean
    }

if __name__ == "__main__":
    run_pipeline(get_cli_config())