# Library Audit Report
**Date:** 2025-11-05  
**Scope:** Comparison against `tutorials_orig/Solution_3.1_trace-wise_denoise_synth-test-data.ipynb`

---

## Executive Summary

The library implementation is **more correct** than the reference tutorial in several key areas. However, **one critical bug** was found in the evaluation loop (missing progress bar update), and several design improvements are recommended.

---

## Critical Issues Found

### 1. **BUG: Missing progress bar update in `n2v_evaluate`**
**Location:** `src/blindspot_denoise/training_loop.py:172`  
**Severity:** Medium (functional correctness, not algorithmic)

**Issue:**  
The `n2v_train` function correctly calls `pbar.update(1)` when a persistent progress bar is passed, but `n2v_evaluate` is missing this update call. This causes the validation progress bar to freeze when using persistent bars across epochs.

**Fix Required:**
```python
# After line 172 in n2v_evaluate, add:
if pbar is not None:
    pbar.update(1)
```

---

## Bugs in Original Tutorial (Fixed in Library)

### 1. **CRITICAL: `model.train()` in evaluation function**
**Location:** Tutorial cell 34, line 28  
**Tutorial Code:**
```python
def n2v_evaluate(...):
    model.train()  # <-- WRONG! Should be model.eval()
```

**Library Fix:** ✅  
`src/blindspot_denoise/training_loop.py:137` correctly uses `model.eval()`

**Impact:** Using `model.train()` during validation enables dropout and batch norm training mode, causing non-deterministic validation metrics and incorrect evaluation.

---

### 2. **Unnecessary `optimizer.zero_grad()` in evaluation**
**Location:** Tutorial cell 34, line 36  
**Tutorial Code:**
```python
def n2v_evaluate(...):
    ...
    for dl in tqdm(data_loader):
        X, y, mask = dl[0].to(device), dl[1].to(device), dl[2].to(device)
        optimizer.zero_grad()  # <-- UNNECESSARY in eval, wastes cycles
```

**Library Fix:** ✅  
No `optimizer.zero_grad()` in `n2v_evaluate` since gradients are not computed.

**Impact:** Minor performance overhead.

---

### 3. **Incorrect RMSE computation scope**
**Location:** Tutorial cell 32, line 54  
**Tutorial Code:**
```python
accuracy += np.sqrt(np.mean((y.cpu().numpy().ravel() - ypred.ravel())**2))
```

**Analysis:**  
- Tutorial computes RMSE over **entire prediction** (including uncorrupted regions)
- Library computes RMSE only over **masked (corrupted) regions**: `(yprob - y) * (1 - mask)`

**Which is correct?**  
The library approach is **more aligned with blind-spot methodology** since:
- Loss is only computed on masked pixels: `criterion(yprob * (1 - mask), y * (1 - mask))`
- RMSE should measure the same regions as the loss for meaningful comparison
- Uncorrupted regions are not trained on, so including them in RMSE dilutes the metric

**Recommendation:** Keep library implementation. The tutorial's full-image RMSE is misleading.

---

### 4. **Deprecated weight initialization**
**Location:** `tutorials_orig/tutorial_utils.py:267-268`  
**Tutorial Code:**
```python
nn.init.xavier_normal(m.weight)  # Deprecated
nn.init.constant(m.bias, 0)      # Deprecated
```

**Library Fix:** ✅  
`src/blindspot_denoise/utils.py:20-21` uses modern API:
```python
nn.init.xavier_normal_(m.weight)
nn.init.constant_(m.bias, 0)
```

---

## Library Improvements Over Tutorial

### 1. **Memory Efficiency**
- **Tutorial:** Loads all patches into RAM, then creates TensorDataset
  ```python
  train_X = np.expand_dims(corrupted_patches[:n_training], axis=1)  # Full copy in RAM
  ```
- **Library:** Uses streaming dataset with on-the-fly patch extraction from memmap
  ```python
  clean_shot = np.asarray(self._d[shot_idx])  # View from memmap, minimal RAM
  ```

**Impact:** Library can handle arbitrarily large datasets without RAM overflow.

---

### 2. **Mixed Precision Training (AMP)**
- **Tutorial:** No AMP support
- **Library:** Full AMP support via `torch.amp.autocast` and `GradScaler`

**Impact:** 1.5-2x faster training on modern GPUs with minimal accuracy loss.

---

### 3. **Non-blocking Transfers**
- **Tutorial:** Blocking GPU transfers: `.to(device)`
- **Library:** Non-blocking: `.to(device, non_blocking=True)`

**Impact:** Overlaps CPU→GPU data transfer with computation for better throughput.

---

### 4. **RMSE Computed in PyTorch**
- **Tutorial:** Converts to numpy for RMSE:
  ```python
  ypred = yprob.detach().cpu().numpy().astype(float)
  accuracy += np.sqrt(np.mean((y.cpu().numpy().ravel() - ypred.ravel())**2))
  ```
- **Library:** Stays in torch:
  ```python
  rmse = torch.sqrt(torch.mean(((yprob.detach() - y) * (1 - mask)) ** 2)).item()
  ```

**Impact:** Avoids CPU↔GPU sync overhead.

---

## Potential Issues (Minor)

### 1. **RandomState vs Generator Consistency**
**Location:** `src/blindspot_denoise/utils.py:113`  
**Code:**
```python
self._rng = np.random.RandomState(seed) if seed is not None else np.random
```

**Issue:** When `seed=None`, falls back to global `np.random` module, which may be affected by external code. For test set (`seed=None` in line 188), this could reduce reproducibility if global seed changes.

**Recommendation:**  
Use a fresh `RandomState` for test set with a different fixed seed:
```python
# In make_streaming_data_loader, line 188:
seed=seed + 1 if seed is not None else None  # Offset test seed
```

---

### 2. **Checkpoint Compatibility**
**Location:** `src/blindspot_denoise/infer.py:28-54`  
**Status:** ✅ Already handled with fallback logic

The library correctly handles PyTorch 2.6's `weights_only=True` default with safe fallbacks.

---

### 3. **Padding Edge Cases**
**Location:** `src/blindspot_denoise/infer.py:83-90`  
**Code:**
```python
stride = 2 ** int(net_levels)
pad_h = (-H) % stride
pad_w = (-W) % stride
```

**Analysis:** Padding logic is correct. Uses `mode='edge'` to avoid artifacts.

**Edge Case:** If `H` or `W` is already divisible by `stride`, `pad_h/pad_w = 0`, which is handled correctly.

---

## Memory Leak Analysis

### ✅ No memory leaks detected

**Checked:**
1. **Gradient accumulation:** `optimizer.zero_grad()` called correctly in training loop
2. **Detach in metrics:** `yprob.detach()` used before RMSE computation
3. **Context managers:** `torch.no_grad()` used in evaluation
4. **Persistent workers:** Correctly set only when `num_workers > 0`
5. **Memmap views:** Use `np.asarray()` to create views, not copies

---

## Recommendations

### High Priority
1. **Fix missing `pbar.update(1)` in `n2v_evaluate`** (See Critical Issue #1)

### Medium Priority
2. Consider using deterministic test set RNG (offset seed) for better reproducibility

### Low Priority
3. Add gradient clipping for robustness (optional):
   ```python
   torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
   ```

---

## Conclusion

The library implementation is **algorithmically sound** and **more efficient** than the reference tutorial. The tutorial itself contains several bugs (most notably `model.train()` in evaluation) that have been correctly fixed in the library.

**Action Required:** Fix the missing progress bar update in `n2v_evaluate`.
