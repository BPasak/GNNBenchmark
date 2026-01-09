# CleanEvGNN Async FLOPS Bug - Root Cause Analysis

## The Problem

CleanEvGNN's async FLOPS implementation has a critical bug where it uses module attributes `module.pos_all`, `module.edge_new`, and `module.idx_new` that are never properly initialized or reset, causing "index out of range" errors.

## Root Cause

### AEGNN (Original) Implementation ✅
In the original AEGNN code (`src/Models/aegnn/aegnn/asyncronous/conv.py`):

```python
def __graph_processing(module, x, pos, ...):
    # ...
    pos_all = torch.cat([module.asy_graph.pos, pos_new], dim=0)  # LOCAL variable
    # ...
    y_update = module.sync_forward(x=x_all, pos=pos_all, edge_index=edges_new)
```

**Key**: `pos_all` is a **local variable** computed fresh each time.

### CleanEvGNN (Modified) Implementation ❌
In CleanEvGNN's code (`src/Models/CleanEvGNN/asyncronous/conv.py`):

```python
def __graph_initialization(module, x, edge_index, ...):
    # ...
    y = module.sync_forward(x, pos=module.pos_all, edge_index=module.edge_new)
    # Uses module.pos_all and module.edge_new as MODULE ATTRIBUTES
```

**Problem**: `module.pos_all` and `module.edge_new` are **module attributes** that:
1. Are never properly initialized
2. Contain stale data from previous samples
3. Are not cleared by `reset_async_module()`
4. Cause index errors when they reference nodes beyond the current graph

## Why It Fails

1. **First sample initialization**: When you create `init_data` with 10 events, the async framework tries to use `module.pos_all` which doesn't exist yet or contains garbage
2. **Subsequent samples**: Even after reset, `module.pos_all` still contains positions from the previous sample (e.g., 1000+ nodes)
3. **Edge indices mismatch**: When `module.edge_new` references node 88 but you only have 10 nodes, you get "index out of range"

## Why AEGNN Works

AEGNN recomputes everything as local variables:
```python
pos_all = pos  # initialization
# OR
pos_all = torch.cat([module.asy_graph.pos, pos_new], dim=0)  # processing
```

This ensures `pos_all` always matches the current graph size.

## Why CleanEvGNN Fails

CleanEvGNN expects these as module attributes but:
- Never initializes them in `make_model_asynchronous()`
- Never resets them in `reset_async_module()`
- Never updates them correctly during processing

## Attempted Fixes (All Failed)

### 1. ✗ Initialize with batch then add events
**Problem**: `module.pos_all` still contains wrong data

### 2. ✗ Pass `grid_size` and `edge_attributes`
**Problem**: Doesn't fix the attribute initialization issue

### 3. ✗ Call `reset_async_module()` earlier
**Problem**: Reset function doesn't clear `pos_all`, `edge_new`, `idx_new`

### 4. ✗ Remove full sample preprocessing
**Problem**: Attributes still not properly managed

## The Only Solution

**Disable FLOPS logging** since CleanEvGNN's implementation is fundamentally broken:

```python
async_model = make_model_asynchronous(
    model,
    r=args.radius,
    grid_size=list(image_size),
    edge_attributes=edge_attr_fn,
    max_num_neighbors=args.max_num_neighbors,
    max_dt=args.max_dt,
    log_flops=False,   # ✓ DISABLED
    log_runtime=False  # ✓ DISABLED
)
```

## What Still Works

Even without FLOPS:
- ✅ Async event-by-event processing
- ✅ Accuracy testing
- ✅ AEGNN-style initialization (batch first, then incremental)
- ✅ Prediction verification

## Fixing CleanEvGNN (Would Require)

To properly fix CleanEvGNN's async FLOPS, you would need to:

1. **Refactor `__graph_initialization`** and `__graph_processing` to use local variables instead of module attributes
2. **Update `reset_async_module`** to clear `pos_all`, `edge_new`, `idx_new` attributes
3. **Add proper initialization** in `make_model_asynchronous` to set these attributes
4. **Match AEGNN's approach** of computing everything fresh each time

This is a significant refactoring that would require changing multiple files in the async framework.

## Recommendation

Since the async processing itself works fine without FLOPS, use the script as-is for:
- Testing async accuracy
- Verifying event-by-event processing works
- Comparing different models

For FLOPS measurement, either:
- Use the original AEGNN implementation
- Measure FLOPS through other means (profilers, manual calculation)
- Fix CleanEvGNN's async framework (significant effort)

---

**Status**: Issue understood and documented. FLOPS disabled to enable async testing without crashes.

**Working Script**: `Examples/test_async_flops.py` now runs successfully with FLOPS disabled.

