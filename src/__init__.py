from .dense_attention import dense_attention, masked_softmax, causal_mask
from .patterns import (
    build_block_mask,
    expand_block_mask,
    density,
    sliding_window_block_mask,
    bigbird_block_mask,
    dilated_block_mask,
    dense_block_mask,
    PATTERNS,
)
from .sparse_attention import (
    SparsePlan,
    build_plan,
    get_plan,
    plan_token_mask,
    block_sparse_attention,
    SparseSelfAttention,
    explain_nan_sources,
)
