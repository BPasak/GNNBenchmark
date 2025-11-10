import time
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

@torch.no_grad()
def evaluate_async_backbone_plus_head(
    backbone,            # async-wrapped GraphRes with model.fc = nn.Identity()
    head,                # original Linear, e.g. 512->2 (we'll adapt if needed)
    dataset, device,
    *, init_events=5000, stream_events=1,
    max_sequences=10, verbose=True
):
    backbone.eval(); head.eval()
    total = hits = 0
    nseq = min(max_sequences, len(dataset))

    # small helper: build init graph fast by subgraphing first N0 nodes
    def build_init(sample, N0):
        mask = torch.zeros(sample.x.size(0), dtype=torch.bool, device=device)
        mask[:N0] = True
        ei0, ea0 = subgraph(mask, sample.edge_index, edge_attr=sample.edge_attr, relabel_nodes=True)
        return Data(
            x=sample.x[:N0], pos=sample.pos[:N0],
            edge_index=ei0.long(), edge_attr=ea0,
            batch=torch.zeros(N0, dtype=torch.long, device=device)
        )

    # We’ll keep an adapter head if feature dims don’t match
    adapter_head = None

    for si in range(nseq):
        t0 = time.time()
        sample = dataset[si].to(device)
        N = int(sample.x.size(0))
        if N < 2:
            continue
        N0 = min(init_events, max(2, N - 1))

        # reset async cache per sequence if available
        if hasattr(backbone, "_async_impl") and hasattr(backbone._async_impl, "reset_cache"):
            backbone._async_impl.reset_cache()

        # 1) async init
        init_graph = build_init(sample, N0)
        feats = backbone(init_graph)        # <- whatever async backbone returns (likely [B, 32])

        # Detect feature size and adapt the head if needed
        if feats.dim() != 2:
            # make it [B, F]
            feats = feats.view(feats.size(0), -1)

        F = feats.size(1)
        if adapter_head is None:
            if F != head.in_features:
                if verbose:
                    print(f"[async] backbone feature dim = {F}, but head expects {head.in_features}. "
                          f"Using adapter head ({F}→{head.out_features}).")
                adapter_head = nn.Linear(F, head.out_features).to(device)
                adapter_head.load_state_dict(head.state_dict(), strict=False)  # keep output bias/weights shape
            else:
                adapter_head = head  # perfectly matches

        logits = adapter_head(feats)

        # 2) stream k events (single-node inputs)
        last_feats = None
        end = min(N, N0 + stream_events)
        for j in range(N0, end):
            ev = Data(
                x=sample.x[j:j+1], pos=sample.pos[j:j+1],
                batch=torch.zeros(1, dtype=torch.long, device=device)
            )
            last_feats = backbone(ev)
        if last_feats is not None:
            if last_feats.dim() != 2:
                last_feats = last_feats.view(last_feats.size(0), -1)
            logits = adapter_head(last_feats)

        y = sample.y.view(-1)
        pred = logits.argmax(-1)
        hits += int((pred == y).sum().item()); total += y.numel()

        if verbose:
            shape_show = tuple((last_feats if last_feats is not None else feats).shape)
            print(f"[seq {si+1}/{nseq}] init={N0}, stream={end-N0} | feat={shape_show} | "
                  f"pred={pred.item()} y={y.item()} | {time.time()-t0:.2f}s")

    acc = hits / max(total, 1)
    return acc
