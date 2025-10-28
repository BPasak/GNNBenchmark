import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from GCNEncoder import GCNEncoder
from SAGEEncoder import SAGEEncoder
from FullModel_GCN_SAGE import FullModel, ClassifierHead
from train_ncars_from_events import train_one_epoch, evaluate, take_first
#run with:
#python .\Main.py --root "C:\Users\hanne\Documents\Hannes\Uni\Maastricht\Project\GNNBenchmark\src\Models\AEGNN\data\ncars" --sensor_wh 304,240 --model sage --k 16 --max_events 5000 --epochs 30 --batch_size 64 --lr 1e-3 --wd 0  --limit_per_split 20 --async_eval

from ProcessNCars import NCarsEventsGraphDataset





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Folder containing train/, val/, test/")
    ap.add_argument("--events_name", type=str, default=None,
                    help="Exact events filename. If None, first *.txt not named is_car.txt is used.")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_events", type=int, default=5000)
    ap.add_argument("--sensor_wh", type=str, default=None, help="e.g. '304,240'. If None, per-seq min/max.")
    ap.add_argument("--model", choices=["sage", "gcn"], default="sage")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--async_eval", action="store_true",
                    help="Wrap the ENCODER with AEGNNAsyncWrapper for evaluation/inference.")
    ap.add_argument("--limit_per_split", type=int, default=0,
                    help="Use only the first N sequences for each split (0 = all).")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_wh = None
    if args.sensor_wh:
        w, h = [int(s.strip()) for s in args.sensor_wh.split(",")]
        sensor_wh = (w, h)

    # Split dirs
    root = Path(args.root)
    train_dir = root / "training"
    val_dir   = root / "validation"
    test_dir  = root / "test"
    for d in [train_dir, val_dir, test_dir]:
        assert d.exists(), f"Missing split directory: {d}"

    # Datasets
    precompute_gcn = (args.model == "gcn")
    train_ds = NCarsEventsGraphDataset(str(train_dir), args.events_name, args.k, args.max_events,
                                       sensor_wh, precompute_gcn, cache=True)
    val_ds   = NCarsEventsGraphDataset(str(val_dir),   args.events_name, args.k, args.max_events,
                                       sensor_wh, precompute_gcn, cache=True)
    test_ds  = NCarsEventsGraphDataset(str(test_dir),  args.events_name, args.k, args.max_events,
                                       sensor_wh, precompute_gcn, cache=True)

    # only first 20 for train and val as requested
    limit = args.limit_per_split or 0
    if limit:
        train_ds = take_first(train_ds, limit)
        val_ds = take_first(val_ds, limit)
        # leave test full; uncomment next line if you also want to cap test
        test_ds  = take_first(test_ds, limit)

    # Infer dims
    sample: Data = train_ds[0]
    in_ch = sample.x.size(-1)
    num_classes = int(max(sample.y.max().item(), 1) + 1)

    # Build encoder + head
    if args.model == "sage":
        encoder = SAGEEncoder(in_ch, hid=args.hidden)
    else:
        encoder = GCNEncoder(in_ch, hid=args.hidden)
    head = ClassifierHead(hid=args.hidden, num_classes=num_classes, dropout=args.dropout, bias=True)

    model = FullModel(encoder, head).to(device)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Optim / loss
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    # ===========
    # Train (sync)
    # ===========
    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt, crit)
        va_loss, va_acc = evaluate(model, val_loader, device, crit)
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # ===========================
    # Optional async evaluation
    # ===========================
    if args.async_eval:
        # Wrap ONLY the encoder; keep the head outside (bias allowed).
        try:
            # local import to avoid hard dependency when not used
            from .AEGNNwrapper import AEGNNAsyncWrapper  # if running as module
        except Exception:
            from AEGNNwrapper import AEGNNAsyncWrapper   # if running directly
        base_encoder = model.encoder
        enc_async = AEGNNAsyncWrapper(base_encoder)  # no kwargs; we learned to avoid unsupported args
        print("[AEGNN] is_async=", enc_async.is_async, "| why_not=", enc_async.why_not_async)
        model.encoder = enc_async.to(device)  # swap in-place

    # Final test
    te_loss, te_acc = evaluate(model, test_loader, device, crit)
    print(f"Test  | loss {te_loss:.4f} acc {te_acc:.3f}")


if __name__ == "__main__":
    main()