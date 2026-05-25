import numpy as np
import pickle
import time

from .train_utils import *
from .model_registry import get_model_spec, get_model_spec_needs_seq_len
from .train_dataclasses import *

def load_RL_data(roots: dict, split_cfg: SplitConfig) -> DataArrays:
    iq, ant_pos, angles, positions, keys = load_iq_dataset(roots["IQ"])
    
    angles_hat = get_baseline_data(roots["BASELINE"])

    T = split_cfg.drop_N_first

    data = DataArrays(
        iq=iq[T:],
        angles_hat=angles_hat,
        positions=positions[T:],
        angles=angles[T:],
    )

    return filter_by_elevation(data, split_cfg.max_elevation)

def filter_by_elevation(data: DataArrays, max_elevation: float) -> DataArrays:
    mask = data.angles[:, 1] <= np.deg2rad(max_elevation)

    return DataArrays(
        iq=data.iq[mask],
        angles_hat=data.angles_hat,
        positions=data.positions[mask],
        angles=data.angles[mask],
    )

def filter_by_azimuth(data: DataArrays, max_az: float, min_az:float) -> DataArrays:
    mask = (
        (data.angles[:, 0] >= np.deg2rad(min_az)) &
        (data.angles[:, 0] <= np.deg2rad(max_az))
    )

    return DataArrays(
        iq=data.iq[mask],
        angles_hat=data.angles_hat,
        positions=data.positions[mask],
        angles=data.angles[mask],
    )

def create_dataset(
    model_cfg:ModelConfig,
    split_cfg:SplitConfig,
    indices:SplitIndices,
    data:DataArrays,
    shuffle:bool = True
):

    spec = get_model_spec(model_cfg)
    split = data.normalized_split(indices)

    train_dataset = spec.build_dataset(split.train, True, model_cfg, shuffle)
    val_dataset = spec.build_dataset(split.val, False, model_cfg, shuffle)
    test_dataset = spec.build_dataset(split.test, False, model_cfg, False)

    return train_dataset, val_dataset, test_dataset

def create_model(model_cfg:ModelConfig):
    spec = get_model_spec(model_cfg)
    model = spec.build_model(model_cfg)
    return model

def create_synth_dataset(model_cfg:ModelConfig, split_cfg:SplitConfig, synth_data:DataArrays):

    iq_norm = normalize_full(synth_data.iq)
    pos_norm = normalize_full(synth_data.positions)
    angles = synth_data.angles
    angles_hat = synth_data.angles_hat

    size = iq_norm.shape[0]
    t = int(size * 0.9)

    train_data = DataArrays(iq_norm[:t], angles_hat[:t], pos_norm[:t], angles[:t])
    val_data = DataArrays(iq_norm[t:], angles_hat[t:], pos_norm[t:], angles[t:])

    spec = get_model_spec(model_cfg)
    train_dataset = spec.build_dataset(train_data, True, model_cfg, True)
    val_dataset = spec.build_dataset(val_data, False, model_cfg, True)

    return train_dataset, val_dataset

def make_walk_forward_splits(
    n_samples: int,
    split_cfg: SplitConfig,
    model_cfg: ModelConfig,
) -> list[SplitIndices]:
    
    cfg = split_cfg
    spec = get_model_spec(model_cfg)
    seq_len = model_cfg.seq_len

    step_size = cfg.step_size or cfg.test_size
    n_context = seq_len - 1 if spec.needs_seq_len else 0

    splits: list[SplitIndices] = []
    min_train_end = cfg.val_size + seq_len
    train_end = max(cfg.train_size, min_train_end)

    while train_end < n_samples:
        test_start = train_end
        test_end = test_start + cfg.test_size

        if test_end > n_samples:
            if not cfg.allow_partial_last:
                break
            test_end = n_samples

        train_start = (
            0
            if cfg.max_train_size is None
            else max(0, train_end - cfg.max_train_size)
        )

        full_train = np.arange(train_start, train_end)

        if len(full_train) <= cfg.val_size:
            break

        train_idx = full_train[:-cfg.val_size]
        raw_val_idx = full_train[-cfg.val_size:]
        raw_test_idx = np.arange(test_start, test_end)

        if n_context > 0:
            val_context = train_idx[-n_context:]
            test_context = raw_val_idx[-n_context:]

            val_idx = np.concatenate([val_context, raw_val_idx])
            test_idx = np.concatenate([test_context, raw_test_idx])
        else:
            val_idx = raw_val_idx
            test_idx = raw_test_idx

        split = SplitIndices(
            train=train_idx,
            val=val_idx,
            test=test_idx,
        )

        if not split.is_valid():
            break

        splits.append(split)
        train_end += step_size

    return splits

def make_loader(dataset, batch_size=64, shuffle=False, seed=1111):
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model) -> None:
    print(
        f"Model created | "
        f"trainable parameters: {count_trainable_params(model):,}"
    )

def print_fold_header(fold_id: int, indices: SplitIndices) -> None:
    print()
    print("=" * 80)
    print(
        f"Fold {fold_id} | "
        f"train=[{indices.train[0]}, {indices.train[-1]}] | "
        f"val=[{indices.val[0]}, {indices.val[-1]}] | "
        f"test=[{indices.test[0]}, {indices.test[-1]}]"
    )
    print("=" * 80)

def with_sequence_context(indices: SplitIndices, seq_len: int) -> SplitIndices:
    context = indices.val[-(seq_len - 1):]

    return SplitIndices(
        train=indices.train,
        val=indices.val,
        test=np.concatenate([context, indices.test]),
    )

def evaluate_fold(model, test_loader, device, fold_id: int, valid_test_idx: np.ndarray) -> FoldResult:
    test_loss, gt_rad, pred_rad = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    gtaz, gtel, predaz, predel = angles_rad_to_deg(gt_rad, pred_rad)

    rmse_az, rmse_el = compute_angle_rmse_deg(
        gtaz=gtaz,
        gtel=gtel,
        predaz=predaz,
        predel=predel,
        elevation_bias_deg=0.0,
    )

    return FoldResult(
        fold_id=fold_id,
        test_loss=test_loss,
        rmse_az=rmse_az,
        rmse_el=rmse_el,
        gt_rad=gt_rad,
        pred_rad=pred_rad,
        test_idx=valid_test_idx.copy(),
    )

def get_valid_test_idx(indices: SplitIndices, model_cfg) -> np.ndarray:
    if get_model_spec_needs_seq_len(model_cfg):
        return indices.test[model_cfg.seq_len - 1:]

    return indices.test

def aggregate_fold_results(results: list[FoldResult]) -> WalkForwardResult:
    gt_rad = np.concatenate([r.gt_rad for r in results], axis=0)
    pred_rad = np.concatenate([r.pred_rad for r in results], axis=0)
    test_idx = np.concatenate([r.test_idx for r in results], axis=0)
    fold_ids = np.concatenate(
        [np.full(len(r.test_idx), r.fold_id) for r in results],
        axis=0,
    )

    gtaz, gtel, predaz, predel = angles_rad_to_deg(gt_rad, pred_rad)

    rmse_az, rmse_el = compute_angle_rmse_deg(
        gtaz=gtaz,
        gtel=gtel,
        predaz=predaz,
        predel=predel,
        elevation_bias_deg=0.0,
    )

    return WalkForwardResult(
        gt_rad=gt_rad,
        pred_rad=pred_rad,
        test_idx=test_idx,
        fold_ids=fold_ids,
        rmse_az=rmse_az,
        rmse_el=rmse_el,
    )

def summarize_walkforward_result(result: WalkForwardResult, split_cfg) -> None:
    gt_deg = np.rad2deg(result.gt_rad)
    pred_deg = np.rad2deg(result.pred_rad)

    plot_walkforward_oos(
        t=result.test_idx,
        gtaz=gt_deg[:, 0],
        gtel=gt_deg[:, 1],
        predaz=pred_deg[:, 0],
        predel=pred_deg[:, 1],
        fold_ids=result.fold_ids,
        elevation_bias_deg=0.0,
        error_window=split_cfg.test_size,
    )

    print()
    print("Overall walk-forward metrics")
    print(f"Azimuth RMSE   : {result.rmse_az:.3f} deg")
    print(f"Elevation RMSE : {result.rmse_el:.3f} deg")

def print_fold_result(result: FoldResult) -> None:
    print(
        f"Fold {result.fold_id} | "
        f"test_loss={result.test_loss:.6f} | "
        f"rmse_az={result.rmse_az:.3f} deg | "
        f"rmse_el={result.rmse_el:.3f} deg"
    )

def train_walk_forward(device, roots, split_configs:SplitConfig, model_configs:ModelConfig):

    # --- Real Life data ---
    data = load_RL_data(roots, split_configs)

    # --- Synth data ---
    if split_configs.enable_synth:
        synth_train, synth_val = None, None

        if not roots["SYNTH"]:
            raise ValueError("Synthetic data is enabled, but roots['SYNTH'] is empty.")
        
        for path in roots["SYNTH"]:
            with open(path, "rb") as f:
                synth = pickle.load(f)
                synth = DataArrays(synth["IQ"], synth["ANG_HAT"], synth["POS"], synth["ANG"])

            if synth_train is None or synth_val is None:
                synth_train, synth_val = create_synth_dataset(model_configs, split_configs, synth)
            else:
                new_train, new_val = create_synth_dataset(model_configs, split_configs, synth)
                synth_train.concat(new_train, shuffle=True)
                synth_val.concat(new_val, shuffle=True)      

        synth_train_loader = make_loader(synth_train, batch_size=model_configs.synth_batch_size, shuffle=True)
        synth_val_loader = make_loader(synth_val, batch_size=model_configs.synth_batch_size, shuffle=True)


    # --- Prepare Splits ---
    splits = make_walk_forward_splits(n_samples=len(data), split_cfg=split_configs, model_cfg=model_configs)
    model = None
    fold_results: list[FoldResult] = []
    online_train_time_s, online_n_samples = 0.0, 0

    # Main loop
    for fold_id, indices in enumerate(splits, start=1):
        print_fold_header(fold_id, indices)

        should_create_model = model is None or not split_configs.train_online
        should_train_on_synth = fold_id == 1 and split_configs.enable_synth
        should_train_online = fold_id > 1 and split_configs.train_on_previous_data

        train_dataset, val_dataset, test_dataset = create_dataset(model_configs, split_configs,indices,data)
        train_loader = make_loader(train_dataset, batch_size=model_configs.batch_size)
        val_loader = make_loader(val_dataset, batch_size=model_configs.batch_size)
        test_loader = make_loader(test_dataset, batch_size=model_configs.batch_size)
        valid_test_idx = get_valid_test_idx(indices, model_configs)

        if should_create_model:
            model = create_model(model_configs)
            print_model_summary(model)

        if should_train_on_synth:
            train_model(
                model=model,
                train_loader=synth_train_loader,
                val_loader=synth_val_loader,
                epochs=model_configs.synth_epochs,
                lr=model_configs.lr,
                device=device,
            )

        if should_train_online:

            if split_configs.should_freeze:
                model.freeze_cnn()

            t0 = time.perf_counter()
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=model_configs.epochs,
                lr=model_configs.lr,
                device=device,
            )
            online_train_time_s += time.perf_counter() - t0
            online_n_samples += len(train_dataset)

        fold_result = evaluate_fold(
            model=model,
            test_loader=test_loader,
            device=device,
            fold_id=fold_id,
            valid_test_idx=valid_test_idx,
        )

        print_fold_result(fold_result)
        fold_results.append(fold_result)

    if online_n_samples > 0:
        train_time_per_sample_ms = 1000.0 * online_train_time_s / online_n_samples
        train_time_per_sample_per_epoch_ms = (
            1000.0 * online_train_time_s / (online_n_samples * model_configs.epochs)
        )

        print(f"Online training time total      : {online_train_time_s:.4f} s")
        print(f"Online training time per sample : {train_time_per_sample_ms:.4f} ms")
        print(f"Online training time per sample per epoch : {train_time_per_sample_per_epoch_ms:.4f} ms")
    else:
        print("Online training skipped; train_on_previous_data=False")

    walkforward_result = aggregate_fold_results(fold_results)
    summarize_walkforward_result(walkforward_result, split_configs)
