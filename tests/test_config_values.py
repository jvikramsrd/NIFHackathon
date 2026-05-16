def test_device_is_torch_device():
    import torch
    import config as CFG
    assert isinstance(CFG.DEVICE, torch.device)

def test_road_class_weight_is_5_0():
    import config as CFG
    assert CFG.STAGE1["class_weights"][2] == 5.0

def test_stage1_uses_manet_mit_b4():
    import config as CFG
    assert CFG.STAGE1["arch"] == "MAnet"
    assert CFG.STAGE1["encoder"] == "mit_b4"

def test_arcface_m_is_0_55():
    import config as CFG
    assert CFG.STAGE2A["arcface_m"] == 0.55

def test_sahi_overlap_is_0_45():
    # Bumped from 0.40 to 0.45 to improve small-object recall on
    # transformers (~30-60 px) and wells (~15-30 px).
    import config as CFG
    assert CFG.STAGE2B["sahi_overlap_ratio"] == 0.45


def test_stage1_batch_size_4():
    # batch_size=4 with grad_accum=8 keeps effective batch 32 at half the
    # activation peak — needed because MiT encoders in smp do not expose
    # set_grad_checkpointing.
    import config as CFG
    assert CFG.STAGE1["batch_size"] == 4
    assert CFG.STAGE1["grad_accum"] == 8


def test_stage2b_iou_thresh_0_60():
    import config as CFG
    assert CFG.STAGE2B["iou_thresh"] == 0.60


def test_bf16_amp_has_no_scaler():
    # The codebase is bf16-by-default on CUDA. get_amp_context must return
    # scaler=None for bf16; the training loops then short-circuit the scaler
    # paths to plain torch ops.
    import torch
    from utils.hardware import get_amp_context
    _, scaler = get_amp_context(torch.bfloat16)
    assert scaler is None


def test_fp16_amp_returns_scaler():
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("fp16 GradScaler is a CUDA-only contract; no CUDA on this machine")
    from utils.hardware import get_amp_context
    _, scaler = get_amp_context(torch.float16)
    assert scaler is not None

def test_well_conf_thresh_is_0_10():
    import config as CFG
    assert CFG.STAGE2B["class_conf_thresh"]["well"] == 0.10

def test_agnostic_nms_enabled():
    import config as CFG
    assert CFG.STAGE2B.get("agnostic_nms") is True
