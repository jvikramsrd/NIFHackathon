def test_device_is_torch_device():
    import torch
    import config as CFG
    assert isinstance(CFG.DEVICE, torch.device)

def test_road_class_weight_is_5_0():
    import config as CFG
    assert CFG.STAGE1["class_weights"][2] == 5.0

def test_arcface_m_is_0_55():
    import config as CFG
    assert CFG.STAGE2A["arcface_m"] == 0.55

def test_sahi_overlap_is_0_40():
    import config as CFG
    assert CFG.STAGE2B["sahi_overlap_ratio"] == 0.40

def test_well_conf_thresh_is_0_10():
    import config as CFG
    assert CFG.STAGE2B["class_conf_thresh"]["well"] == 0.10

def test_agnostic_nms_enabled():
    import config as CFG
    assert CFG.STAGE2B.get("agnostic_nms") is True
