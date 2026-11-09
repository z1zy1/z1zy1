"""Manifest integration of the isolated position protocol and real loader."""
import copy
import pytest
from test_rsaca_input_preflight import fixture, semantic_path


@pytest.mark.parametrize('dataset', ['second_cc', 'levir_mci'])
def test_actual_protocol_manifest_accepts_e0_and_checks_semantic_arms(tmp_path, dataset):
    from utils.semantic_control_audit import input_manifest
    from utils.rsaca_position_protocol import verify_manifest
    cfg = fixture(tmp_path, dataset)
    registered = input_manifest(cfg)
    assert not registered['missing']
    assert len(registered['samples']) == 2
    assert registered['samples'][0]['semantic_preflight']['group'] == 'original_empty'
    pure = copy.deepcopy(cfg)
    pure.train.semantic_control_arm = 'E0'
    pure.model.semantic_input_mode = 'none'
    pure.data.use_semantic_maps = False
    assert verify_manifest(pure, registered)['samples'][0]['semantic'] == []
    cfg.train.semantic_control_arm = 'E2'
    cfg.model.semantic_fusion_local_position = 'sincos2d_kv'
    assert verify_manifest(cfg, registered) == registered
    phase = 'sem/B' if dataset == 'second_cc' else 'label'
    semantic_path(cfg, phase).unlink()
    # E0 still only checks the common visual/reference manifest.
    verify_manifest(pure, registered)
    with pytest.raises(ValueError, match='ID=neighbour.png.*required file is missing'):
        verify_manifest(cfg, registered)
