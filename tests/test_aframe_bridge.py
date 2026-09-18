import aframe_bridge


def test_missing_t_event_fails_before_heavy_imports():
    # Validation returns before importing buoy/torch, so this runs bare.
    r = aframe_bridge.run_from_skyportal_inputs({"analysis_parameters": {}})
    assert r["status"] == "failure"
    assert "t_event" in r["message"]


def test_ifos_parsing():
    assert aframe_bridge._ifos({}) == ("H1", "L1")
    assert aframe_bridge._ifos({"ifos": "H1, L1, V1"}) == ("H1", "L1", "V1")
    assert aframe_bridge._ifos({"ifos": ["H1", "V1"]}) == ("H1", "V1")


def test_compute_far_no_background():
    far, bound, n_louder, Tb = aframe_bridge._compute_far(5.0, None)
    assert far is None and bound is None and n_louder is None and Tb is None
