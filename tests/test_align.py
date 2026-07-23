import pytest

from selfies_for_a_year.align import (
    _compute_similarity_transform,
    _face_oval_in_bounds,
    _key_points_in_bounds,
    _yaw_signal,
)


def test_similarity_transform_maps_dst_points_back_to_src():
    src_left, src_right = (100.0, 200.0), (300.0, 200.0)
    dst_left, dst_right = (400.0, 400.0), (600.0, 400.0)

    a, b, c, d, e, f = _compute_similarity_transform(src_left, src_right, dst_left, dst_right)

    def apply(x, y):
        return a * x + b * y + c, d * x + e * y + f

    assert apply(*dst_left) == pytest.approx(src_left)
    assert apply(*dst_right) == pytest.approx(src_right)


def test_similarity_transform_identity_when_points_match():
    left, right = (10.0, 20.0), (50.0, 20.0)
    a, b, c, d, e, f = _compute_similarity_transform(left, right, left, right)

    assert (a, b, c, d, e, f) == pytest.approx((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))


def test_yaw_signal_zero_when_nose_centered():
    left, right = (0.0, 0.0), (100.0, 0.0)
    nose = (50.0, 30.0)
    face = (left, right, nose, [])

    assert _yaw_signal(face) == pytest.approx(0.0)


def test_yaw_signal_nonzero_when_nose_offset():
    left, right = (0.0, 0.0), (100.0, 0.0)
    nose = (70.0, 30.0)  # shifted toward right eye
    face = (left, right, nose, [])

    assert _yaw_signal(face) > 0


def test_yaw_signal_zero_when_eyes_coincide():
    left = right = (10.0, 10.0)
    face = (left, right, (10.0, 20.0), [])

    assert _yaw_signal(face) == 0.0


def test_key_points_in_bounds_true_when_inside():
    face = ((10.0, 10.0), (90.0, 10.0), (50.0, 50.0), [])
    assert _key_points_in_bounds(face, img_w=100, img_h=100) is True


def test_key_points_in_bounds_false_when_eye_off_image():
    face = ((-5.0, 10.0), (90.0, 10.0), (50.0, 50.0), [])
    assert _key_points_in_bounds(face, img_w=100, img_h=100) is False


def test_face_oval_in_bounds_true_when_few_points_off():
    oval = [(50.0, 50.0)] * 30 + [(-1.0, -1.0)] * 5
    face = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), oval)
    assert _face_oval_in_bounds(face, img_w=100, img_h=100) is True


def test_face_oval_in_bounds_false_when_many_points_off():
    oval = [(50.0, 50.0)] * 15 + [(-1.0, -1.0)] * 20
    face = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), oval)
    assert _face_oval_in_bounds(face, img_w=100, img_h=100) is False
