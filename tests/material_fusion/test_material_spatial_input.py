"""material_spatial_input_v1 pure-math tests: area geometry, padding,
invalid/zero semantics, grid contract checks and the deterministic
q-permutation rule (independent hand-computed expectations throughout)."""

from __future__ import annotations

import copy
import hashlib
import json
from unittest import mock

import numpy as np
import pytest

from core.material_spatial_input import (
    GRID_PLANES,
    GRID_SIZE,
    NativeInstanceArrays,
    SpatialRefused,
    apply_q_permutation,
    canvas_offset,
    coverage_bits_hex8,
    coverage_from_overlap,
    exact_cell_overlap,
    group_seed64,
    plan_q_permutation,
    scaled_size,
    validate_grid_planes,
)
from scripts.prepare_material_spatial_input import (
    _build_grid_planes,
)
from core.material_spatial_input import area_average_sat


def _uniform(w: int, h: int, value: float) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


def _mask(w: int, h: int, value: int = 1) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def test_multiple_legal_singleton_groups_are_accepted():
    from core.material_spatial_input_admission import check_permutation_plan
    from scripts.verify_material_spatial_input import (
        _build_grid_planes as independent_grid,
    )

    left = np.full((3, 129), 255, dtype=np.uint8)
    mask = np.ones_like(left)
    valid = np.zeros_like(left)
    valid[1, 1] = 1
    grid = independent_grid(
        left, np.where(valid, 255, 0).astype(np.float32),
        valid.astype(np.float32), mask, valid, mask, np.zeros_like(mask))
    plan = plan_q_permutation(grid["planes"], "singleton_sample")
    groups = plan["regions"]["interior"]["groups"]
    assert len(groups) == 2
    assert all(group["cells"] == 1 for group in groups.values())
    assert check_permutation_plan(
        plan, "singleton_sample", {"singleton_sample"}) == plan


@pytest.mark.parametrize("field,value", [
    ("moved_indices", 0.0), ("moved_indices", False),
    ("value_changed_cells", 0.0), ("value_changed_cells", False),
])
def test_singleton_zero_counts_require_true_integers(field, value):
    from core.material_spatial_input_admission import (
        AdmissionRefused, check_permutation_plan,
    )

    planes = {name: np.zeros((128, 128), dtype=np.float32)
              for name in GRID_PLANES}
    planes["coverage_interior"][0, 0] = 0.5
    planes["coverage_object"][0, 0] = 0.5
    plan = plan_q_permutation(planes, "strict_singleton")
    group = next(iter(plan["regions"]["interior"]["groups"].values()))
    group[field] = value
    with pytest.raises(AdmissionRefused, match=field):
        check_permutation_plan(plan, "strict_singleton", {"strict_singleton"})


class TestGeometry:
    def test_never_upscales_and_rounds(self):
        assert scaled_size(64, 32) == (64, 32)
        assert scaled_size(300, 100) == (128, 43)
        # 128/300: w2 = floor(128+0.5)=128; h2 = floor(100*128/300+0.5)
        assert scaled_size(300, 100)[1] == int(np.floor(100 * 128 / 300
                                                       + 0.5))
        assert scaled_size(257, 257) == (128, 128)
        assert scaled_size(1, 1000)[0] == 1

    def test_offset_splits_with_remainder_right_bottom(self):
        # w2=64: x0 = 32, remainder 0; w2=63: x0=32, remainder 1 right
        assert canvas_offset(64, 64) == (32, 32)
        assert canvas_offset(63, 63) == (32, 32)
        assert canvas_offset(62, 62) == (33, 33)


class TestAreaAverage:
    def test_exact_block_mean(self):
        # a 4x4 source scaled to 2 cells per axis: every cell is the
        # mean of its 2x2 block (hand expectation)
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.ones((4, 4), dtype=bool)
        response, denom = area_average_sat(src, mask, 2, 2, (0, 0),
                                            grid=2)
        assert denom == pytest.approx(np.full((2, 2), 4.0))
        expected = src.reshape(2, 2, 2, 2).mean(axis=(1, 3))
        assert response == pytest.approx(expected)

    def test_fractional_edges_hand_expectation(self):
        # a 3-wide source in 2 cells: edges 0, 1.5, 3 -> cell 0 weights
        # (1, 0.5), cell 1 weights (0.5, 1); hand-computed values
        src = np.array([[1.0, 3.0, 9.0]], dtype=np.float32)
        mask = np.ones((1, 3), dtype=bool)
        response, denom = area_average_sat(src, mask, 2, 1, (0, 0),
                                            grid=2)
        # cell 0: (1*1 + 3*0.5) / 1.5 = 5/3; cell 1: (3*0.5 + 9*1)/1.5 = 7
        assert response[0, 0] == pytest.approx((1 * 1 + 3 * 0.5) / 1.5,
                                               abs=1e-6)
        assert response[0, 1] == pytest.approx((3 * 0.5 + 9 * 1.0) / 1.5,
                                               abs=1e-6)
        assert denom[0, 0] == pytest.approx(1.5)
        assert denom[0, 1] == pytest.approx(1.5)

    def test_thin_band_survives(self):
        # a one-pixel-wide band of value 5 amid zeros keeps its value in
        # its own cell (area averaging never smears it away)
        src = np.zeros((4, 4), dtype=np.float32)
        src[:, 2] = 5.0
        band = np.zeros((4, 4), dtype=bool)
        band[:, 2] = True
        response, denom = area_average_sat(src, band, 4, 4, (0, 0),
                                            grid=4)
        assert denom[2, 2] == pytest.approx(1.0)
        assert response[2, 2] == pytest.approx(5.0)

    def test_identity_when_not_scaling(self):
        src = np.random.default_rng(3).normal(size=(5, 7)).astype(
            np.float32)
        mask = np.ones((5, 7), dtype=bool)
        w2, h2 = scaled_size(7, 5, grid=7)
        x0, y0 = canvas_offset(w2, h2, grid=7)
        assert (w2, h2) == (7, 5) and (x0, y0) == (0, 1)
        response, denom = area_average_sat(src, mask, w2, h2, (x0, y0),
                                            grid=7)
        assert np.array_equal(response[y0:y0 + 5, x0:x0 + 7], src)
        assert np.array_equal(denom[y0:y0 + 5, x0:x0 + 7],
                              np.ones((5, 7)))
        assert np.all(response[:y0, :] == 0.0)
        assert np.all(response[y0 + 5:, :] == 0.0)
        assert np.all(denom[:y0, :] == 0.0)

    def test_padding_is_zero_and_offset_correct(self):
        src = _uniform(10, 10, 0.5)
        mask = np.ones((10, 10), dtype=bool)
        w2, h2 = scaled_size(10, 10, grid=8)
        x0, y0 = canvas_offset(w2, h2, grid=8)
        assert (w2, h2) == (8, 8) and (x0, y0) == (0, 0)
        w2b, h2b = scaled_size(9, 9, grid=8)
        xb, yb = canvas_offset(w2b, h2b, grid=8)
        assert xb == (8 - w2b) // 2 and yb == (8 - h2b) // 2
        response, denom = area_average_sat(src, mask, w2b, h2b,
                                            (xb, yb), grid=8)
        assert np.all(response[:yb, :] == 0.0)
        assert np.all(denom[:yb, :] == 0.0)
        assert np.all(response[:, xb + w2b:] == 0.0)
        # the remainder column sits on the RIGHT
        if 8 - xb - w2b > 0:
            assert np.all(response[:, xb + w2b - 1] != 0.0)

    def test_invalid_nan_does_not_pollute(self):
        src = np.full((4, 4), np.nan, dtype=np.float32)
        src[0, 0] = 2.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        response, denom = area_average_sat(src, mask, 4, 4, (0, 0),
                                            grid=4)
        assert np.all(np.isfinite(response))
        assert response[0, 0] == pytest.approx(2.0)
        assert denom[0, 0] == pytest.approx(1.0)
        assert denom[1, 1] == pytest.approx(0.0)
        assert response[1, 1] == pytest.approx(0.0)

    def test_builder_and_verifier_coverage_agree(self):
        """The builder (SAT + exact support) and verifier (its own exact
        weight-matrix reference) grid planes must agree within atol=1e-6
        for a non-integer scaling; the coverage planes are bit-identical
        and in [0,1]."""
        from scripts.verify_material_spatial_input import (
            _build_grid_planes as _verify_grid_planes,
        )

        rng = np.random.default_rng(11)
        for (w, h) in [(199, 299), (300, 200), (97, 311)]:
            L = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
            R = rng.integers(0, 255, size=(h, w)).astype(np.float32)
            q = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
            M = (rng.random((h, w)) > 0.1).astype(np.uint8)
            V = (M & (rng.random((h, w)) > 0.2)).astype(np.uint8)
            interior = (rng.random((h, w)) > 0.5).astype(np.uint8)
            band = ((M > 0) & (interior == 0)).astype(np.uint8)
            builder = _build_grid_planes(L, R, q, M, V, interior, band)
            verifier = _verify_grid_planes(L, R, q, M, V, interior, band)
            for plane_name in GRID_PLANES:
                b = builder["planes"][plane_name]
                v = verifier["planes"][plane_name]
                assert np.allclose(b, v, atol=1e-6, rtol=0), \
                    f"{plane_name} mismatch at w={w} h={h}: max diff " \
                    f"{np.abs(b.astype(np.float64) - v.astype(np.float64)).max()}"
            for name in ("coverage_interior", "coverage_band",
                         "coverage_object"):
                assert np.array_equal(
                    builder["planes"][name].view(np.uint32),
                    verifier["planes"][name].view(np.uint32)), \
                    f"{name} is not bit-identical at w={w} h={h}"
                plane = builder["planes"][name]
                assert np.all(plane >= 0) and np.all(plane <= 1.0), \
                    f"{name} out of [0,1] at w={w} h={h}"

    def test_denominator_is_coverage_in_unit_square(self):
        src = _uniform(2, 2, 0.7)
        mask = np.array([[True, False], [True, True]])
        response, denom = area_average_sat(src, mask, 2, 2, (0, 0),
                                            grid=2)
        assert denom[0, 0] == pytest.approx(1.0)
        assert denom[0, 1] == pytest.approx(0.0)
        assert denom[1, 1] == pytest.approx(1.0)
        assert response[0, 0] == pytest.approx(0.7)


class TestMissingVsRealZero:
    def test_missing_region_vs_genuine_zero_q(self):
        # a missing region (V=0) yields D=0 and response +0; a genuine
        # zero-q pixel inside A yields D>0 and response 0 — separated by
        # the coverage plane
        q = np.zeros((4, 4), dtype=np.float32)
        v = np.zeros((4, 4), dtype=np.uint8)
        v[0, :] = 1          # row 0: a genuine zero-q pixel (V=1)
        # (2, 2) stays V=0: a MISSING pixel
        m = np.ones((4, 4), dtype=np.uint8)
        interior = np.zeros((4, 4), dtype=np.uint8)
        band = np.ones((4, 4), dtype=np.uint8)
        grid = _build_grid_planes(_uniform(4, 4, 0.5),
                                  np.zeros((4, 4), np.float32), q, m, v,
                                  interior, band)
        planes = grid["planes"]
        y0, x0 = grid["offset"]  # w2=h2=4: the window starts at (62, 62)
        wy, wx = y0, x0          # the genuine zero-q pixel (row 0)
        my, mx = y0 + 2, x0 + 2  # the missing pixel (2, 2)
        assert planes["coverage_band"][wy, wx] > 0
        assert planes["q_band"][wy, wx] == 0.0        # genuine zero
        assert planes["coverage_band"][my, mx] == 0.0  # missing
        assert planes["q_band"][my, mx] == 0.0        # missing, +0 is
        # not a real measurement; the coverage plane separates them


class TestGridContract:
    def test_nine_planes_float32_and_zero_padding(self):
        rng = np.random.default_rng(5)
        L = rng.integers(0, 255, size=(40, 60), dtype=np.uint8)
        R = rng.integers(0, 255, size=(40, 60)).astype(np.float32)
        q = rng.uniform(-1, 1, size=(40, 60)).astype(np.float32)
        M = _mask(60, 40)
        V = M.copy()
        interior = M.copy()
        band = np.zeros_like(M)
        grid = _build_grid_planes(L, R, q, M, V, interior, band)
        assert set(grid["planes"]) == set(GRID_PLANES)
        for name, plane in grid["planes"].items():
            assert plane.shape == (GRID_SIZE, GRID_SIZE)
            assert plane.dtype == np.float32
        w2, h2 = scaled_size(60, 40)
        x0, y0 = canvas_offset(w2, h2)
        # padding rows/cols are exactly zero on every plane
        for plane in grid["planes"].values():
            assert np.all(plane[:y0, :] == 0.0)
            assert np.all(plane[:, x0 + w2:] == 0.0)

    def test_v_subset_and_region_values_enforced(self):
        L = np.zeros((8, 8), np.uint8)
        R = np.zeros((8, 8), np.float32)
        q = np.zeros((8, 8), np.float32)
        M = _mask(8, 8)
        V = _mask(8, 8)
        V[0, 0] = 0  # V not a subset (M=1 everywhere, V has an extra 0 is
        # fine); make V exceed M instead
        M2 = np.zeros((8, 8), np.uint8)
        V2 = _mask(8, 8)
        with pytest.raises(Exception):
            _build_grid_planes(L, R, q, M2, V2, M2, np.zeros_like(M2))

    def test_native_record_validation(self):
        ok = NativeInstanceArrays(
            sample_name="s", frame="f", L=np.zeros((4, 4), np.uint8),
            R=np.zeros((4, 4), np.float32), q=np.zeros((4, 4), np.float32),
            M=_mask(4, 4), V=_mask(4, 4),
            region=np.full((4, 4), 2, np.uint8),
            quality=np.zeros(4, np.float32))
        ok.validate()
        bad = NativeInstanceArrays(
            sample_name="s", frame="f", L=np.zeros((4, 4), np.uint8),
            R=np.ones((4, 4), np.float32), q=np.zeros((4, 4), np.float32),
            M=np.zeros((4, 4), np.uint8), V=_mask(4, 4),
            region=np.zeros((4, 4), np.uint8),
            quality=np.zeros(4, np.float32))
        with pytest.raises(SpatialRefused):
            bad.validate()  # V is not a subset of M
        nonfloat = NativeInstanceArrays(
            sample_name="s", frame="f", L=np.zeros((4, 4), np.uint8),
            R=np.zeros((4, 4), np.float64), q=np.zeros((4, 4), np.float32),
            M=_mask(4, 4), V=_mask(4, 4),
            region=np.zeros((4, 4), np.uint8),
            quality=np.zeros(4, np.float32))
        with pytest.raises(SpatialRefused):
            nonfloat.validate()


class TestPermutation:
    def _grid_with_groups(self):
        # 4 cells with two distinct coverage values (two groups of 2)
        coverage = np.zeros((2, 2), dtype=np.float32)
        coverage[0, 0] = 0.25
        coverage[0, 1] = 0.25
        coverage[1, 0] = 0.5
        coverage[1, 1] = 0.5
        q = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        return {"q_interior": q, "coverage_interior": coverage,
                "q_band": np.zeros((2, 2), np.float32),
                "coverage_band": np.zeros((2, 2), np.float32)}

    def test_deterministic_and_grouped(self):
        planes = self._grid_with_groups()
        plan_a = plan_q_permutation(planes, "sample")
        plan_b = plan_q_permutation(planes, "sample")
        assert plan_a["regions"]["interior"] == plan_b["regions"][
            "interior"]
        interior = plan_a["regions"]["interior"]
        assert interior["movable_cells"] == 4
        assert len(interior["groups"]) == 2  # two distinct coverages
        for bits, group in interior["groups"].items():
            assert group["cells"] == 2
            assert group["permuted"] is True
            # the permutation stays INSIDE its coverage group: applying
            # it never moves a value across coverages
        permuted = apply_q_permutation(copy.deepcopy(planes), plan_a)
        assert np.array_equal(permuted["coverage_interior"],
                              planes["coverage_interior"])
        # the (q, coverage) bit multisets are identical per region
        for region in ("interior", "band"):
            before = np.stack([
                planes[f"q_{region}"].ravel().view(np.uint32),
                planes[f"coverage_{region}"].ravel().view(np.uint32)],
                axis=1)
            after = np.stack([
                permuted[f"q_{region}"].ravel().view(np.uint32),
                permuted[f"coverage_{region}"].ravel().view(np.uint32)],
                axis=1)
            before_sorted = before[np.lexsort(before.T[::-1])]
            after_sorted = after[np.lexsort(after.T[::-1])]
            assert np.array_equal(before_sorted, after_sorted)

    def test_seed64_derivation(self):
        domain = ("material_spatial_input_v1|sample|interior|00000000|"
                  "2026")
        expected = int.from_bytes(
            hashlib.sha256(domain.encode("utf-8")).digest()[:8], "little")
        assert group_seed64("sample", "interior", "00000000") == expected

    def test_single_cell_group_keeps_value(self):
        coverage = np.zeros((2, 2), dtype=np.float32)
        coverage[0, 0] = 0.1
        coverage[1, 1] = 0.2  # every coverage value unique -> singleton
        coverage[0, 1] = 0.3
        coverage[1, 0] = 0.4
        q = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        planes = {"q_interior": q, "coverage_interior": coverage,
                  "q_band": np.zeros((2, 2), np.float32),
                  "coverage_band": np.zeros((2, 2), np.float32)}
        plan = plan_q_permutation(planes, "s")
        interior = plan["regions"]["interior"]
        assert interior["moved_indices"] == 0
        assert interior["value_changed_cells"] == 0
        assert all(g["permuted"] is False
                   for g in interior["groups"].values())

    def test_all_equal_values_move_without_value_change(self):
        coverage = np.zeros((2, 2), dtype=np.float32)
        coverage[:] = 0.7
        q = np.full((2, 2), 0.5, dtype=np.float32)
        planes = {"q_interior": q, "coverage_interior": coverage,
                  "q_band": np.zeros((2, 2), np.float32),
                  "coverage_band": np.zeros((2, 2), np.float32)}
        plan = plan_q_permutation(planes, "s")
        interior = plan["regions"]["interior"]
        assert interior["moved_indices"] > 0
        assert interior["value_changed_cells"] == 0

    def test_invalid_cells_never_move(self):
        coverage = np.zeros((2, 2), dtype=np.float32)
        coverage[0, 0] = 0.0   # D=0: invalid, never movable
        coverage[0, 1] = 0.6
        coverage[1, 0] = 0.6
        coverage[1, 1] = 0.6
        q = np.array([[9.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        planes = {"q_interior": q, "coverage_interior": coverage,
                  "q_band": np.zeros((2, 2), np.float32),
                  "coverage_band": np.zeros((2, 2), np.float32)}
        plan = plan_q_permutation(planes, "s")
        interior = plan["regions"]["interior"]
        assert interior["movable_cells"] == 3
        all_cells = [c for g in interior["groups"].values()
                     for c in g.get("cell_indices", [])]
        assert 0 not in all_cells, "the D=0 cell must never move"
        permuted = apply_q_permutation(copy.deepcopy(planes), plan)
        assert permuted["q_interior"][0, 0] == 9.0

    def test_coverage_bits_hex8(self):
        value = np.float32(0.5)
        assert coverage_bits_hex8(value) == f"{value.view(np.uint32):08x}"


class TestGrayNormalizationAndImmutability:
    def test_native_R_255_maps_to_grid_1_and_inputs_unchanged(self):
        from scripts.prepare_material_spatial_input import _build_grid_planes

        L = np.full((2, 2), 255, dtype=np.uint8)
        R = np.array([[255.0, 255.0], [0.0, 0.0]], dtype=np.float32)
        q = np.array([[0.5, -0.5], [0.25, -0.25]], dtype=np.float32)
        M = _mask(2, 2)
        V = _mask(2, 2)
        interior = _mask(2, 2)
        band = np.zeros_like(M)
        before = {name: np.array(value, copy=True) for name, value in
                  (("L", L), ("R", R), ("q", q), ("M", M), ("V", V),
                   ("interior", interior), ("band", band))}
        grid = _build_grid_planes(L, R, q, M, V, interior, band)
        x0, y0 = grid["offset"]
        # R/255: a native 255 becomes exactly 1.0 on the grid (the 2x2
        # crop maps 1:1 into the window at the letterbox offset)
        assert grid["planes"]["R_interior"][y0, x0] == np.float32(1.0)
        assert grid["planes"]["R_interior"][y0 + 1, x0] == np.float32(0.0)
        assert grid["planes"]["L_interior"][y0, x0] == np.float32(1.0)
        # the native arrays are never modified in place
        for name, value in (("L", L), ("R", R), ("q", q), ("M", M),
                            ("V", V), ("interior", interior),
                            ("band", band)):
            assert np.array_equal(before[name].view(np.uint8),
                                  value.view(np.uint8)), name

    def test_exact_overlap_conservation_and_support_equivalence(self):
        rng = np.random.default_rng(19)
        w, h = 300, 200
        w2, h2 = scaled_size(w, h)
        offset = canvas_offset(w2, h2)
        mask = (rng.random((h, w)) > 0.7).astype(np.uint8)
        overlap = exact_cell_overlap(mask, w, h, w2, h2, offset)
        # the scaled areas sum exactly to the scaled pixel areas
        assert int(overlap.sum()) == int(w2 * h2) * int(mask.sum())
        coverage = coverage_from_overlap(overlap, w, h)
        # T == 0 exactly when the coverage bits are zero, and every
        # T > 0 cell keeps a strictly positive float32 coverage
        bits = coverage.view(np.uint32)
        assert np.array_equal(bits != 0, overlap > 0)
        assert np.all(coverage[overlap > 0] > 0)


class TestExactSupportSemantics:
    """True zero support must produce positive-zero coverage with no SAT
    noise, and true tiny support must never be thresholded away."""

    def _single_pixel_grid(self):
        from scripts.prepare_material_spatial_input import _build_grid_planes

        w, h = 300, 200
        w2, h2 = scaled_size(w, h)
        L = np.full((h, w), 200, dtype=np.uint8)
        R = np.full((h, w), 100, dtype=np.float32)
        q = np.full((h, w), 0.5, dtype=np.float32)
        M = np.zeros((h, w), dtype=np.uint8)
        M[150, 150] = 1
        V = M.copy()
        interior = M.copy()
        band = np.zeros_like(M)
        grid = _build_grid_planes(L, R, q, M, V, interior, band)
        offset = grid["offset"]
        # the hand-computed exact overlap of the single pixel with every
        # cell (independent min/max arithmetic, scaled units)
        hand_T = np.zeros((h2, w2), dtype=np.int64)
        for j in range(w2):
            lx = max(0, min((j + 1) * w, (150 + 1) * w2)
                     - max(j * w, 150 * w2))
            if lx == 0:
                continue
            for i in range(h2):
                ly = max(0, min((i + 1) * h, (150 + 1) * h2)
                         - max(i * h, 150 * h2))
                if ly:
                    hand_T[i, j] = lx * ly
        return grid, (w, h, w2, h2, offset), hand_T

    def test_true_zero_support_has_no_noise(self):
        grid, (w, h, w2, h2, offset), hand_T = self._single_pixel_grid()
        coverage = grid["planes"]["coverage_interior"]
        window = coverage[offset[1]:offset[1] + h2,
                          offset[0]:offset[0] + w2]
        supported = window.view(np.uint32) != 0
        assert np.array_equal(supported, hand_T > 0)
        # every unsupported cell of every response plane: positive-zero
        # bits (no -0.0, no 1e-13 SAT noise)
        full_supported = coverage.view(np.uint32) != 0
        for name in ("q_interior", "L_interior", "R_interior"):
            bits = grid["planes"][name].view(np.uint32)
            assert np.all(bits[~full_supported] == 0), name
        assert np.all(grid["planes"]["coverage_band"].view(np.uint32)
                      == 0)
        object_bits = grid["planes"]["coverage_object"].view(np.uint32)
        assert np.array_equal(object_bits != 0, full_supported)
        # the plan sees exactly the hand-counted supported cells
        plan = plan_q_permutation(grid["planes"], "s")
        assert plan["regions"]["interior"]["movable_cells"] == \
            int((hand_T > 0).sum())
        assert plan["regions"]["band"]["movable_cells"] == 0

    def test_tiny_positive_support_survives(self):
        grid, (w, h, w2, h2, offset), hand_T = self._single_pixel_grid()
        coverage = grid["planes"]["coverage_interior"]
        window = coverage[offset[1]:offset[1] + h2,
                          offset[0]:offset[0] + w2]
        for i, j in zip(*np.nonzero(hand_T)):
            assert window[i, j].view(np.uint32) != 0
            expected = float(hand_T[i, j]) / float(w * h)
            assert float(window[i, j]) == pytest.approx(expected, rel=1e-6)
        assert float(window[window != 0][0]) > 1e-3  # far above any
        # conceivable noise threshold


class TestSatNumericStability:
    def test_true_zero_cells_survive_sat_cancellation(self):
        """The frozen counterexample: 178x151, seed20 random gray/255,
        all-ones mask; every source pixel intersecting the listed grid
        cells is zeroed. The four-corner SAT cancellation used to leave
        tiny nonzero responses there; the per-cell convex-hull patch
        must make those true-zero cells positive zero and keep
        everything inside the physical range."""
        from scripts.prepare_material_spatial_input import (
            _build_grid_planes,
        )

        w, h = 178, 151
        rng = np.random.default_rng(20)
        gray = rng.integers(0, 256, size=(h, w)).astype(np.uint8)
        w2, h2 = scaled_size(w, h)
        x0, y0 = canvas_offset(w2, h2)
        for gy in range(10, 91, 10):
            for gx in range(10, 111, 10):
                j, i = gx - x0, gy - y0
                lo_x, hi_x = j * w, (j + 1) * w
                lo_y, hi_y = i * h, (i + 1) * h
                for y in range(h):
                    ly = min(hi_y, (y + 1) * h2) - max(lo_y, y * h2)
                    if ly <= 0:
                        continue
                    for x in range(w):
                        lx = min(hi_x, (x + 1) * w2) - max(lo_x, x * w2)
                        if lx > 0:
                            gray[y, x] = np.uint8(0)
        L = gray
        R = gray.astype(np.float32)
        q = np.zeros((h, w), dtype=np.float32)
        M = np.ones((h, w), dtype=np.uint8)
        V = M.copy()
        interior = M.copy()
        band = np.zeros_like(M)
        grid = _build_grid_planes(L, R, q, M, V, interior, band)
        planes = grid["planes"]
        n_true_zero = 0
        for gy in range(10, 91, 10):
            for gx in range(10, 111, 10):
                for plane in ("L_interior", "R_interior"):
                    value = planes[plane][gy, gx]
                    assert 0.0 <= float(value) <= 1.0, (plane, gy, gx)
                coverage = planes["coverage_interior"][gy, gx]
                assert coverage > 0  # the cell itself is fully covered
                # the response must be positive zero exactly when every
                # intersecting source value was zeroed
                block = gray[max(0, (gy - y0) * h // h2):
                             ((gy - y0) + 1) * h // h2 + 1,
                             max(0, (gx - x0) * w // w2):
                             ((gx - x0) + 1) * w // w2 + 1]
                if block.size and int(block.max()) == 0:
                    n_true_zero += 1
                    for plane in ("L_interior", "R_interior"):
                        bits = planes[plane][gy, gx].view(np.uint32)
                        assert bits == 0, (plane, gy, gx,
                                           planes[plane][gy, gx])
        assert n_true_zero >= 40, n_true_zero


class TestVerifierReferenceIndependence:
    def test_builder_sat_failure_leaves_verifier_reference_working(self):
        """The verifier's grid reference must not route through the
        builder's SAT area function even when that function raises."""
        from scripts import prepare_material_spatial_input as prep
        from scripts import verify_material_spatial_input as verify
        import core.material_spatial_input as core_mod

        def bomb(*a, **k):
            raise AssertionError("the builder SAT must not be called")

        rng = np.random.default_rng(23)
        w, h = 150, 100
        L = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
        R = rng.integers(0, 255, size=(h, w)).astype(np.float32)
        q = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
        M = (rng.random((h, w)) > 0.3).astype(np.uint8)
        V = M.copy()
        interior = M.copy()
        band = np.zeros_like(M)
        # sanity: the patched symbol is the one the builder calls
        with mock.patch.object(prep, "area_average_sat", bomb), \
                mock.patch.object(core_mod, "area_average_sat", bomb):
            with pytest.raises(AssertionError):
                prep._build_grid_planes(L, R, q, M, V, interior, band)
            grid = verify._build_grid_planes(L, R, q, M, V, interior,
                                             band)
        assert set(grid["planes"]) == set(GRID_PLANES)


class TestSavedPlaneContract:
    """The strict float32 contract on the saved planes: positive-zero
    bit patterns, ranges, dtype/shape — no allclose-only loosening."""

    def _planes(self):
        from scripts.prepare_material_spatial_input import _build_grid_planes

        w, h = 5, 5
        L = np.full((h, w), 128, dtype=np.uint8)
        R = np.full((h, w), 64, dtype=np.float32)
        q = np.full((h, w), 0.25, dtype=np.float32)
        M = np.zeros((h, w), dtype=np.uint8)
        M[2, 2] = 1
        V = M.copy()
        interior = M.copy()
        band = np.zeros_like(M)
        grid = _build_grid_planes(L, R, q, M, V, interior, band)
        return grid["planes"], (grid["w2"], grid["h2"], grid["offset"])

    def test_valid_planes_pass(self):
        planes, geometry = self._planes()
        validate_grid_planes(planes, geometry[0], geometry[1], geometry[2])

    def test_padding_bit_patterns_are_refused(self):
        planes, geometry = self._planes()
        w2, h2, (x0, y0) = geometry
        assert x0 > 0 and y0 > 0  # padding exists for the 5x5 crop
        for value, label in ((np.float32(-0.0), "negative zero"),
                             (np.float32(1e-40), "tiny nonzero")):
            tampered = {k: np.array(v, copy=True)
                        for k, v in planes.items()}
            tampered["q_band"][0, 0] = value
            with pytest.raises(SpatialRefused):
                validate_grid_planes(tampered, w2, h2, (x0, y0)), label
            tampered = {k: np.array(v, copy=True)
                        for k, v in planes.items()}
            tampered["coverage_object"][0, 0] = value
            with pytest.raises(SpatialRefused):
                validate_grid_planes(tampered, w2, h2, (x0, y0)), label

    def test_range_and_structure_refusals(self):
        planes, geometry = self._planes()
        w2, h2, offset = geometry
        tampered = {k: np.array(v, copy=True) for k, v in planes.items()}
        tampered["coverage_interior"][y0c := offset[1] + h2 - 1,
                                     offset[0]] = np.float32(1.5)
        with pytest.raises(SpatialRefused):
            validate_grid_planes(tampered, w2, h2, offset)
        tampered = {k: np.array(v, copy=True) for k, v in planes.items()}
        tampered["coverage_interior"][offset[1], offset[0]] = \
            np.float32(-0.0)
        with pytest.raises(SpatialRefused):
            validate_grid_planes(tampered, w2, h2, offset)
        tampered = {k: np.array(v, copy=True) for k, v in planes.items()}
        tampered["q_interior"][offset[1], offset[0]] = np.float32(np.nan)
        with pytest.raises(SpatialRefused):
            validate_grid_planes(tampered, w2, h2, offset)
        tampered = {k: v.astype(np.float64)
                    for k, v in planes.items()}
        with pytest.raises(SpatialRefused):
            validate_grid_planes(tampered, w2, h2, offset)
        tampered = {k: v[:, :127] for k, v in planes.items()}
        with pytest.raises(SpatialRefused):
            validate_grid_planes(tampered, w2, h2, offset)

    def test_response_must_be_positive_zero_without_support(self):
        planes, geometry = self._planes()
        w2, h2, offset = geometry
        tampered = {k: np.array(v, copy=True) for k, v in planes.items()}
        # a padding cell of q_interior must be +0; 1e-30 is refused
        tampered["q_interior"][0, 0] = np.float32(1e-30)
        with pytest.raises(SpatialRefused):
            validate_grid_planes(tampered, w2, h2, offset)
