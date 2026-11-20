# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
"""
Tests for the dual-source alignment feature pipeline
(:mod:`tsfresh.feature_extraction.alignment`).
"""

import numpy as np
import pandas as pd
import pytest

import tsfresh
from tsfresh.feature_extraction.alignment import (
    DualSourceResult,
    extract_features_dual_source,
    match_ordered_observations,
)

FC_PARAMETERS = {
    "sum_values": None,
    "maximum": None,
    "minimum": None,
    "length": None,
}


def make_example_sides():
    """
    Two small sensor sources with non-aligned sampling times.

    left entity 1/temp:    t=[0, 1, 2, 3]      v=[10, 11, 12, 13]
    right entity 1/temp:   t=[0.1, 1.9, 2.05, 3.2]  v=[1, 2, 3, 4]
    left entity 1/pressure exists only on the left,
    entity 2/temp exists only on the right,
    entity 3/temp is outside of the tolerance on both sides,
    entity 3/humidity:     left t=[0, 1] v=[5, 7], right t=[0.05, 1.05] v=[1, 2]
    """
    left = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 3, 3, 3, 3],
            "kind": ["temp"] * 4 + ["pressure"] * 2 + ["temp"] * 2 + ["humidity"] * 2,
            "time": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 10.0, 0.0, 1.0],
            "value": [10.0, 11.0, 12.0, 13.0, 100.0, 101.0, 20.0, 21.0, 5.0, 7.0],
            "obs": ["L{}".format(i) for i in range(10)],
        }
    )
    right = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
            "kind": ["temp"] * 4 + ["temp"] * 2 + ["temp"] + ["humidity"] * 2,
            "time": [0.1, 1.9, 2.05, 3.2, 0.0, 1.0, 5.0, 0.05, 1.05],
            "value": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 30.0, 1.0, 2.0],
            "obs": ["R{}".format(i) for i in range(9)],
        }
    )
    return left, right


def run_example(left, right, **kwargs):
    params = dict(
        column_id="id",
        column_kind="kind",
        column_sort="time",
        column_value="value",
        column_obs_id="obs",
        tolerance=0.25,
        default_fc_parameters=FC_PARAMETERS,
        n_jobs=0,
        disable_progressbar=True,
    )
    params.update(kwargs)
    return extract_features_dual_source(left, right, **params)


class TestMatchOrderedObservations:
    def test_matches_within_tolerance(self):
        pairs = match_ordered_observations([0.0, 1.0, 2.0], [0.1, 1.1, 2.1], 0.25)
        assert pairs == [(0, 0), (1, 1), (2, 2)]

    def test_tolerance_boundary_is_inclusive(self):
        assert match_ordered_observations([0.0], [0.5], 0.5) == [(0, 0)]

    def test_beyond_tolerance_no_pairs(self):
        assert match_ordered_observations([0.0, 1.0], [0.3, 1.3], 0.2) == []

    def test_zero_tolerance_matches_equal_times(self):
        assert match_ordered_observations([1.0, 2.0], [2.0, 3.0], 0.0) == [(1, 0)]

    def test_one_to_one_no_reuse_of_observations(self):
        # a per-side nearest neighbour search would pair both left
        # observations with the single right one; here every observation
        # may be used at most once
        pairs = match_ordered_observations([0.0, 10.0], [9.8], 0.5)
        assert pairs == [(1, 0)]

    def test_cardinality_is_maximized_before_cost_is_minimized(self):
        # the cheapest single pair would be (1, 0) with cost 0.1, but two
        # pairs are preferred even though they cost more in total
        pairs = match_ordered_observations([0.0, 0.9], [0.8, 1.7], 1.0)
        assert pairs == [(0, 0), (1, 1)]

    def test_total_abs_time_difference_is_minimized(self):
        # two maximum-cardinality matchings exist:
        # {(0, 0), (1, 1)} with cost 0.2 + 0.5 = 0.7 and
        # {(0, 0), (1, 2)} with cost 0.2 + 0.8 = 1.0
        pairs = match_ordered_observations([0.0, 5.0], [0.2, 4.5, 5.8], 1.0)
        assert pairs == [(0, 0), (1, 1)]

    def test_pairs_do_not_cross(self):
        pairs = match_ordered_observations([0.0, 0.6, 1.2], [0.05, 0.65, 1.25], 0.2)
        left_indices = [p[0] for p in pairs]
        right_indices = [p[1] for p in pairs]
        assert left_indices == sorted(left_indices)
        assert right_indices == sorted(right_indices)
        assert pairs == [(0, 0), (1, 1), (2, 2)]

    def test_unsorted_input_returns_positional_indices_in_time_order(self):
        pairs = match_ordered_observations([2.0, 0.0, 1.0], [1.05, 0.05, 2.05], 0.1)
        # sorted left times 0.0, 1.0, 2.0 sit at positions 1, 2, 0;
        # sorted right times 0.05, 1.05, 2.05 sit at positions 1, 0, 2
        assert pairs == [(1, 1), (2, 0), (0, 2)]

    def test_tied_optima_are_allowed(self):
        # both right observations are equally good partners; any of the two
        # optimal matchings may be returned
        pairs = match_ordered_observations([0.0], [-0.1, 0.1], 0.1)
        assert len(pairs) == 1
        assert pairs[0][0] == 0
        assert pairs[0][1] in (0, 1)

    def test_empty_inputs(self):
        assert match_ordered_observations([], [], 1.0) == []
        assert match_ordered_observations([1.0], [], 1.0) == []
        assert match_ordered_observations([], [1.0], 1.0) == []

    def test_integer_times_are_matched_exactly(self):
        pairs = match_ordered_observations([0, 5, 10], [1, 6, 20], 1)
        assert pairs == [(0, 0), (1, 1)]

    def test_nan_and_infinite_times_fail(self):
        with pytest.raises(ValueError):
            match_ordered_observations([0.0, np.nan], [0.0], 1.0)
        with pytest.raises(ValueError):
            match_ordered_observations([0.0], [np.inf], 1.0)
        with pytest.raises(ValueError):
            match_ordered_observations([0.0], [-np.inf], 1.0)

    def test_invalid_tolerance_fails(self):
        for tolerance in (-1.0, np.inf, np.nan, "a lot", None):
            with pytest.raises(ValueError):
                match_ordered_observations([0.0], [0.0], tolerance)

    def test_invalid_time_arrays_fail(self):
        with pytest.raises(ValueError):
            match_ordered_observations([[0.0, 1.0]], [0.0], 1.0)
        with pytest.raises(ValueError):
            match_ordered_observations(["soon"], [0.0], 1.0)


class TestExtractFeaturesDualSource:
    def test_returns_result_with_reports(self):
        left, right = make_example_sides()
        result = run_example(left, right)
        assert isinstance(result, DualSourceResult)
        assert isinstance(result.features, pd.DataFrame)
        assert isinstance(result.aligned, pd.DataFrame)
        assert isinstance(result.unmatched, pd.DataFrame)

    def test_aligned_report_keeps_both_sides_and_difference(self):
        left, right = make_example_sides()
        aligned = run_example(left, right).aligned

        assert list(aligned.columns) == [
            "id",
            "kind",
            "left_id",
            "right_id",
            "left_time",
            "right_time",
            "left_value",
            "right_value",
            "value_diff",
            "time_diff",
        ]
        assert len(aligned) == 5

        temp = aligned[(aligned.id == 1) & (aligned.kind == "temp")]
        assert list(temp.left_id) == ["L0", "L2", "L3"]
        assert list(temp.right_id) == ["R0", "R2", "R3"]
        assert list(temp.left_time) == [0.0, 2.0, 3.0]
        assert list(temp.right_time) == [0.1, 2.05, 3.2]
        assert list(temp.left_value) == [10.0, 12.0, 13.0]
        assert list(temp.right_value) == [1.0, 3.0, 4.0]
        # the difference is left minus right
        assert list(temp.value_diff) == [9.0, 9.0, 9.0]
        np.testing.assert_allclose(temp.time_diff, [-0.1, -0.05, -0.2])

        humidity = aligned[aligned.kind == "humidity"]
        assert list(humidity.value_diff) == [4.0, 5.0]

    def test_unmatched_report_keeps_source_side(self):
        left, right = make_example_sides()
        unmatched = run_example(left, right).unmatched

        assert list(unmatched.columns) == [
            "id",
            "kind",
            "side",
            "obs_id",
            "time",
            "value",
        ]
        # left t=1.0 of entity 1/temp has no partner within the tolerance
        # right t=1.9 of entity 1/temp stays unmatched as well
        own = unmatched[(unmatched.id == 1) & (unmatched.kind == "temp")]
        assert list(own.side) == ["left", "right"]
        assert list(own.obs_id) == ["L1", "R1"]
        assert list(own.time) == [1.0, 1.9]
        assert list(own.value) == [11.0, 2.0]

        # the pressure channel exists only on the left side
        pressure = unmatched[unmatched.kind == "pressure"]
        assert list(pressure.side) == ["left", "left"]
        assert list(pressure.obs_id) == ["L4", "L5"]

        # entity 2 exists only on the right side
        entity2 = unmatched[unmatched.id == 2]
        assert list(entity2.side) == ["right", "right"]
        assert list(entity2.obs_id) == ["R4", "R5"]

        # entity 3/temp is outside of the tolerance on both sides
        entity3 = unmatched[(unmatched.id == 3) & (unmatched.kind == "temp")]
        assert list(entity3.side) == ["left", "left", "right"]
        assert list(entity3.obs_id) == ["L6", "L7", "R6"]

    def test_every_observation_appears_exactly_once(self):
        left, right = make_example_sides()
        result = run_example(left, right)

        for side, df, id_column in (
            ("left", left, "left_id"),
            ("right", right, "right_id"),
        ):
            original = set(zip(df.id, df.kind, df.obs))
            paired = set(
                zip(result.aligned.id, result.aligned.kind, result.aligned[id_column])
            )
            own_unmatched = result.unmatched[result.unmatched.side == side]
            unpaired = set(
                zip(own_unmatched.id, own_unmatched.kind, own_unmatched.obs_id)
            )
            assert not (paired & unpaired)
            assert paired | unpaired == original

    def test_features_are_computed_from_real_differences(self):
        left, right = make_example_sides()
        features = run_example(left, right).features

        # entity 1/temp: differences are [9, 9, 9] at the left times [0, 2, 3]
        assert features.loc[1, "temp__sum_values"] == 27.0
        assert features.loc[1, "temp__maximum"] == 9.0
        assert features.loc[1, "temp__minimum"] == 9.0
        assert features.loc[1, "temp__length"] == 3.0

        # entity 3/humidity: differences are [4, 5] at the left times [0, 1]
        assert features.loc[3, "humidity__sum_values"] == 9.0
        assert features.loc[3, "humidity__maximum"] == 5.0
        assert features.loc[3, "humidity__length"] == 2.0

    def test_no_zero_padding_and_no_zero_imputation(self):
        left, right = make_example_sides()
        result = run_example(left, right)

        # entity 3/temp has no pairs: its features stay missing and are
        # not filled with zeros
        assert np.isnan(result.features.loc[3, "temp__sum_values"])
        # entity 2 has no pairs at all and gets no row
        assert 2 not in result.features.index
        # the pressure channel has no pairs anywhere: no fabricated columns
        assert "pressure__sum_values" not in result.features.columns
        # the unmatched observations did not leak into the difference series
        # (entity 1/temp has exactly 3 pairs, not 4 zero-padded ones)
        assert result.features.loc[1, "temp__length"] == 3.0

    def test_pairing_ignores_observation_values(self):
        left, right = make_example_sides()
        result = run_example(left, right)

        right_permuted = right.copy()
        rng = np.random.RandomState(42)
        right_permuted["value"] = rng.permutation(right_permuted["value"].to_numpy())
        result_permuted = run_example(left, right_permuted)

        # the times are unchanged, so the pairing must be identical
        assert list(result.aligned.left_id) == list(result_permuted.aligned.left_id)
        assert list(result.aligned.right_id) == list(result_permuted.aligned.right_id)

    def test_kind_to_fc_parameters_are_respected_per_channel(self):
        left, right = make_example_sides()
        features = run_example(
            left,
            right,
            default_fc_parameters=None,
            kind_to_fc_parameters={
                "temp": {"sum_values": None},
                "humidity": {"maximum": None},
            },
        ).features
        assert features.loc[1, "temp__sum_values"] == 27.0
        assert features.loc[3, "humidity__maximum"] == 5.0
        assert "temp__maximum" not in features.columns
        assert "humidity__sum_values" not in features.columns

    def test_difference_series_is_indexed_by_left_times(self):
        # linear_trend_timewise regresses the values on the *index* of the
        # series (in hours); it only runs if the index is a DatetimeIndex
        # built from the real left observation times
        left = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "kind": ["temp"] * 3,
                "time": pd.to_datetime(
                    ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
                ),
                "value": [10.0, 11.0, 12.0],
            }
        )
        right = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "kind": ["temp"] * 3,
                "time": pd.to_datetime(
                    ["2020-01-01 00:01", "2020-01-01 01:01", "2020-01-01 02:01"]
                ),
                "value": [10.0, 10.0, 10.0],
            }
        )
        result = extract_features_dual_source(
            left,
            right,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value",
            tolerance=pd.Timedelta("5min"),
            default_fc_parameters={"linear_trend_timewise": [{"attr": "slope"}]},
            n_jobs=0,
            disable_progressbar=True,
        )
        # the differences [0, 1, 2] over the left times [0h, 1h, 2h] have slope 1
        assert result.features.loc[
            1, 'temp__linear_trend_timewise__attr_"slope"'
        ] == pytest.approx(1.0)
        # the aligned report keeps the real datetimes and the signed time difference
        assert list(result.aligned.time_diff) == [pd.Timedelta("-1min")] * 3
        assert result.aligned.left_time.iloc[0] == pd.Timestamp("2020-01-01 00:00")

    def test_generated_observation_ids_when_column_missing(self):
        left, right = make_example_sides()
        left = left.drop(columns="obs")
        right = right.drop(columns="obs")
        result = extract_features_dual_source(
            left,
            right,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value",
            tolerance=0.25,
            default_fc_parameters=FC_PARAMETERS,
            n_jobs=0,
            disable_progressbar=True,
        )
        # positions in the input data frames are used as observation ids
        temp = result.aligned[result.aligned.kind == "temp"]
        assert list(temp.left_id) == [0, 2, 3]
        assert list(temp.right_id) == [0, 2, 3]

    def test_integer_kinds_are_matched_as_strings(self):
        left = pd.DataFrame(
            {"id": [1, 1], "kind": [7, 7], "time": [0.0, 1.0], "value": [1.0, 2.0]}
        )
        right = pd.DataFrame(
            {"id": [1, 1], "kind": [7, 7], "time": [0.1, 1.1], "value": [0.5, 0.5]}
        )
        result = extract_features_dual_source(
            left,
            right,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value",
            tolerance=0.2,
            default_fc_parameters={"sum_values": None},
            n_jobs=0,
            disable_progressbar=True,
        )
        assert len(result.aligned) == 2
        assert result.features.loc[1, "7__sum_values"] == pytest.approx(2.0)

    def test_reports_are_saved_to_disk(self, tmp_path):
        left, right = make_example_sides()
        output_dir = tmp_path / "reports" / "run1"
        result = run_example(left, right, output_dir=output_dir)

        aligned_file = output_dir / "aligned_pairs.csv"
        unmatched_file = output_dir / "unmatched_observations.csv"
        features_file = output_dir / "features.csv"
        assert aligned_file.exists()
        assert unmatched_file.exists()
        assert features_file.exists()

        aligned = pd.read_csv(aligned_file)
        assert len(aligned) == len(result.aligned)
        assert list(aligned.columns) == list(result.aligned.columns)

        unmatched = pd.read_csv(unmatched_file)
        assert len(unmatched) == len(result.unmatched)
        assert set(unmatched.side) == {"left", "right"}

        features = pd.read_csv(features_file, index_col="id")
        assert features.loc[1, "temp__sum_values"] == 27.0

    def test_empty_side_leaves_everything_unmatched(self):
        left, right = make_example_sides()
        empty_left = left.iloc[0:0].copy()
        result = run_example(empty_left, right)

        assert len(result.aligned) == 0
        assert len(result.features) == 0
        assert len(result.unmatched) == len(right)
        assert set(result.unmatched.side) == {"right"}

    def test_old_extract_features_api_still_works(self):
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "kind": ["a", "a", "a", "a"],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )
        features = tsfresh.extract_features(
            df,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="val",
            default_fc_parameters={"sum_values": None},
            n_jobs=0,
            disable_progressbar=True,
        )
        assert features.loc[1, "a__sum_values"] == 3.0
        assert features.loc[2, "a__sum_values"] == 7.0


class TestExtractFeaturesDualSourceValidation:
    def setup_method(self):
        self.left, self.right = make_example_sides()
        self.kwargs = dict(
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value",
            tolerance=0.25,
            n_jobs=0,
            disable_progressbar=True,
        )

    def test_duplicate_observation_ids_fail(self):
        left = self.left.copy()
        left.loc[1, "obs"] = "L0"  # duplicates L0 within entity 1/temp
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            extract_features_dual_source(
                left, self.right, column_obs_id="obs", **self.kwargs
            )

    def test_same_observation_id_in_different_channels_is_allowed(self):
        left = self.left.copy()
        # L4 belongs to entity 1/pressure: no duplicate within one channel
        left.loc[4, "obs"] = "L0"
        result = extract_features_dual_source(
            left, self.right, column_obs_id="obs", **self.kwargs
        )
        assert len(result.aligned) == 5

    def test_missing_columns_fail(self):
        with pytest.raises(ValueError, match="Column not found"):
            extract_features_dual_source(
                self.left.drop(columns="time"), self.right, **self.kwargs
            )
        with pytest.raises(ValueError, match="Column not found"):
            extract_features_dual_source(
                self.left, self.right, column_obs_id="does_not_exist", **self.kwargs
            )

    def test_missing_parameters_fail(self):
        for missing in (
            "column_id",
            "column_kind",
            "column_sort",
            "column_value",
            "tolerance",
        ):
            kwargs = dict(self.kwargs)
            kwargs[missing] = None
            with pytest.raises(ValueError):
                extract_features_dual_source(self.left, self.right, **kwargs)

    def test_illegal_times_fail(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            left = self.left.copy()
            left.loc[0, "time"] = bad_value
            with pytest.raises(ValueError, match="time"):
                extract_features_dual_source(left, self.right, **self.kwargs)

    def test_nat_times_fail(self):
        left = pd.DataFrame(
            {
                "id": [1, 1],
                "kind": ["temp", "temp"],
                "time": [pd.Timestamp("2020-01-01"), pd.NaT],
                "value": [1.0, 2.0],
            }
        )
        right = pd.DataFrame(
            {
                "id": [1],
                "kind": ["temp"],
                "time": [pd.Timestamp("2020-01-01")],
                "value": [1.0],
            }
        )
        with pytest.raises(ValueError, match="time"):
            extract_features_dual_source(
                left, right, **dict(self.kwargs, tolerance=pd.Timedelta("1s"))
            )

    def test_non_datetime_non_numeric_times_fail(self):
        left = self.left.copy()
        left["time"] = ["t{}".format(i) for i in range(len(left))]
        with pytest.raises(ValueError, match="time"):
            extract_features_dual_source(left, self.right, **self.kwargs)

    def test_incompatible_time_types_fail(self):
        left = self.left.copy()
        left["time"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
            left["time"], unit="s"
        )
        with pytest.raises(ValueError, match="[Ii]ncompatible"):
            extract_features_dual_source(left, self.right, **self.kwargs)

    def test_non_finite_values_fail(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            right = self.right.copy()
            right.loc[0, "value"] = bad_value
            with pytest.raises(ValueError, match="non-finite"):
                extract_features_dual_source(self.left, right, **self.kwargs)

    def test_non_numeric_values_fail(self):
        right = self.right.copy()
        right["value"] = ["high"] * len(right)
        with pytest.raises(ValueError, match="numeric"):
            extract_features_dual_source(self.left, right, **self.kwargs)

    def test_nan_in_id_or_kind_fails(self):
        left = self.left.copy()
        left.loc[0, "id"] = np.nan
        with pytest.raises(ValueError):
            extract_features_dual_source(left, self.right, **self.kwargs)
        right = self.right.copy()
        right.loc[0, "kind"] = None
        with pytest.raises(ValueError):
            extract_features_dual_source(self.left, right, **self.kwargs)

    def test_reserved_column_names_fail(self):
        with pytest.raises(ValueError, match="reserved"):
            extract_features_dual_source(
                self.left, self.right, **dict(self.kwargs, column_id="time")
            )

    def test_invalid_tolerance_fails(self):
        for tolerance in (-0.5, np.inf, np.nan):
            with pytest.raises(ValueError):
                extract_features_dual_source(
                    self.left, self.right, **dict(self.kwargs, tolerance=tolerance)
                )
