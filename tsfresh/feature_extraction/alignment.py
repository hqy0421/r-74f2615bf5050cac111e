# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
"""
This module contains the dual-source alignment feature pipeline.

Two sensors (a "left" and a "right" source) often sample the same quantities
at different, non-aligned times.  Subtracting the two sources row-by-row or
picking the nearest neighbour of every observation independently re-uses
observations and produces inconsistent difference series.

The pipeline in this module instead computes, separately for every entity
(id) and channel (kind), a one-to-one, non-crossing matching between the
observation times of the two sources:

* primary objective: maximize the number of pairs,
* secondary objective: minimize the total absolute time difference,

subject to a maximal allowed time tolerance.  The matching only looks at the
observation times, never at the measured values.

From the matched pairs a difference series (left value minus right value,
indexed by the left observation times) is built per entity and channel and
handed to the native :func:`~tsfresh.feature_extraction.extraction.extract_features`.
Every original observation ends up in exactly one pair or in the unmatched
report of its own side: unmatched observations are never padded with zero
differences and channels without pairs are not imputed with zeros.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from tsfresh import defaults
from tsfresh.feature_extraction.extraction import extract_features

# choice codes of the dynamic program in match_ordered_observations
_MATCH = 1
_SKIP_LEFT = 2
_SKIP_RIGHT = 3

#: fixed column names of the alignment and unmatched reports.  User supplied
#: id/kind column names must not collide with them.
_RESERVED_REPORT_COLUMNS = frozenset(
    [
        "left_id",
        "right_id",
        "left_time",
        "right_time",
        "left_value",
        "right_value",
        "value_diff",
        "time_diff",
        "side",
        "obs_id",
        "time",
        "value",
    ]
)

#: columns of the aligned pairs report (in addition to the id/kind columns)
ALIGNED_REPORT_COLUMNS = [
    "left_id",
    "right_id",
    "left_time",
    "right_time",
    "left_value",
    "right_value",
    "value_diff",
    "time_diff",
]

#: columns of the unmatched observations report (in addition to the id/kind columns)
UNMATCHED_REPORT_COLUMNS = ["side", "obs_id", "time", "value"]


class DualSourceResult(NamedTuple):
    """
    Result of :func:`extract_features_dual_source`.

    :cvar features: the feature matrix calculated by the native tsfresh feature
        extraction on the difference series.  The index contains the entity ids,
        the columns are named ``<kind>__<feature>`` and therefore carry the
        channel identity.  Entity/channel combinations without pairs are not
        imputed: they simply have no entries (missing values stay missing, they
        are never filled with zeros).
    :cvar aligned: the alignment report.  One row per matched pair with the id
        and kind of the entity/channel, both observation ids (``left_id``,
        ``right_id``), both times (``left_time``, ``right_time``), both values
        (``left_value``, ``right_value``), the difference ``value_diff``
        (left minus right) and the signed time difference ``time_diff``.
    :cvar unmatched: the unmatched observations report.  One row per
        observation that could not be paired, with the id and kind of the
        entity/channel, the source ``side`` ("left" or "right") and the
        observation id, time and value.  Together with ``aligned`` every
        original observation is contained exactly once.
    """

    features: pd.DataFrame
    aligned: pd.DataFrame
    unmatched: pd.DataFrame


def match_ordered_observations(left_times, right_times, tolerance):
    """
    Compute a one-to-one, non-crossing matching between two sequences of
    observation times.

    A pair ``(i, j)`` is allowed if ``abs(left_times[i] - right_times[j]) <=
    tolerance``.  Every observation takes part in at most one pair and pairs
    do not cross: if the observations of both sides are brought into time
    order, the partners of consecutive left observations are also
    consecutive.  Among all matchings the one with the largest number of
    pairs is chosen; among those, the one with the smallest total absolute
    time difference.  If several matchings are equally optimal (a tie), one
    of them is returned deterministically.

    Only the times are used for the decision, never any observed values.

    The matching is computed exactly with a dynamic program (similar to
    sequence alignment) in ``O(len(left_times) * len(right_times))`` time and
    memory.

    :param left_times: one-dimensional array-like of observation times of the
        left source.  Must be numeric (integer or float) and finite.  May be
        unordered; pairs are reported as positional indices into the given
        order.  Datetimes have to be converted to numbers (e.g. nanoseconds)
        by the caller.
    :type left_times: array-like
    :param right_times: same as ``left_times`` for the right source.
    :type right_times: array-like
    :param tolerance: maximal allowed absolute time difference of a pair.
        Must be a finite, non-negative number.  If both time arrays are
        integer typed and the tolerance is integral, the computation is done
        exactly in int64.
    :type tolerance: int or float

    :return: list of ``(left_index, right_index)`` tuples with positional
        indices into the input sequences, sorted by increasing left (and
        therefore also right) time.
    :rtype: list[tuple[int, int]]
    """
    a = _as_time_array(left_times, "left_times")
    b = _as_time_array(right_times, "right_times")

    try:
        tol = float(tolerance)
    except (TypeError, ValueError):
        raise ValueError(
            "tolerance must be a real number, got {!r}".format(tolerance)
        ) from None
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(
            "tolerance must be a finite, non-negative number, got {!r}".format(
                tolerance
            )
        )

    if a.dtype.kind in "iu" and b.dtype.kind in "iu" and tol.is_integer():
        # exact integer arithmetic (e.g. datetimes in nanoseconds)
        a = a.astype(np.int64)
        b = b.astype(np.int64)
        tol_value = np.int64(tol)
        cost_dtype = np.int64
    else:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        tol_value = tol
        cost_dtype = np.float64
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("times must be finite, found NaN or infinite values")

    # bring both sides into time order (stable, so equal times keep their
    # input order) and remember where every observation came from
    order_a = np.argsort(a, kind="stable")
    order_b = np.argsort(b, kind="stable")
    sorted_a = a[order_a]
    sorted_b = b[order_b]
    m = len(sorted_a)
    n = len(sorted_b)

    # count[i, j] / cost[i, j]: number of pairs and total absolute time
    # difference of an optimal matching of sorted_a[:i] with sorted_b[:j].
    # Optimality is lexicographic: first more pairs, then smaller cost.
    count = np.zeros((m + 1, n + 1), dtype=np.int64)
    cost = np.zeros((m + 1, n + 1), dtype=cost_dtype)
    choice = np.zeros((m + 1, n + 1), dtype=np.int8)

    for i in range(1, m + 1):
        a_i = sorted_a[i - 1]
        for j in range(1, n + 1):
            # option 1: leave left observation i-1 unmatched
            best_count = count[i - 1, j]
            best_cost = cost[i - 1, j]
            best_choice = _SKIP_LEFT
            # option 2: leave right observation j-1 unmatched
            c = count[i, j - 1]
            k = cost[i, j - 1]
            if c > best_count or (c == best_count and k < best_cost):
                best_count = c
                best_cost = k
                best_choice = _SKIP_RIGHT
            # option 3: match left i-1 with right j-1
            diff = a_i - sorted_b[j - 1]
            if diff < 0:
                diff = -diff
            if diff <= tol_value:
                c = count[i - 1, j - 1] + 1
                k = cost[i - 1, j - 1] + diff
                # on an exact tie prefer matching over leaving observations
                # unmatched (both are optimal, ties are allowed)
                if c > best_count or (c == best_count and k <= best_cost):
                    best_count = c
                    best_cost = k
                    best_choice = _MATCH
            count[i, j] = best_count
            cost[i, j] = best_cost
            choice[i, j] = best_choice

    # trace back the decisions to recover the pairs
    pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        step = choice[i, j]
        if step == _MATCH:
            pairs.append((int(order_a[i - 1]), int(order_b[j - 1])))
            i -= 1
            j -= 1
        elif step == _SKIP_LEFT:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _as_time_array(times, name):
    """
    Convert the input to a one-dimensional numpy array of numeric times.

    :param times: the input times
    :param name: name of the parameter (for error messages)
    :return: the times as numpy array
    :raise: ``ValueError`` if the input is not one-dimensional or not numeric
    """
    arr = np.asarray(times)
    if arr.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    if arr.dtype.kind not in "iuf":
        raise ValueError(
            "{} must contain numeric times, got dtype {}".format(name, arr.dtype)
        )
    return arr


def _standardize_side(
    df, side, column_id, column_kind, column_sort, column_value, column_obs_id
):
    """
    Validate one of the two input data frames and bring it into the internal
    column layout ``id, kind, time, value, obs_id, tkey``.

    ``time`` keeps the original time representation (for the reports) while
    ``tkey`` contains the numeric key used for the matching (nanoseconds for
    datetimes, float otherwise).

    :param df: the input data frame of this side
    :param side: "left" or "right" (for error messages)
    :return: tuple of the standardized data frame and the time kind
        ("datetime" or "numeric")
    :raise: ``ValueError`` for missing columns, duplicate observation ids,
        illegal times or non-finite data
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "The {} data must be a pandas DataFrame, got {}".format(side, type(df))
        )

    for column, role in (
        (column_id, "column_id"),
        (column_kind, "column_kind"),
        (column_sort, "column_sort"),
        (column_value, "column_value"),
    ):
        if column not in df.columns:
            raise ValueError(
                "Column not found in the {} data: {} ({})".format(side, column, role)
            )

    if df[column_id].isnull().any():
        raise ValueError(
            "Column must not contain NaN values: {} ({} data)".format(column_id, side)
        )
    if df[column_kind].isnull().any():
        raise ValueError(
            "Column must not contain NaN values: {} ({} data)".format(column_kind, side)
        )

    times = df[column_sort]
    if is_datetime64_any_dtype(times.dtype):
        if times.isnull().any():
            raise ValueError(
                "Column contains illegal times (NaT): {} ({} data)".format(
                    column_sort, side
                )
            )
        time_kind = "datetime"
        # normalize to nanoseconds since epoch (UTC) for an exact integer matching
        tkey = (
            pd.to_datetime(times, utc=True)
            .astype("datetime64[ns, UTC]")
            .astype("int64")
            .to_numpy()
        )
    elif is_numeric_dtype(times.dtype) and not is_bool_dtype(times.dtype):
        time_kind = "numeric"
        tkey = times.to_numpy(dtype="float64")
        if not np.isfinite(tkey).all():
            raise ValueError(
                "Column contains illegal (non-finite) times: {} ({} data)".format(
                    column_sort, side
                )
            )
    else:
        raise ValueError(
            "Column contains illegal times: {} ({} data) has dtype {}, only "
            "datetime64 and numeric times are supported".format(
                column_sort, side, times.dtype
            )
        )

    values = df[column_value]
    if (
        not is_numeric_dtype(values.dtype)
        or is_bool_dtype(values.dtype)
        or is_complex_dtype(values.dtype)
    ):
        raise ValueError(
            "Column must contain numeric data: {} ({} data) has dtype {}".format(
                column_value, side, values.dtype
            )
        )
    value_array = values.to_numpy(dtype="float64")
    if not np.isfinite(value_array).all():
        raise ValueError(
            "Column contains non-finite data (NaN or infinite): {} ({} data)".format(
                column_value, side
            )
        )

    if column_obs_id is not None:
        if column_obs_id not in df.columns:
            raise ValueError(
                "Column not found in the {} data: {} (column_obs_id)".format(
                    side, column_obs_id
                )
            )
        if df[column_obs_id].isnull().any():
            raise ValueError(
                "Column must not contain NaN values: {} ({} data)".format(
                    column_obs_id, side
                )
            )
        obs_ids = df[column_obs_id].to_numpy()
    else:
        # fall back to the position of the observation in the input data frame
        obs_ids = np.arange(len(df))

    standardized = pd.DataFrame(
        {
            "id": df[column_id].to_numpy(),
            "kind": df[column_kind].astype(str).to_numpy(),
            "time": times.to_numpy(),
            "value": value_array,
            "obs_id": obs_ids,
            "tkey": tkey,
        }
    )

    duplicated = standardized.duplicated(["id", "kind", "obs_id"])
    if duplicated.any():
        examples = standardized.loc[duplicated, ["id", "kind", "obs_id"]].head(3)
        raise ValueError(
            "Duplicate observation ids in the {} data within the same entity "
            "and channel: {}".format(
                side,
                [tuple(row) for row in examples.itertuples(index=False, name=None)],
            )
        )

    return standardized, time_kind


def _verify_conservation(
    left_std, right_std, aligned, unmatched, column_id, column_kind
):
    """
    Check the central bookkeeping invariant of the dual-source alignment:
    every original observation appears in exactly one pair or exactly once in
    the unmatched report of its own side.

    The invariant holds by construction; this function is a defensive
    verification and raises a ``RuntimeError`` if it is ever violated.
    """
    for side, std, id_column in (
        ("left", left_std, "left_id"),
        ("right", right_std, "right_id"),
    ):
        original = set(zip(std["id"], std["kind"], std["obs_id"]))
        if len(aligned) > 0:
            paired = set(
                zip(aligned[column_id], aligned[column_kind], aligned[id_column])
            )
        else:
            paired = set()
        own_unmatched = unmatched[unmatched["side"] == side]
        if len(own_unmatched) > 0:
            unpaired = set(
                zip(
                    own_unmatched[column_id],
                    own_unmatched[column_kind],
                    own_unmatched["obs_id"],
                )
            )
        else:
            unpaired = set()
        if paired & unpaired:
            raise RuntimeError(
                "internal error: observations of the {} side appear both in "
                "pairs and in the unmatched report".format(side)
            )
        if paired | unpaired != original:
            raise RuntimeError(
                "internal error: observations of the {} side got lost or were "
                "invented during alignment".format(side)
            )


def extract_features_dual_source(
    left,
    right,
    column_id=None,
    column_kind=None,
    column_sort=None,
    column_value=None,
    tolerance=None,
    column_obs_id=None,
    default_fc_parameters=None,
    kind_to_fc_parameters=None,
    output_dir=None,
    chunksize=defaults.CHUNKSIZE,
    n_jobs=defaults.N_PROCESSES,
    show_warnings=defaults.SHOW_WARNINGS,
    disable_progressbar=defaults.DISABLE_PROGRESSBAR,
    distributor=None,
):
    """
    Align two observation sources in time and extract features from their
    difference series.

    Both sources are long-format data frames with the same column layout: one
    row per observation with an entity id, a channel (kind), a time and a
    value.  For every entity and channel the observation times of the two
    sides are matched one-to-one and non-crossing by
    :func:`match_ordered_observations`: as many pairs as possible are formed,
    and among those matchings the total absolute time difference is
    minimized.  Only pairs within ``tolerance`` are allowed and the matching
    never looks at the observed values.

    Every matched pair produces one point of a difference series (left value
    minus right value), indexed by the *left* observation time.  The
    difference series of all entities and channels are handed to the native
    :func:`~tsfresh.feature_extraction.extraction.extract_features`, so the
    returned feature matrix has the entity ids as index and one column group
    ``<kind>__<feature>`` per channel.

    Bookkeeping guarantees:

    * every original observation appears in exactly one pair or exactly once
      in the unmatched report of its own side,
    * unmatched observations are never padded with zero differences,
    * entities/channels without pairs get no fabricated (zero) features -
      their feature values simply stay missing,
    * duplicate observation ids, illegal times (NaT, non-finite, unsupported
      dtype) and non-finite values raise a ``ValueError``.

    :param left: the left source as long-format pandas.DataFrame.
    :type left: pandas.DataFrame
    :param right: the right source as long-format pandas.DataFrame.
    :type right: pandas.DataFrame

    :param column_id: name of the entity id column (in both data frames).
    :type column_id: str
    :param column_kind: name of the channel (kind) column (in both data
        frames).  Kinds are compared as strings, like in the native feature
        extraction.
    :type column_kind: str
    :param column_sort: name of the time column.  Must contain datetime64 or
        finite numeric values; the dtype must be compatible on both sides.
    :type column_sort: str
    :param column_value: name of the value column.  Must contain finite
        numeric values.
    :type column_value: str

    :param tolerance: maximal allowed absolute time difference of a pair.
        A number for numeric times, a :class:`pandas.Timedelta` (or something
        parseable into one) for datetime times.
    :type tolerance: int, float or pandas.Timedelta

    :param column_obs_id: optional name of an observation id column (same
        name in both data frames).  The ids must be unique per entity and
        channel on each side and are kept in the reports.  If ``None``, the
        position of the observation in its input data frame is used.
    :type column_obs_id: str

    :param default_fc_parameters: mapping from feature calculator names to
        parameters, passed to the native feature extraction.
    :type default_fc_parameters: dict
    :param kind_to_fc_parameters: mapping from kind names to fc_parameters,
        passed to the native feature extraction.
    :type kind_to_fc_parameters: dict

    :param output_dir: if given, the three reports are saved into this
        directory (it is created if necessary): ``aligned_pairs.csv``,
        ``unmatched_observations.csv`` and ``features.csv``.
    :type output_dir: str or pathlib.Path

    :param chunksize: passed to the native feature extraction.
    :param n_jobs: passed to the native feature extraction.
    :param show_warnings: passed to the native feature extraction.
    :param disable_progressbar: passed to the native feature extraction.
    :param distributor: passed to the native feature extraction.

    :return: a :class:`DualSourceResult` with the feature matrix and the
        alignment and unmatched reports.
    :rtype: DualSourceResult
    """
    for role, column in (
        ("column_id", column_id),
        ("column_kind", column_kind),
        ("column_sort", column_sort),
        ("column_value", column_value),
    ):
        if column is None:
            raise ValueError("A value for {} needs to be supplied".format(role))
    if tolerance is None:
        raise ValueError("A value for tolerance needs to be supplied")
    reserved = _RESERVED_REPORT_COLUMNS & {column_id, column_kind}
    if reserved:
        raise ValueError(
            "The id/kind column names collide with the reserved report column "
            "names {}, please rename them".format(sorted(reserved))
        )

    left_std, left_time_kind = _standardize_side(
        left, "left", column_id, column_kind, column_sort, column_value, column_obs_id
    )
    right_std, right_time_kind = _standardize_side(
        right, "right", column_id, column_kind, column_sort, column_value, column_obs_id
    )
    if left_time_kind != right_time_kind:
        raise ValueError(
            "Incompatible time columns: the left data has {} times but the "
            "right data has {} times".format(left_time_kind, right_time_kind)
        )

    if left_time_kind == "datetime":
        try:
            tolerance_key = int(pd.Timedelta(tolerance).value)
        except (TypeError, ValueError):
            raise ValueError(
                "tolerance must be convertible to a pandas.Timedelta for "
                "datetime times, got {!r}".format(tolerance)
            ) from None
    else:
        tolerance_key = tolerance

    left_groups = {
        key: group for key, group in left_std.groupby(["id", "kind"], sort=False)
    }
    right_groups = {
        key: group for key, group in right_std.groupby(["id", "kind"], sort=False)
    }
    # deterministic iteration order over the union of both sides' groups
    all_keys = sorted(set(left_groups) | set(right_groups), key=repr)

    aligned_rows = []
    unmatched_rows = []
    for entity, kind in all_keys:
        left_group = left_groups.get((entity, kind))
        right_group = right_groups.get((entity, kind))

        if left_group is None or right_group is None:
            # the channel exists on one side only: nothing can be paired
            for side, group in (("left", left_group), ("right", right_group)):
                if group is None:
                    continue
                group = group.sort_values("tkey", kind="stable")
                for obs_id, time, value in zip(
                    group["obs_id"].to_numpy(),
                    group["time"].to_numpy(),
                    group["value"].to_numpy(),
                ):
                    unmatched_rows.append(
                        {
                            column_id: entity,
                            column_kind: kind,
                            "side": side,
                            "obs_id": obs_id,
                            "time": time,
                            "value": value,
                        }
                    )
            continue

        left_group = left_group.sort_values("tkey", kind="stable")
        right_group = right_group.sort_values("tkey", kind="stable")

        left_obs = left_group["obs_id"].to_numpy()
        left_time = left_group["time"].to_numpy()
        left_value = left_group["value"].to_numpy()
        right_obs = right_group["obs_id"].to_numpy()
        right_time = right_group["time"].to_numpy()
        right_value = right_group["value"].to_numpy()

        pairs = match_ordered_observations(
            left_group["tkey"].to_numpy(),
            right_group["tkey"].to_numpy(),
            tolerance_key,
        )

        matched_left = np.zeros(len(left_group), dtype=bool)
        matched_right = np.zeros(len(right_group), dtype=bool)
        for left_index, right_index in pairs:
            matched_left[left_index] = True
            matched_right[right_index] = True
            aligned_rows.append(
                {
                    column_id: entity,
                    column_kind: kind,
                    "left_id": left_obs[left_index],
                    "right_id": right_obs[right_index],
                    "left_time": left_time[left_index],
                    "right_time": right_time[right_index],
                    "left_value": left_value[left_index],
                    "right_value": right_value[right_index],
                    "value_diff": left_value[left_index] - right_value[right_index],
                    "time_diff": left_time[left_index] - right_time[right_index],
                }
            )

        # every observation that is not part of a pair goes into the
        # unmatched report of its own side (never padded with zero diffs)
        for index in range(len(left_group)):
            if not matched_left[index]:
                unmatched_rows.append(
                    {
                        column_id: entity,
                        column_kind: kind,
                        "side": "left",
                        "obs_id": left_obs[index],
                        "time": left_time[index],
                        "value": left_value[index],
                    }
                )
        for index in range(len(right_group)):
            if not matched_right[index]:
                unmatched_rows.append(
                    {
                        column_id: entity,
                        column_kind: kind,
                        "side": "right",
                        "obs_id": right_obs[index],
                        "time": right_time[index],
                        "value": right_value[index],
                    }
                )

    aligned = pd.DataFrame(
        aligned_rows, columns=[column_id, column_kind] + ALIGNED_REPORT_COLUMNS
    )
    unmatched = pd.DataFrame(
        unmatched_rows, columns=[column_id, column_kind] + UNMATCHED_REPORT_COLUMNS
    )

    _verify_conservation(
        left_std, right_std, aligned, unmatched, column_id, column_kind
    )

    if len(aligned) > 0:
        # the difference series: left value - right value, indexed by the
        # real left observation times
        differences = pd.DataFrame(
            {
                "id": aligned[column_id].to_numpy(),
                "kind": aligned[column_kind].to_numpy(),
                "time": aligned["left_time"].to_numpy(),
                "value": aligned["value_diff"].to_numpy(),
            }
        )
        differences.index = pd.Index(differences["time"].to_numpy())

        features = extract_features(
            differences,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value",
            default_fc_parameters=default_fc_parameters,
            kind_to_fc_parameters=kind_to_fc_parameters,
            chunksize=chunksize,
            n_jobs=n_jobs,
            show_warnings=show_warnings,
            disable_progressbar=disable_progressbar,
            # never impute: missing features of pair-less entities/channels
            # must stay missing and not be filled with zeros
            impute_function=None,
            distributor=distributor,
        )
    else:
        features = pd.DataFrame()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        aligned.to_csv(output_path / "aligned_pairs.csv", index=False)
        unmatched.to_csv(output_path / "unmatched_observations.csv", index=False)
        features.to_csv(output_path / "features.csv", index_label=column_id)

    return DualSourceResult(features=features, aligned=aligned, unmatched=unmatched)
