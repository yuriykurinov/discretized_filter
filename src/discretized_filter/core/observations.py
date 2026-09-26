"""Fine-grid Gaussian and Poisson observations of a piecewise constant state.

Channel coefficients must be time homogeneous. Functions keep the existing
``(t, y, theta)`` signature, but are evaluated at ``t=0`` here. Gaussian
variance must be a positive constant, independent of the hidden state.
"""

from dataclasses import dataclass

import numpy as np

from discretized_filter.core.densities import NORMAL, POISSON, ObsChannel


def split_channels(channels):
    """Validate genuine diffusion/counting specs and return their indices."""
    continuous, counting = [], []
    for k, ch in enumerate(channels):
        if not isinstance(ch, ObsChannel):
            raise TypeError(f"Channel {k} must be an ObsChannel")
        if any(value is not None for value in (ch.noise, ch.loc, ch.scale, ch.alpha)):
            raise ValueError(f"Channel {k}: custom noise/location-scale generators are unsupported")
        if ch.kind == NORMAL and ch.intensity is None:
            continuous.append(k)
        elif ch.kind == POISSON and ch.drift is None and ch.var is None:
            counting.append(k)
        else:
            raise ValueError(f"Channel {k}: only genuine NORMAL and POISSON channels are supported")
    return np.array(continuous, dtype=int), np.array(counting, dtype=int)


def _time_grid(times, name="times", minimum=1):
    result = np.asarray(times, dtype=float)
    if (result.ndim != 1 or result.size < minimum
            or not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0)):
        raise ValueError(f"{name} must be a finite, strictly increasing 1D grid")
    return result


def _lookup_times(stored, requested):
    """Resolve floating-point copies of existing times, never interpolate."""
    right = np.clip(np.searchsorted(stored, requested), 0, len(stored) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(stored[left] - requested) < abs(stored[right] - requested),
                       left, right)
    tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, abs(requested))
    if np.any(abs(stored[indices] - requested) > tolerance):
        raise ValueError("Sampling requires a subset of stored times; interpolation/extrapolation is unsupported")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Sampling times must resolve to distinct stored nodes")
    return indices


@dataclass
class ContinuousObservations:
    """Cumulative observations and exact counting-event metadata.

    ``continuous``/``counting`` have one row per time; ``increments`` has one
    row per interval, in the original channels order. Event channels use local
    counting indices. Samples retain absolute cumulative values, including
    when the requested range starts after zero.
    """

    times: np.ndarray
    continuous: np.ndarray
    counting: np.ndarray
    continuous_indices: np.ndarray
    counting_indices: np.ndarray
    event_times: np.ndarray
    event_channels: np.ndarray

    def __post_init__(self):
        self.times = _time_grid(self.times).copy()
        self.continuous_indices = np.asarray(self.continuous_indices, dtype=int).copy()
        self.counting_indices = np.asarray(self.counting_indices, dtype=int).copy()
        indices = np.concatenate((self.continuous_indices, self.counting_indices))
        if (self.continuous_indices.ndim != 1 or self.counting_indices.ndim != 1
                or not np.array_equal(np.sort(indices), np.arange(len(indices)))):
            raise ValueError("Channel indices must partition the original channels")
        self.continuous = np.asarray(self.continuous, dtype=float).copy()
        counts = np.asarray(self.counting, dtype=float)
        if (self.continuous.shape != (len(self.times), len(self.continuous_indices))
                or not np.all(np.isfinite(self.continuous))):
            raise ValueError("Invalid cumulative continuous array")
        if (counts.shape != (len(self.times), len(self.counting_indices))
                or not np.all(np.isfinite(counts)) or np.any(counts < 0)
                or np.any(counts != np.floor(counts)) or np.any(np.diff(counts, axis=0) < 0)):
            raise ValueError("Invalid cumulative counting array")
        self.counting = counts.astype(np.int64)
        self.event_times = np.asarray(self.event_times, dtype=float).copy()
        raw_channels = np.asarray(self.event_channels)
        if (self.event_times.ndim != 1 or raw_channels.shape != self.event_times.shape
                or not np.all(np.isfinite(self.event_times))
                or np.any(np.diff(self.event_times) < 0)
                or np.any(self.event_times < self.times[0])
                or np.any(self.event_times > self.times[-1])
                or np.any(raw_channels != np.floor(raw_channels))
                or np.any(raw_channels < 0) or np.any(raw_channels >= len(self.counting_indices))):
            raise ValueError("Invalid counting-event metadata")
        self.event_channels = raw_channels.astype(int)

    @property
    def increments(self):
        """Consecutive differences in original channel order."""
        result = np.empty((len(self.times) - 1,
                           len(self.continuous_indices) + len(self.counting_indices)))
        result[:, self.continuous_indices] = np.diff(self.continuous, axis=0)
        result[:, self.counting_indices] = np.diff(self.counting, axis=0)
        return result

    def sample(self, times, include_events=False):
        """Select existing nodes, optionally adding all events in the range."""
        requested = _time_grid(times)
        indices = _lookup_times(self.times, requested)
        selected = self.times[indices]
        event_mask = (self.event_times >= selected[0]) & (self.event_times <= selected[-1])
        if include_events:
            selected = np.unique(np.concatenate((selected, self.event_times[event_mask])))
            indices = _lookup_times(self.times, selected)
        return ContinuousObservations(
            self.times[indices], self.continuous[indices], self.counting[indices],
            self.continuous_indices, self.counting_indices,
            self.event_times[event_mask], self.event_channels[event_mask],
        )


def _coefficient(fn, state, theta, name):
    value = np.asarray(fn(0.0, state[np.newaxis, :], int(theta)), dtype=float)
    if value.shape != (1, 1) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must return a finite array with shape (1, 1)")
    return value[0, 0]


def generate_continuous_observations(t_grid, theta, y, jump_ends, channels, seed=None):
    """Generate one observed path, resolving every hidden holding segment.

    ``theta[s]`` and ``y[s]`` hold on ``[jump_ends[s-1], jump_ends[s])``;
    the first segment starts at zero. ``jump_ends`` includes the latent path's
    terminal time. The observation grid must start at zero and be covered by
    this path. Coefficients are time homogeneous (a caller contract).

    Poisson counts and sorted uniform event times are generated per hidden
    segment. All events are inserted before sampling independent Gaussian
    increments, so diffusion observations exist at event times as well.
    """
    times = _time_grid(t_grid, "t_grid", minimum=2)
    ends = _time_grid(jump_ends, "jump_ends")
    theta = np.asarray(theta)
    y = np.asarray(y, dtype=float)
    if (times[0] != 0 or ends[0] <= 0 or times[-1] > ends[-1]
            or theta.shape != ends.shape or y.ndim != 2 or y.shape[0] != len(ends)
            or y.shape[1] == 0 or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(theta)) or np.any(theta < 0)
            or np.any(theta != np.floor(theta))):
        raise ValueError("Invalid latent path or observation grid coverage")
    channels = tuple(channels)
    continuous_indices, counting_indices = split_channels(channels)
    starts = np.concatenate(([0.0], ends[:-1]))
    active = starts < times[-1]
    starts, ends = starts[active], np.minimum(ends[active], times[-1])
    theta, y = theta[active], y[active]
    drift = np.empty((len(ends), len(continuous_indices)))
    variance = np.empty_like(drift)
    intensity = np.empty((len(ends), len(counting_indices)))
    for s, (state, theta_s) in enumerate(zip(y, theta)):
        for k, original in enumerate(continuous_indices):
            ch = channels[original]
            drift[s, k] = _coefficient(ch.drift, state, theta_s, "drift")
            variance[s, k] = _coefficient(ch.var, state, theta_s, "variance")
        for r, original in enumerate(counting_indices):
            intensity[s, r] = _coefficient(channels[original].intensity, state, theta_s, "intensity")
    if (np.any(variance <= 0)
            or not np.allclose(variance, variance[:1], rtol=1e-12, atol=0)):
        raise ValueError("Gaussian variance must be positive and constant across hidden states")
    if np.any(intensity < 0):
        raise ValueError("Poisson intensity must be nonnegative")

    rng = np.random.default_rng(seed)
    event_times, event_channels = [], []
    for s, (start, end) in enumerate(zip(starts, ends)):
        for r in range(len(counting_indices)):
            number = rng.poisson(intensity[s, r] * (end - start))
            events = rng.uniform(start, end, number)
            # Zero is an RNG endpoint but not an event of a process starting at zero.
            events = np.maximum(events, np.nextafter(start, end))
            event_times.extend(events)
            event_channels.extend([r] * number)
    event_times = np.asarray(event_times, dtype=float)
    event_channels = np.asarray(event_channels, dtype=int)
    order = np.argsort(event_times, kind="stable")
    event_times, event_channels = event_times[order], event_channels[order]
    times = np.unique(np.concatenate((times, event_times)))

    # Integrate the drift analytically over holding segments, regardless of
    # how many hidden jumps fall between consecutive observation times.
    drift_prefix = np.vstack((np.zeros((1, len(continuous_indices))),
                              np.cumsum(drift * (ends - starts)[:, None], axis=0)))
    segment = np.minimum(np.searchsorted(ends, times, side="left"), len(ends) - 1)
    integrated_drift = drift_prefix[segment] + drift[segment] * (times - starts[segment])[:, None]
    increments = np.diff(integrated_drift, axis=0)
    increments += rng.standard_normal(increments.shape) * np.sqrt(np.diff(times)[:, None] * variance[0])
    continuous = np.vstack((np.zeros((1, len(continuous_indices))), np.cumsum(increments, axis=0)))
    counting = np.empty((len(times), len(counting_indices)), dtype=np.int64)
    for r in range(len(counting_indices)):
        counting[:, r] = np.searchsorted(event_times[event_channels == r], times, side="right")
    return ContinuousObservations(times, continuous, counting, continuous_indices,
                                  counting_indices, event_times, event_channels)
