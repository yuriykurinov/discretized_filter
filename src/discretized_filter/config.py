"""
Система конфигов эксперимента: ``set_config`` / ``get_config``.

Конфиг -- модуль в ``configs/<name>.py``, описывающий "сырые" параметры
эксперимента (``_RAW_NAMES``, ``channels`` -- список ``core.densities.
ObsChannel``); ``set_config`` грузит его и строит производные величины
(сетки, ``C``, ``obs_density``, ``pi``, ``get_obs`` и т.д.), публикуя их
как атрибуты возвращаемого объекта и в ``globals()`` этого модуля.
"""
import importlib.util
import os
import sys
import types
from math import ceil
from pathlib import Path

import numpy as np
import numba as nb

from discretized_filter.paths import CONFIGS_DIR
from discretized_filter.utils.grids import set_seed, cartesian_product
from discretized_filter.utils.distributions import get_pi_family, build_pi
from discretized_filter.core.densities import (
    NORMAL, POISSON, PARETO, EXPONENTIAL, UNIFORM,
    n_params_for, make_obs_density,
)
from discretized_filter.core.filter import filter_step as generic_filter_step
from discretized_filter.core.filter_normal import filter_step_normal
from discretized_filter.core.observations import generate_continuous_observations
from discretized_filter.core.smjp import (
    make_discretized_xi, make_discretized_eta, make_xi_generator,
    make_discretized_pareto, make_discretized_exponential,
    make_discretized_uniform,
)


# Обязательные "сырые" имена, которые должен определять файл конфига.
_RAW_NAMES = (
    'exp_id', 'T', 'ht', 'seed', 'N', 'Lambda', 'y_intervals', 'num1',
    'pi_family', 'channels', 'n_points', 'two_jumps',
)

# Производные имена, которые строит сборщик; вместе с _RAW_NAMES образуют
# публичный набор атрибутов активного конфига (без ведущего "_").
_DERIVED_NAMES = (
    'M', 'K', 'P', 't_net_filtering', 'p0', 'lam', 'Lam', 'nets', 'M_net',
    'delta', 'deltas', 'pi', 'pi_init', 'C', 'obs_density', 'get_y',
    'get_obs', 'rng', 'num_nodes', 'shared_grid', 'filter_step',
    'get_continuous_obs', 'continuous_indices', 'counting_indices',
)

_PUBLIC_NAMES = _RAW_NAMES + _DERIVED_NAMES

_ACTIVE = None

__all__ = ['set_config', 'get_config']


def _resolve_config_path(name):
    """Резолвит имя конфига ('foo', 'foo.py' или путь) в файл; имя без
    расширения ищется относительно CONFIGS_DIR."""
    p = Path(name)
    if p.suffix != '.py':
        p = p.with_name(p.name + '.py')

    if p.is_file():
        return p.resolve()

    candidate = CONFIGS_DIR / p.name
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f'конфиг не найден: {name!r} (искали {p} и {candidate})'
    )


def _load_config_module(path):
    """Загружает модуль конфига из файла -- заново исполняет файл при каждом
    вызове (никакого протухшего состояния между конфигами), но регистрирует
    модуль в sys.modules под именем, производным от файла: без этого джитованные
    функции конфига с cache=True не находят свой модуль при перекомпиляции
    (numba пишет имя модуля в кэш на диске и разрешает его через
    importlib.import_module при следующей загрузке)."""
    module_name = f'discretized_filter._configs.{path.stem}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_std_fn(var_fn):
    """
    Оборачивает var(t, y, theta) (дисперсию gg^T) в СКО для
    make_discretized_xi, которая ждёт функцию СКО и возводит её в квадрат
    сама (см. core/smjp.py: S[t] = sigma(...)**2).
    """
    @nb.njit(nogil=True)
    def std_fn(t, y, theta):
        return np.sqrt(var_fn(t, y, theta))
    return std_fn


def _build_get_obs(channels):
    """
    Строит generic get_obs(t_net_filtering, theta, y, t) по списку каналов.

    Диспетчеризация учитывает порождающий процесс: ``intensity`` --
    считающий процесс (make_discretized_eta); exact kind -- location-scale смесь
    по временам пребывания; иначе непрерывный снос/диффузия
    (make_discretized_xi либо, если задан ``ch.noise``, make_xi_generator).
    Приращения складываются в столбцы результата в порядке channels,
    с отбрасыванием индекса 0 (момент t=0, приращение всегда 0).
    """
    std_fns = tuple(
        _make_std_fn(ch.var) if (ch.intensity is None and ch.loc is None) else None
        for ch in channels
    )
    xi_fns = tuple(
        None if (ch.intensity is not None or ch.loc is not None)
        else (make_discretized_xi if ch.noise is None else make_xi_generator(ch.noise))
        for ch in channels
    )

    def get_obs(t_net_filtering, theta, y, t):
        cols = []
        for ch, std_fn, xi_fn in zip(channels, std_fns, xi_fns):
            if ch.intensity is not None:
                d = make_discretized_eta(t_net_filtering, ch.intensity, theta, y, t)
                cols.append(d[1:])
            elif ch.kind == EXPONENTIAL:
                d = make_discretized_exponential(
                    t_net_filtering, ch.loc, ch.scale, theta, y, t)
                cols.append(d[1:])
            elif ch.kind == UNIFORM:
                d = make_discretized_uniform(
                    t_net_filtering, ch.loc, ch.scale, theta, y, t)
                cols.append(d[1:])
            elif ch.loc is not None:
                d = make_discretized_pareto(
                    t_net_filtering, ch.loc, ch.scale, ch.alpha, theta, y, t)
                cols.append(d[1:])
            else:
                d = xi_fn(t_net_filtering, ch.drift, std_fn, theta, y, t, 1)
                cols.append(d[1:, 0])
        return np.stack(cols, axis=-1)

    return get_obs


def _build(module, path):
    """Строит производные величины конфига из загруженного модуля."""
    for name in _RAW_NAMES:
        if not hasattr(module, name):
            raise AttributeError(
                f'конфиг {path} не определяет обязательный параметр {name!r}'
            )

    exp_id = module.exp_id
    T = module.T
    ht = module.ht
    seed = module.seed
    N = module.N
    num1 = module.num1
    n_points = module.n_points
    two_jumps = module.two_jumps
    pi_family = module.pi_family
    channels = list(module.channels)
    y_intervals = list(module.y_intervals)

    M = len(y_intervals)
    K = len(channels)
    P = max(n_params_for(ch.kind) for ch in channels)

    # Lambda задаётся только внедиагональными интенсивностями; диагональ
    # строится как lambda_ii = -sum_{j != i} lambda_ij.
    Lambda = np.array(module.Lambda, dtype=np.float64, copy=True)
    np.fill_diagonal(Lambda, 0.0)
    for i in range(Lambda.shape[0]):
        Lambda[i, i] = -np.sum(Lambda[i])

    lam = np.diagonal(Lambda).copy()
    Lam = Lambda - np.diag(lam)

    t_net_filtering = np.array([tt * ht for tt in range(ceil(T / ht))])

    p0 = getattr(module, 'p0', None)
    if p0 is None:
        # стационарное распределение генератора Lambda: левый нуль-вектор
        # Lambda^T (последняя строка V^T в SVD Lambda^T = U S V^T)
        _, _, vh = np.linalg.svd(Lambda.T)
        p0 = vh[-1].copy()
    else:
        p0 = np.array(p0, dtype=np.float64, copy=True)
    p0 = p0 / np.sum(p0)

    # num1 -- либо одно число узлов на все координаты Y, либо своё число на
    # каждую координату (в старых конфигах было num1=20, num2=100).
    num_nodes = [int(num1)] * M if np.isscalar(num1) else [int(v) for v in num1]
    if len(num_nodes) != M:
        raise ValueError(
            f'num1: ожидалось одно число или {M} значений (по числу координат '
            f'Y), получено {len(num_nodes)}'
        )

    # shared_grid: одна общая сетка на все состояния, построенная по
    # объединению носителей, вместо своей сетки на каждое состояние.
    # Так устроены старые конфиги (см. results/example_for_ia/config.py);
    # носитель pi при этом всё равно свой у каждого состояния -- он задаётся
    # y_intervals, а вне него плотность зануляется.
    shared_grid = bool(getattr(module, 'shared_grid', False))

    if shared_grid:
        bounds = [
            (np.min(y_intervals[i]), np.max(y_intervals[i])) for i in range(M)
        ]
        steps = [
            (b - a) / (num_nodes[i] - 1) for i, (a, b) in enumerate(bounds)
        ]
        nets = [
            [
                np.linspace(a, b, num=num_nodes[i])
                for i, (a, b) in enumerate(bounds)
            ]
            for _ in range(N)
        ]
        delta = np.full(N, np.prod(steps))
        deltas = list(steps)
    else:
        nets = [
            [
                np.linspace(
                    y_intervals[i][n, 0], y_intervals[i][n, 1], num=num_nodes[i]
                )
                for i in range(M)
            ]
            for n in range(N)
        ]
        delta = np.array([
            np.prod([
                (y_intervals[i][n, 1] - y_intervals[i][n, 0]) / (num_nodes[i] - 1)
                for i in range(M)
            ])
            for n in range(N)
        ])
        # deltas -- шаг по каждой координате, взятый по состоянию 0 (как в
        # прежнем config.py: delta1/delta2 считались только по первой строке
        # y1_intervals/y2_intervals).
        deltas = [
            (y_intervals[i][0, 1] - y_intervals[i][0, 0]) / (num_nodes[i] - 1)
            for i in range(M)
        ]

    M_net = np.array([cartesian_product(net) for net in nets])
    n_grid = M_net.shape[1]

    pi = build_pi(pi_family, N, M_net, nets, y_intervals, delta)
    pi_init = p0[:, np.newaxis] * pi

    C = np.zeros((N, n_grid, K, P))
    for n in range(N):
        for k, ch in enumerate(channels):
            if ch.kind == NORMAL:
                C[n, :, k, 0] = ch.drift(-1, M_net[n], n)[:, 0]
                C[n, :, k, 1] = ch.var(-1, M_net[n], n)[:, 0]
            elif ch.kind == POISSON:
                C[n, :, k, 0] = ch.intensity(-1, M_net[n], n)[:, 0]
            elif ch.kind == PARETO:
                C[n, :, k, 0] = ch.loc(-1, M_net[n], -1)[:, 0]
                C[n, :, k, 1] = ch.scale(-1, M_net[n], -1)[:, 0]
                C[n, :, k, 2] = ch.alpha(-1, M_net[n], -1)[:, 0]
            elif ch.kind in (EXPONENTIAL, UNIFORM):
                C[n, :, k, 0] = ch.loc(-1, M_net[n], -1)[:, 0]
                C[n, :, k, 1] = ch.scale(-1, M_net[n], -1)[:, 0]
            else:
                raise ValueError(f'неизвестный вид канала наблюдения: {ch.kind}')

    # normal_pdf не защищена от нулевой дисперсии, pareto_obs_pdf -- от
    # alpha <= 2 (дисперсия шума бесконечна). Проверяем по ВСЕЙ сетке, а не
    # только там, где pi > 0: zero_jump_kernel вычисляется для каждого узла
    # безусловно, а 0.0 * NaN == NaN.
    for k, ch in enumerate(channels):
        if ch.kind == NORMAL:
            bad = ~(C[:, :, k, 1] > 0)
            if np.any(bad):
                n_bad = int(bad.sum())
                raise AssertionError(
                    f'канал {k} (NORMAL): дисперсия <= 0 в {n_bad} узлах сетки '
                    f'(из {bad.size}); normal_pdf не защищена от нулевой '
                    'дисперсии и даст NaN'
                )
        elif ch.kind == PARETO:
            bad_scale = ~(C[:, :, k, 1] > 0)
            bad_alpha = ~(C[:, :, k, 2] > 2.0)
            if np.any(bad_scale) or np.any(bad_alpha):
                raise AssertionError(
                    f'канал {k} (PARETO): scale <= 0 в {int(bad_scale.sum())} '
                    f'узлах, alpha <= 2 в {int(bad_alpha.sum())} узлах сетки '
                    f'(из {bad_scale.size}); pareto_obs_pdf вырождается в '
                    'тождественный ноль в обоих случаях'
                )
        elif ch.kind in (EXPONENTIAL, UNIFORM):
            bad_scale = ~(C[:, :, k, 1] > 0)
            if np.any(bad_scale):
                name = 'EXPONENTIAL' if ch.kind == EXPONENTIAL else 'UNIFORM'
                raise AssertionError(
                    f'канал {k} ({name}): scale <= 0 в '
                    f'{int(bad_scale.sum())} узлах сетки '
                    f'(из {bad_scale.size}); location-scale плотность '
                    'вырождается в тождественный ноль'
                )

    obs_density = make_obs_density(tuple(ch.kind for ch in channels))

    # гауссовский случай устойчив: параметры складываются линейно, поэтому
    # работает специализированное ядро без буферов и вызова obs_density
    filter_step = (
        filter_step_normal if all(ch.kind == NORMAL for ch in channels)
        else generic_filter_step
    )

    fam = get_pi_family(pi_family)
    get_y = fam.sampler

    get_obs = _build_get_obs(channels)

    def get_continuous_obs(t_grid, theta, y, jump_ends, seed=None):
        """Генерирует непрерывную траекторию для гауссовских/пуассоновских каналов."""
        return generate_continuous_observations(
            t_grid, theta, y, jump_ends, channels, seed=seed,
        )

    continuous_indices = np.array([k for k, ch in enumerate(channels) if ch.kind == NORMAL], dtype=int)
    counting_indices = np.array([k for k, ch in enumerate(channels) if ch.kind == POISSON], dtype=int)

    rng = np.random.default_rng(seed=seed)

    cfg = types.SimpleNamespace(
        exp_id=exp_id, T=T, ht=ht, seed=seed, N=N, Lambda=Lambda,
        y_intervals=y_intervals, num1=num1, pi_family=pi_family,
        channels=channels, n_points=n_points, two_jumps=two_jumps,
        M=M, K=K, P=P, t_net_filtering=t_net_filtering, p0=p0, lam=lam,
        Lam=Lam, nets=nets, M_net=M_net, delta=delta, deltas=deltas,
        pi=pi, pi_init=pi_init, C=C, obs_density=obs_density, get_y=get_y,
        get_obs=get_obs, rng=rng, num_nodes=num_nodes, shared_grid=shared_grid,
        filter_step=filter_step,
        get_continuous_obs=get_continuous_obs,
        continuous_indices=continuous_indices, counting_indices=counting_indices,
    )
    # метаданные для save_config_copy -- не публикуются через globals()/__all__
    cfg._config_path = path
    cfg._config_name = path.stem
    return cfg


def set_config(name=None):
    """Устанавливает активный конфиг и публикует его величины в этот модуль
    (``name`` по умолчанию берётся из DFILTER_CONFIG)."""
    global _ACTIVE

    if name is None:
        name = os.environ.get('DFILTER_CONFIG')
        if name is None:
            raise ValueError(
                'set_config: имя конфига не передано, и переменная '
                'окружения DFILTER_CONFIG не установлена'
            )

    path = _resolve_config_path(name)
    module = _load_config_module(path)

    # сеем все ГПСЧ до того, как что-либо случайное могло произойти:
    # numpy (глобальный RandomState), numpy Generator и джитованный ГПСЧ
    # numba (у него отдельное состояние).
    seed = module.seed
    np.random.seed(seed)
    set_seed(seed)

    cfg = _build(module, path)

    # убрать протухшие имена предыдущего конфига
    for stale_name in globals().get('__all__', ()):
        if stale_name not in ('set_config', 'get_config'):
            globals().pop(stale_name, None)

    public = {name_: getattr(cfg, name_) for name_ in _PUBLIC_NAMES}
    globals().update(public)
    globals()['__all__'] = list(_PUBLIC_NAMES) + ['set_config', 'get_config']

    _ACTIVE = cfg
    return cfg


def get_config():
    """Возвращает активный конфиг; ошибка с понятным сообщением, если
    ``set_config`` ещё не был вызван."""
    if _ACTIVE is None:
        raise RuntimeError(
            'конфиг не установлен: вызовите '
            'discretized_filter.config.set_config(name) перед использованием'
        )
    return _ACTIVE
