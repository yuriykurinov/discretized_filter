# Реструктуризация репозитория + произвольные плотности

Дата: 2026-08-03. Ветка: `restructure`. Базовый коммит: `95a7ce1`.

Решения пользователя:
1. Папка `configs/`, вызов `set_config('config_123')`, результаты сохраняются как раньше.
2. Папки `src/`, `notebooks/`, `scripts/` + отдельный документ с предложениями по улучшению.
3. Произвольные плотности — **и** плотность приращения наблюдений (сейчас жёстко `norm`
   в `filter.py`), **и** условная плотность `π(y|θ)`.
4. API конфига: глобальный синглтон + `import *`.
5. Ноутбуки переезжают в `notebooks/`, импорты переписываются.

Авторитетный вывод формул: `article/my_notes/discretized_filter.tex`.

---

## 0. Целевая структура

```
/home/yk/discretized_filter/
├── configs/
│   ├── __init__.py
│   └── example_for_arc_new.py         # порт текущего config.py
├── src/discretized_filter/
│   ├── __init__.py
│   ├── paths.py                       # PROJECT_ROOT, saved_path_dir(exp_id)
│   ├── config.py                      # set_config / get_config / поддержка import *
│   ├── core/
│   │   ├── __init__.py
│   │   ├── densities.py               # плотности приращения наблюдений
│   │   ├── filter.py                  # ядра + класс Filter
│   │   └── smjp.py                    # sparse_mc, make_discretized_*
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── grids.py                   # cartesian_product, get_moments, to_discrete, ...
│   │   ├── distributions.py           # реестр семейств π(y|θ)
│   │   └── io.py                      # save_path, load_saved_path, save_config_copy
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
├── scripts/
│   ├── _bootstrap.py
│   ├── run_single_path.py
│   ├── run_monte_carlo.py
│   ├── filter.job
│   └── monte-carlo.job
├── notebooks/                          # пять существующих ноутбуков
├── plans/  docs/  article/
├── pyproject.toml
└── saved_path_*/  results/             # НЕ трогать
```

Окружение: `.venv` (Python 3.14.6, numba 0.65.1, numpy 2.4.6, scipy 1.17.1), менеджер `uv`.
Новых зависимостей не добавлять.

Пустой каталог `discretized_filter/` в корне (только `__pycache__`, не под git) — удалить.

---

## 1. Плотность приращения наблюдений (ядро, требует аккуратности)

### 1.1 Что есть сейчас

`F[m, y] ∈ R^K` — скорости сноса, `G[m, y] ∈ R^K` — скорости дисперсии.
Правдоподобие приращения на траектории с временами пребывания `u_j` в состояниях `(i_j, y_j)`:

```
∏_k  N( obs_k ;  Σ_j u_j F[i_j, y_j, k] ,  Σ_j u_j G[i_j, y_j, k] )
```

Это формулы (transition_kernel:1–2) из `discretized_filter.tex`, где
`F(Z) = Σ_j u_j f(θ, Y)`, `G(Z) = Σ_j u_j gg^T(θ, Y)`.

Ключевое свойство, на котором держится вся схема: **параметры плотности накапливаются
линейно по временам пребывания**. Нормальное распределение — частный случай (P = 2
параметра). Это верно для любого безгранично делимого приращения.

### 1.2 Обобщение

Заменить `F`, `G` одним массивом характеристик

```
C : float64[N, n_grid, K, P]      # C[n, y, k, p] — скорость p-го параметра k-го канала
```

Закон приращения по каналу `k` определяется вектором `Σ_j u_j C[i_j, y_j, k, :] ∈ R^P`.

Плотность — джитованная функция

```python
obs_density(obs: float64[:], params: float64[:, :]) -> float64
#   obs.shape    == (K,)
#   params.shape == (K, P)
#   возвращает совместную плотность приращения ∏_k p_k(obs_k ; params[k])
```

Встроенные виды каналов (`core/densities.py`):

| код | вид | используемые слоты | плотность |
|---|---|---|---|
| 0 | `NORMAL` | 2: (f, gg^T) | `N(x; θ0, θ1) = exp(-(x-θ0)²/(2θ1)) / sqrt(2π θ1)` |
| 1 | `POISSON` | 1: (интенсивность) | `exp(x·log θ0 − θ0 − lgamma(x+1))` |

`P = max_k (число слотов канала k)`, неиспользуемые слоты — нули.

Фабрика: `make_obs_density(kinds)` → джитованное замыкание по кортежу целых кодов.
Пользователь может передать **свою** джитованную функцию с той же сигнатурой —
это и есть «произвольная плотность».

Замечание по текущему коду: канал `η` (считающий, `deta = np.random.poisson(...)`)
сейчас обрабатывается нормальной плотностью со средним = дисперсией = `h·h(y)`,
т.е. гауссовской аппроксимацией пуассоновского. После обобщения его следует
объявить как `POISSON`.

### 1.3 Исправляемая ошибка в `zero_jump_kernel`

Сейчас:

```python
log_res = (ht * lam[m] - 0.5*np.log(np.pi*sigma_sq_2)) - (ht*F[m,y] - obs)**2 / sigma_sq_2
return np.exp(np.sum(log_res))
```

`log_res` — вектор длины `K`, поэтому член `ht·λ_mm` суммируется **K раз**:
получается `exp(K·h·λ_mm)` вместо `exp(h·λ_mm)` из (transition_kernel:1).
Множитель зависит от `m`, значит искажает относительные веса состояний.
Для `K = 1` (конфиг `ia_example`) ошибка неактивна; для `K = 2` — активна.
`integrand` / `integrand2` этой ошибки не содержат: там экспонента вынесена из `np.prod`.

Правильно:

```
zero_jump = exp(ht * lam[m]) * obs_density(obs, ht * C[m, y])
```

### 1.4 Новые сигнатуры ядер

```
zero_jump_kernel(m, y, obs, ht, C, lam, obs_density, buf)
integrand(tau, m, y, n, v, obs, pi, C, Lambda, ht, obs_density, buf)
single_jump_kernel(m, y, n, v, obs, ht, C, Lambda, pi, method, n_points, obs_density, buf)
integrand2(tau1, tau2, m, y, k, z, n, v, obs, pi, C, Lambda, ht, obs_density, buf)
double_jump_kernel(m, y, n, v, obs, ht, C, Lambda, pi, method, n_points, delta, obs_density, buf)
filter_step(psi, obs, C, Lambda, lam, pi, ht, delta, N, n_points, two_jumps, obs_density)
```

Смешивание параметров остаётся ровно тем же, что и было для `F`/`G`:
`params = tau*C[n,v] + (ht-tau)*C[m,y]` (и аналогично с двумя скачками).

**Производительность.** `tau*C[n,v] + (ht-tau)*C[m,y]` во внутреннем цикле — это
аллокация массива `(K, P)` на каждый вызов. Так делать нельзя. Выделять буфер
`buf = np.empty((K, P))` **один раз на итерацию `nb.prange` по `y`** и прокидывать
его вниз по цепочке вызовов; смешивание писать явными циклами по `k, p` в `buf`.

**Компиляция.** Явные eager-сигнатуры `@nb.njit(nb.float64(...))` придётся снять —
`obs_density` приходит как dispatcher. Оставить `fastmath=True`, `parallel=True`
на `filter_step`, ленивую компиляцию. Замерить время шага фильтра до/после
и указать в отчёте.

### 1.5 Критерии приёмки

- Конфиг с одним нормальным каналом (`K = 1`): новый `filter_step` совпадает со
  старым (коммит `95a7ce1`) на 100 шагах фильтрации с точностью `1e-12`
  по `theta_est` и `y_est`. При `K = 1` ошибка 1.3 неактивна, так что совпадение
  должно быть точным.
- `K = 2`, оба канала `NORMAL`: результат отличается от старого ровно на
  исправление 1.3 — проверить, умножив старое ядро на `exp(-ht*lam[m])`.
- Пуассоновский канал: `make_obs_density((POISSON,))` на сетке `x = 0..20`,
  `θ0 = 3.7` суммируется в 1 с точностью `1e-12`.
- Нормальный канал: интеграл по `x` численно равен 1.

---

## 2. Произвольная π(y|θ)

`src/discretized_filter/utils/distributions.py`.

Сейчас `get_distributions` в `utils.py` строит `pi_uniform`, остальные варианты
(треугольные, арксинус, 3-точечное) закомментированы; «зависимое» распределение
собрано вручную в `config.py` (`gamma_d`, `cond_d`), а парный ему генератор
`get_y_dependent` живёт в ячейке `filtering.ipynb`. Это ровно тот случай, когда
плотность в фильтре и генератор в модели расходятся незаметно.

### 2.1 Интерфейс

```python
@dataclass
class PiFamily:
    name: str
    pdf_grid: Callable       # (state, M_net_n, y_intervals) -> np.ndarray (n_grid,), ненормированная
    sampler: Callable        # njit: (state, y_intervals) -> кортеж/список из M чисел
    normalize_on_grid: bool  # нормировать ли по сетке
```

`pdf_grid` работает с **совместной** плотностью на сетке `M_net[n]` формы `(n_grid, M)`,
поэтому поддерживаются и неразделимые по координатам распределения (как «зависимое»).
Для разделимых — хелпер `separable(pdf_1d)`, собирающий произведение по координатам.

Реестр: `register_pi_family(name)` (декоратор) + `get_pi_family(name)`.
В конфиге можно указать имя строкой либо передать свой объект `PiFamily` —
это и есть «произвольная плотность».

### 2.2 Нормировка

`filter_step` интегрирует как `Σ_y ... * delta[n]`, где `delta[n]` — произведение
шагов по координатам. Поэтому:

- `normalize_on_grid = True` → `pdf /= pdf.sum() * delta[n]` (как сейчас делают
  `triang`, `arcsine_dist_pdf`);
- `normalize_on_grid = False` → аналитическая нормировка (равномерное: `1/(b-a)`
  по каждой координате) — **оставить именно так для `uniform`**, иначе значения
  разойдутся с текущими и сломается проверка воспроизводимости из п. 1.5.

### 2.3 Встроенные семейства

`uniform` (аналитическая нормировка), `triangular` (симметричное),
`triangular_right`, `arcsine`, `three_point`, `dependent_gamma`
(перенести `gamma_d` + `cond_d` из `config.py` и `get_y_dependent` из
`filtering.ipynb` — плотность и генератор в одном месте).

Заодно исправить в перенесённом коде: дублирующееся определение `unif`
в `utils.py` и несогласованные формы `(N, M_net.shape[0])` против
`(N, M_net.shape[1])` в `get_distributions` (правильно `shape[1]` — число узлов).

### 2.4 Критерий приёмки

Для каждого встроенного семейства: сгенерировать `10^6` выборок джитованным
`sampler`, построить гистограмму по той же сетке, сравнить с `pdf_grid`.
Относительная ошибка в L1 < 5 %. Это прямая защита от расхождения
«плотность в фильтре ↔ генератор траекторий». Тест-скрипт — во временную папку,
в репозиторий не коммитить.

---

## 3. Система конфигов

### 3.1 Файл конфига (`configs/<name>.py`) — только «сырые» параметры

```
exp_id, T, ht, seed, N, Lambda (без диагонали), y_intervals, num_nodes,
channels (список спецификаций каналов наблюдения), pi_family,
n_points, two_jumps, опционально p0
```

Спецификация канала:

```python
ObsChannel(kind='normal', drift=f_jitted, var=gg_jitted)     # P-слоты (f, gg^T)
ObsChannel(kind='poisson', intensity=h_jitted)               # P-слот (h)
```

`f`, `gg`, `h` — джитованные функции `(t, y, theta) -> float64[:, :]`, как сейчас
`g`, `sigma`, `h` в `config.py`.

### 3.2 Вычисляемые величины (строит `src/discretized_filter/config.py`)

`M`, `K`, `P`, `t_net_filtering`, `p0` (стационарное из `Lambda`, если не задано),
`lam`, `Lam`, `nets`, `M_net`, `delta`, `deltas`, `pi`, `pi_init`, `C`,
`obs_density`, `get_y`, `get_obs`.

### 3.3 API

```python
from discretized_filter.config import set_config
set_config('example_for_arc_new')      # или 'example_for_arc_new.py', или путь
from discretized_filter.config import *   # N, T, ht, M_net, C, pi, ...
```

`set_config(name)`:
1. находит `PROJECT_ROOT/configs/<name>.py`, грузит через
   `importlib.util.spec_from_file_location`;
2. сеет все ГПСЧ (`np.random.seed`, `default_rng`, numba `set_seed`);
3. считает производные величины;
4. кладёт всё в `globals()` модуля и выставляет динамический `__all__`;
5. сохраняет в `_ACTIVE`, возвращает объект конфига.

`get_config()` — с внятной ошибкой, если конфиг не установлен.
Поддержать переменную окружения `DFILTER_CONFIG` (для скриптов/кластера).

### 3.4 Сохранение результатов — **без изменений**

`save_path(exp_id, ...)` → `PROJECT_ROOT/saved_path_{exp_id}/{theta,y,t,theta_est,y_est,observations}.pkl`.
`load_saved_path(exp_id)` читает оттуда же. Пути резолвить от `PROJECT_ROOT`
(`paths.py`), а не от `cwd` — иначе ноутбуки из `notebooks/` не найдут данные.
Старые каталоги `saved_path_*` должны читаться как раньше.

`save_config_copy(exp_id)` копирует **активный файл конфига** в
`saved_path_{exp_id}/config.py` (имя файла сохранить прежним для совместимости),
добавляя в шапку имя конфига и метку времени.

---

## 4. Скрипты и ноутбуки

`scripts/run_single_path.py`, `scripts/run_monte_carlo.py`:
аргумент `--config <имя>` (по умолчанию — `DFILTER_CONFIG`), у второго ещё
`--paths` (по умолчанию 10000). Логика — из `single_path_for_server.py` и
`multiple_paths_for_server.py` без изменений по существу.
`.job`-файлы обновить под новые пути.

Ноутбуки перенести `git mv` в `notebooks/` и переписать ячейки импорта.
`rmse_plots.ipynb` открывает `'saved_path_' + id + '/rmse_...'` напрямую —
перевести на хелпер путей. В `filtering.ipynb` убрать локальный
`get_y_dependent`, взять семейство из реестра.

Установка: `pyproject.toml` + `uv pip install -e .` в `.venv`.
Плюс `_bootstrap.py` (ищет корень по `pyproject.toml` и добавляет `src` в
`sys.path`) как запасной вариант для кластера без установки.
