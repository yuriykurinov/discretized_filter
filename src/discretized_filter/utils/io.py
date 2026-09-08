import pickle
import os

from discretized_filter.paths import saved_path_dir
from discretized_filter.config import get_config


def load_saved_path(exp_id):
    exp_path = saved_path_dir(exp_id)

    with open(os.path.join(exp_path, 't.pkl'), 'rb') as f:
        t = pickle.load(f)
    with open(os.path.join(exp_path, 'theta.pkl'), 'rb') as f:
        theta = pickle.load(f)
    with open(os.path.join(exp_path, 'y.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(exp_path, 'theta_est.pkl'), 'rb') as f:
        theta_est = pickle.load(f)
    with open(os.path.join(exp_path, 'y_est.pkl'), 'rb') as f:
        y_est = pickle.load(f)
    with open(os.path.join(exp_path, 'observations.pkl'), 'rb') as f:
        observations = pickle.load(f)
    # with open(os.path.join(exp_path, 'dxi.pkl'), 'rb') as f:
    #     dxi = pickle.load(f)
    # with open(os.path.join(exp_path, 'deta.pkl'), 'rb') as f:
    #     deta = pickle.load(f)
    return theta, y, t, theta_est, y_est, observations

# копия активного конфига с метаданными
def save_config_copy(exp_id):
    config_content = None
    exp_path = saved_path_dir(exp_id)
    os.makedirs(exp_path, exist_ok=True)

    config_name = None
    try:
        cfg = get_config()
        config_file_path = cfg._config_path
        config_name = cfg._config_name
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
    except Exception as e:
        print(f"Не удалось сохранить config.py: {e}")

    if config_content:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        header = f"""# Копия конфига для эксперимента {exp_id}\n# Имя конфига: {config_name}\n# Сохранено: {timestamp}\n# {'='*60}\n"""
        config_save_path = os.path.join(exp_path, 'config.py')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            f.write(header + config_content)

def save_path(exp_id, theta, y, t, theta_est, y_est, observations):
    exp_path = saved_path_dir(exp_id)
    os.makedirs(exp_path, exist_ok=True)

    with open(os.path.join(exp_path, 'theta_est.pkl'), 'wb') as f:
        pickle.dump(theta_est, f)
    with open(os.path.join(exp_path, 'y_est.pkl'), 'wb') as f:
        pickle.dump(y_est, f)
    with open(os.path.join(exp_path, 'theta.pkl'), 'wb') as f:
        pickle.dump(theta, f)
    with open(os.path.join(exp_path, 'y.pkl'), 'wb') as f:
        pickle.dump(y, f)
    with open(os.path.join(exp_path, 't.pkl'), 'wb') as f:
        pickle.dump(t, f)
    with open(os.path.join(exp_path, 'observations.pkl'), 'wb') as f:
        pickle.dump(observations, f)

    print(f"saved to {exp_path}")
