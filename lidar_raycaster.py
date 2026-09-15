import numpy as np
import mujoco

# Параметры сканирования.
# H_RAYS - количество горизонтальных лучей, один луч на градус, полное кольцо 360.
# V_LAYERS - количество вертикальных слоёв, как у Velodyne VLP-16.
# V_FOV_DEG - полный вертикальный угол обзора в градусах, от -15 до +15.
# MAX_RANGE - максимальная дальность в метрах, лучи дальше этого расстояния отбрасываются.

H_RAYS     = 360
V_LAYERS   = 16
V_FOV_DEG  = 30.0
MAX_RANGE  = 30.0

_V_ANGLES_DEG = np.linspace(-V_FOV_DEG / 2, V_FOV_DEG / 2, V_LAYERS)
_H_ANGLES_DEG = np.linspace(0.0, 360.0, H_RAYS, endpoint=False)


def _build_ray_directions_local():
    # Строим матрицу направлений лучей в локальной системе координат лидара.
    # Локальная СК: +X вперёд, +Y влево, +Z вверх.
    # Возвращает массив shape (V_LAYERS * H_RAYS, 3) - единичные векторы.

    v_rad = np.deg2rad(_V_ANGLES_DEG)
    h_rad = np.deg2rad(_H_ANGLES_DEG)

    cos_v = np.cos(v_rad)
    sin_v = np.sin(v_rad)
    cos_h = np.cos(h_rad)
    sin_h = np.sin(h_rad)

    # Broadcast (V, 1) по (1, H) даёт (V, H) для каждой компоненты
    dx = cos_v[:, None] * cos_h[None, :]
    dy = cos_v[:, None] * sin_h[None, :]
    dz = sin_v[:, None] * np.ones(H_RAYS)

    directions = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms


# Предвычисляем один раз при импорте модуля.
# Направления лучей не меняются между сканами - только ориентация лидара меняется.
_RAY_DIRS_LOCAL = _build_ray_directions_local()


class LidarRaycaster:
    # Виртуальный лидар, привязанный к именованному site в MuJoCo-модели.
    # site_name - имя site из XML, откуда испускаются лучи.
    # exclude_body - индекс тела, которое исключается из трассировки.
    # Обычно передаём индекс torso_link, чтобы лидар не видел собственную геометрию робота.

    def __init__(self, model, data, site_name="lidar_site", exclude_body=None):
        self.model = model
        self.data  = data

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(
                "Site '{}' не найден в модели. Проверьте имя в XML.".format(site_name)
            )

        # mj_ray принимает -1 как "не исключать ничего"
        self.exclude_body = exclude_body if exclude_body is not None else -1
        self._n_rays = V_LAYERS * H_RAYS

    def scan(self):
        # Выполняет один полный скан и возвращает облако точек.
        # Предполагает, что mj_forward или mj_step уже вызван - site_xpos актуален.
        # Возвращает ndarray shape (N, 3) в мировых координатах.
        # Лучи, не попавшие ни в что или улетевшие дальше MAX_RANGE, не включаются.

        lidar_pos = self.data.site_xpos[self.site_id].copy()

        # site_xmat хранит матрицу вращения site в мировую СК, row-major, длина 9
        R = self.data.site_xmat[self.site_id].reshape(3, 3)

        # Переводим локальные направления в мировую СК
        world_dirs = _RAY_DIRS_LOCAL @ R.T

        points = []
        for i in range(self._n_rays):
            geom_id_out = np.array([-1], dtype=np.int32)

            dist = mujoco.mj_ray(
                self.model,
                self.data,
                lidar_pos,
                world_dirs[i],
                None,               # geomgroup None = трассируем все группы
                1,                  # flg_static=1 = включаем статичную геометрию
                self.exclude_body,
                geom_id_out,
            )

            if 0.0 < dist <= MAX_RANGE:
                points.append(lidar_pos + dist * world_dirs[i])

        if points:
            return np.array(points, dtype=np.float32)
        else:
            return np.empty((0, 3), dtype=np.float32)

    def scan_with_stats(self):
        # Скан с дополнительной статистикой для отладки.
        # Возвращает кортеж (points, stats).
        # stats - словарь с полями: total_rays, hits, misses, hit_rate,
        # min_dist, max_dist, mean_dist.

        lidar_pos = self.data.site_xpos[self.site_id].copy()
        R         = self.data.site_xmat[self.site_id].reshape(3, 3)
        world_dirs = _RAY_DIRS_LOCAL @ R.T

        points    = []
        dists_hit = []

        for i in range(self._n_rays):
            geom_id_out = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                self.model, self.data,
                lidar_pos, world_dirs[i],
                None, 1, self.exclude_body, geom_id_out,
            )
            if 0.0 < dist <= MAX_RANGE:
                points.append(lidar_pos + dist * world_dirs[i])
                dists_hit.append(dist)

        points_arr = np.array(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)
        dists_arr  = np.array(dists_hit)

        stats = {
            "total_rays" : self._n_rays,
            "hits"       : len(dists_arr),
            "misses"     : self._n_rays - len(dists_arr),
            "hit_rate"   : len(dists_arr) / self._n_rays,
            "min_dist"   : float(dists_arr.min())  if len(dists_arr) else None,
            "max_dist"   : float(dists_arr.max())  if len(dists_arr) else None,
            "mean_dist"  : float(dists_arr.mean()) if len(dists_arr) else None,
        }
        return points_arr, stats
