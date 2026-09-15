import numpy as np
import mujoco
import mujoco.viewer

from lidar_raycaster import LidarRaycaster

XML_PATH = "test_corridor.xml"


def print_scan_summary(stats, points):
    print("")
    print("=" * 50)
    print("  Лучей всего : {}".format(stats["total_rays"]))
    print("  Попаданий   : {}  ({:.1%})".format(stats["hits"], stats["hit_rate"]))
    print("  Промахов    : {}".format(stats["misses"]))

    if stats["hits"] > 0:
        print("  Дист. мин   : {:.3f} м".format(stats["min_dist"]))
        print("  Дист. макс  : {:.3f} м".format(stats["max_dist"]))
        print("  Дист. сред  : {:.3f} м".format(stats["mean_dist"]))
    print("=" * 50)

    if stats["hits"] > 0:
        # Смотрим только на горизонтальный слой - точки на высоте лидара +/- 10 см.
        # В коридоре шириной 2м ожидаем Y от -1.0 до +1.0 и X от -5.0 до +5.0.
        lidar_z   = 1.357
        horiz_mask = np.abs(points[:, 2] - lidar_z) < 0.1
        horiz_pts  = points[horiz_mask]

        if len(horiz_pts) > 0:
            y_coords = horiz_pts[:, 1]
            x_coords = horiz_pts[:, 0]
            print("")
            print("  Проверка горизонтального слоя ({} точек):".format(len(horiz_pts)))
            print("  Y мин : {:.3f}  (ожидаем около -1.0)".format(y_coords.min()))
            print("  Y макс: {:.3f}  (ожидаем около +1.0)".format(y_coords.max()))
            print("  X мин : {:.3f}  (ожидаем около -5.0)".format(x_coords.min()))
            print("  X макс: {:.3f}  (ожидаем около +5.0)".format(x_coords.max()))


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Один forward pass обязателен перед первым сканом -
    # без него site_xpos содержит нули, и позиция лидара будет неверной.
    mujoco.mj_forward(model, data)

    # exclude_body=None потому что в тестовой сцене нет тела робота,
    # нечего исключать
    lidar = LidarRaycaster(model, data, site_name="lidar_site", exclude_body=None)

    print("Лидар инициализирован.")
    print("  site_id   : {}".format(lidar.site_id))
    print("  lidar_pos : {}".format(data.site_xpos[lidar.site_id]))

    print("")
    print("Запускаю диагностический скан...")
    points, stats = lidar.scan_with_stats()
    print_scan_summary(stats, points)

    print("")
    print("Открываю viewer. Закройте окно для выхода.")
    print("Каждые 100 шагов симуляции в терминал печатается краткая статистика скана.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_counter = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            step_counter += 1

            if step_counter % 100 == 0:
                points, stats = lidar.scan_with_stats()
                print("Шаг {:5d}: hits={}, hit_rate={:.1%}, mean_dist={:.2f} м".format(
                    step_counter,
                    stats["hits"],
                    stats["hit_rate"],
                    stats["mean_dist"] if stats["mean_dist"] is not None else 0.0,
                ))

            viewer.sync()


if __name__ == "__main__":
    main()
