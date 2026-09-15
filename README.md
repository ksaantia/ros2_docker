# 3D-картограф: цифровой двойник этажа

Проект по построению 3D-карты виртуального этажа по данным лидара. Робот Unitree G1 и коридор моделируются в MuJoCo, облако точек публикуется в ROS 2, а KISS-ICP регистрирует последовательность сканов и накапливает единую карту в реальном времени.

## Возможности

- виртуальный 3D-лидар на модели Unitree G1;
- raycasting в MuJoCo с полем зрения 360° по горизонтали и 30° по вертикали;
- публикация облака в ROS 2 topic `/pointcloud2` в формате `sensor_msgs/PointCloud2`;
- ручное перемещение робота по виртуальному коридору через `/cmd_vel`;
- визуализация живого облака, траектории и карты в RViz2;
- регистрация сканов KISS-ICP в реальном времени;
- сохранение итоговой карты в `.npy`, `.ply` и `.pcd`.

## Архитектура

```text
MuJoCo: test_corridor.xml + lidar_raycaster.py
                 |
                 v
      lidar_ros2_node.py
       /pointcloud2 + TF
                 |
                 v
      kiss_icp_slam_node.py
       ICP + VoxelHashMap
                 |
       /kiss_icp/global_map
       /kiss_icp/local_map
       /kiss_icp/odometry
       /kiss_icp/path
```

Лидар испускает 360 горизонтальных лучей в каждом из 16 вертикальных слоёв, всего **5760 лучей на скан**. Точки переводятся из локальной системы лидара в мировую систему MuJoCo и публикуются с frame id `lidar_link`.

## Структура проекта

```text
.
├── test_corridor.xml          # Тестовая сцена: коридор 10 x 2 м
├── lidar_raycaster.py         # Raycasting и генерация облака точек
├── lidar_ros2_node.py         # MuJoCo + ROS 2 + /pointcloud2 + TF
├── kiss_icp_slam_node.py      # Регистрация сканов и построение карты
├── test_lidar_raycaster.py    # Диагностический тест виртуального лидара
├── maps/                      # Сохраненные результаты картирования
│   ├── slam_map.npy
│   ├── slam_map.ply
│   └── slam_map.pcd
├── results/                   # Итоговые материалы эксперимента
└── src/                       # Дополнительные эксперименты с Unitree G1
```

## Требования

- macOS или Linux;
- Python 3.10+;
- ROS 2 с пакетами `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `tf2_ros`;
- MuJoCo;
- NumPy;
- `kiss-icp==1.2.3`;
- Open3D для экспорта карты в `.ply` и `.pcd`;
- RViz2.

Важно: ROS 2 и KISS-ICP должны быть доступны одному интерпретатору Python. В ходе проекта `kiss-icp` был установлен в системный Python, а conda использовала другой интерпретатор. Поэтому SLAM-нода запускалась явно через `/usr/bin/python3`.

## Установка

Запустите команды из корня репозитория:

```bash
cd /path/to/ros2_project

# Активировать ROS 2 (подставьте установленный дистрибутив вместо humble)
source /opt/ros/humble/setup.bash

# Установить Python-зависимости в тот же интерпретатор, где запускается SLAM
/usr/bin/python3 -m pip install numpy mujoco open3d kiss-icp==1.2.3
```

Проверка окружения:

```bash
/usr/bin/python3 -c "import mujoco, numpy, open3d, kiss_icp; print('dependencies: OK')"
ros2 doctor --report
```

## Запуск полного эксперимента

### 1. Проверить лидар в MuJoCo

```bash
source /opt/ros/humble/setup.bash
python3 test_lidar_raycaster.py
```

Откроется окно MuJoCo с тестовой сценой. В терминале печатаются число лучей, доля попаданий и диапазон измеренных расстояний.

### 2. Запустить ROS 2-ноду лидара

В первом терминале:

```bash
source /opt/ros/humble/setup.bash
python3 lidar_ros2_node.py
```

Нода запускает симуляцию, публикует `/pointcloud2` с частотой 10 Гц и трансформации `map -> odom -> base_link -> lidar_link`.

Проверить поток данных можно так:

```bash
ros2 topic list
ros2 topic hz /pointcloud2
ros2 topic echo /pointcloud2 --once
```

### 3. Запустить KISS-ICP

Во втором терминале из корня проекта:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 kiss_icp_slam_node.py
```

SLAM-нода подписывается на `/pointcloud2`, отбрасывает точки ближе 0.3 м и дальше 10 м, выполняет voxel downsampling и ICP-регистрацию. Размер voxel-а локальной карты составляет 0.1 м.

### 4. Перемещать робота

Нода принимает стандартное сообщение `geometry_msgs/msg/Twist` на `/cmd_vel`. Например:

```bash
# Движение вперед со скоростью 0.20 м/с
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.20}, angular: {z: 0.0}}"

# Поворот на месте
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.35}}"

# Остановить робота
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

Для ручного управления можно использовать любой ROS 2 teleop-узел, публикующий `/cmd_vel`.

### 5. Настроить RViz2

В третьем терминале:

```bash
source /opt/ros/humble/setup.bash
rviz2
```

В RViz2:

1. установить `Fixed Frame = odom`;
2. добавить `PointCloud2` с topic `/pointcloud2` для живого скана;
3. добавить `/kiss_icp/local_map` для локальной карты;
4. добавить `/kiss_icp/global_map` для накопленной карты;
5. добавить `Path` с topic `/kiss_icp/path` для траектории;
6. при необходимости добавить `Odometry` с topic `/kiss_icp/odometry`.

## Результаты

| Показатель                    |                           Результат |
| ----------------------------- | ----------------------------------: |
| Модель робота                 |                          Unitree G1 |
| Сцена                         | виртуальный коридор MuJoCo 10 x 2 м |
| Частота публикации облака     |                               10 Гц |
| Производительность raycasting |           14.3 мс/скан, около 70 Гц |
| Обработано сканов             |                                 807 |
| Среднее время ICP             |                        14.3 мс/скан |
| Размер итоговой карты         |                        54 194 точки |
| `slam_map.npy`                |                              1.3 МБ |
| `slam_map.ply`                |                              1.3 МБ |
| `slam_map.pcd`                |                              636 КБ |

Готовые файлы находятся в каталоге [`maps/`](maps/). Форматы `.ply` и `.pcd` удобно открывать в Open3D, CloudCompare или MeshLab.

## Форматы сохранения

Пример экспорта массива точек в `.ply` и `.pcd`:

```python
import numpy as np
import open3d as o3d

points = np.load("maps/slam_map.npy")
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("maps/slam_map.ply", cloud)
o3d.io.write_point_cloud("maps/slam_map.pcd", cloud)
```

## Ограничения и дальнейшая работа

В текущей версии наблюдается накопленный drift одометрии: при движении по коридору оцененная траектория постепенно отклоняется от прямой. Это ожидаемое ограничение локальной ICP-регистрации без замыкания петель.

Следующий технический шаг — добавить loop closure и глобальную оптимизацию позы. Это позволит уменьшить накопленную ошибку и повысить точность карты при повторном проходе по уже посещенным участкам.

## Автор

**Полупанова Ксения Дмитриевна**
