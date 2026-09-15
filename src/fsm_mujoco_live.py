#!/usr/bin/env python3
"""Live MuJoCo + FSM for Unitree G1 assistant with active joint motion.
Автоматически перезапускает миссию по кругу — Ctrl+C не нужен для повторных тестов."""
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from assistant_fsm import AssistantRobot


def set_ctrl(model, data, joint_name, value):
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    if act_id >= 0:
        data.ctrl[act_id] = value
        return True
    return False


def apply_arm_motion(model, data, state, t_rel):
    if state == "search":
        arm_val = 0.5 * np.sin(t_rel * 3.0)
        set_ctrl(model, data, "right_shoulder_pitch_joint", arm_val)
        set_ctrl(model, data, "left_shoulder_pitch_joint", -arm_val)

    elif state in ["approach", "grasp"]:
        set_ctrl(model, data, "right_shoulder_pitch_joint", 0.8)
        set_ctrl(model, data, "left_shoulder_pitch_joint", 0.8)
        set_ctrl(model, data, "right_elbow_joint", 0.5)
        set_ctrl(model, data, "left_elbow_joint", 0.5)

    elif state == "transport":
        set_ctrl(model, data, "right_shoulder_pitch_joint", 0.4)
        set_ctrl(model, data, "left_shoulder_pitch_joint", 0.4)
        set_ctrl(model, data, "right_elbow_joint", 1.0)
        set_ctrl(model, data, "left_elbow_joint", 1.0)

    elif state == "release":
        set_ctrl(model, data, "right_shoulder_pitch_joint", 0.0)
        set_ctrl(model, data, "left_shoulder_pitch_joint", 0.0)
        set_ctrl(model, data, "right_elbow_joint", 0.0)
        set_ctrl(model, data, "left_elbow_joint", 0.0)


def run_live():
    xml_path = "mujoco_menagerie/unitree_g1/scene.xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML модель не найдена: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    robot = AssistantRobot()

    def start_new_cycle():
        """Сброс симуляции и старт новой миссии — без выхода из процесса."""
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        robot.state = "idle"
        robot.start_mission()
        print("\n==========================================")
        print(f"[FSM] Новый цикл. Состояние: {robot.state}")
        print("==========================================\n")
        return time.time()

    t0 = start_new_cycle()
    idle_wait_started = None
    IDLE_PAUSE_SEC = 2.0  # пауза в конце idle перед автоперезапуском

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                state = robot.state
                t_rel = time.time() - t0

                apply_arm_motion(model, data, state, t_rel)
                mujoco.mj_step(model, data)

                if state == "search" and t_rel > 3.0:
                    robot.object_found()
                    print(f"[FSM] Обнаружен объект. Переход в: {robot.state}")
                    t0 = time.time()

                elif state == "navigate" and t_rel > 4.0:
                    robot.reached_target_area()
                    print(f"[FSM] Подошел к цели. Переход в: {robot.state}")
                    t0 = time.time()

                elif state == "approach" and t_rel > 3.0:
                    robot.in_position()
                    print(f"[FSM] Позиция для захвата. Переход в: {robot.state}")
                    t0 = time.time()

                elif state == "grasp" and t_rel > 2.0:
                    robot.grasp_success()
                    print(f"[FSM] Захват выполнен. Переход в: {robot.state}")
                    t0 = time.time()

                elif state == "transport" and t_rel > 4.0:
                    robot.reached_delivery_zone()
                    print(f"[FSM] Доставлен в зону. Переход в: {robot.state}")
                    t0 = time.time()

                elif state == "release" and t_rel > 2.0:
                    robot.mission_done()
                    print(f"[FSM] Миссия завершена. Робот вернулся в: {robot.state}")
                    idle_wait_started = time.time()

                elif state == "idle" and idle_wait_started is not None:
                    if time.time() - idle_wait_started > IDLE_PAUSE_SEC:
                        idle_wait_started = None
                        t0 = start_new_cycle()

                viewer.sync()
                time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nЗавершение работы оператором.")


if __name__ == "__main__":
    run_live()
