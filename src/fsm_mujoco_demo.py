#!/usr/bin/env python3
"""
Unitree G1 - Balanced FSM + MuJoCo Physics Engine
Running RL Balance Policy alongside FSM Arm Actions.
"""

import os
import time
import numpy as np
import mujoco
from transitions import Machine
from locomotion.g1_walker import G1Walker

class BalancedG1FSM:
    states = ['IDLE', 'SEARCH', 'NAVIGATE', 'APPROACH', 'GRASP', 'TRANSPORT', 'RELEASE']

    def __init__(self, walker):
        self.walker = walker
        self.machine = Machine(model=self, states=BalancedG1FSM.states, initial='IDLE')

        # Определение переходов
        self.machine.add_transition('start', 'IDLE', 'SEARCH', after='act_search')
        self.machine.add_transition('found_object', 'SEARCH', 'NAVIGATE', after='act_navigate')
        self.machine.add_transition('at_target', 'NAVIGATE', 'APPROACH', after='act_approach')
        self.machine.add_transition('arm_placed', 'APPROACH', 'GRASP', after='act_grasp')
        self.machine.add_transition('grasped', 'GRASP', 'TRANSPORT', after='act_transport')
        self.machine.add_transition('delivered', 'TRANSPORT', 'RELEASE', after='act_release')
        self.machine.add_transition('done', 'RELEASE', 'IDLE')

    def step_balance(self, steps=10, arm_offsets=None):
        """Шаг симуляции с удержанием равновесия нейросетью"""
        for _ in range(steps):
            # 1. Шаг физики через RL-контроллер walker
            if hasattr(self.walker, 'step'):
                self.walker.step()
            else:
                mujoco.mj_step(self.walker.model, self.walker.data)

            # 2. Наложение движения рук поверх стойки
            if arm_offsets:
                for j_name, val in arm_offsets.items():
                    try:
                        j_id = mujoco.mj_name2id(self.walker.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                        if j_id != -1:
                            qadr = self.walker.model.jnt_qposadr[j_id]
                            self.walker.data.qpos[qadr] = val
                    except Exception:
                        pass
            mujoco.mj_forward(self.walker.model, self.walker.data)

    def print_telemetry(self, state_name):
        """Вывод высоты корпуса Z и статуса баланса"""
        pos = self.walker.data.qpos[0:3]
        status = "В СТОЙКЕ (OK)" if pos[2] > 0.5 else "УПАЛ (FAIL)"
        print(f"  [Телеметрия] [{state_name:<9}] | Корпус X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m | Статус: {status}")

    # ── ДЕЙСТВИЯ СОСТОЯНИЙ FSM ──

    def act_search(self):
        print("\n[FSM] 🔍 СОСТОЯНИЕ: SEARCH (Сканирование области в стойке)...")
        for t in range(80):
            arm_pose = {
                'right_shoulder_yaw_joint': 0.3 * np.sin(t * 0.1),
                'left_shoulder_yaw_joint': -0.3 * np.sin(t * 0.1)
            }
            self.step_balance(steps=5, arm_offsets=arm_pose)
            time.sleep(0.002)
        self.print_telemetry("SEARCH")

    def act_navigate(self):
        print("\n[FSM] 🚶 СОСТОЯНИЕ: NAVIGATE (Ходьба к объекту)...")
        # Вызов ходьбы через RL-политику
        if hasattr(self.walker, 'walk_distance'):
            self.walker.walk_distance(1.0)
        elif hasattr(self.walker, 'walk'):
            self.walker.walk(1.0)
        else:
            for _ in range(100):
                self.step_balance(steps=5)
        self.print_telemetry("NAVIGATE")

    def act_approach(self):
        print("\n[FSM] 🎯 СОСТОЯНИЕ: APPROACH (Вытягивание руки к объекту)...")
        for t in range(100):
            p = t / 100.0
            arm_pose = {
                'right_shoulder_pitch_joint': -0.8 * p,
                'right_elbow_joint': 0.6 * p
            }
            self.step_balance(steps=5, arm_offsets=arm_pose)
            time.sleep(0.002)
        self.print_telemetry("APPROACH")

    def act_grasp(self):
        print("\n[FSM] ✊ СОСТОЯНИЕ: GRASP (Захват объекта кистью)...")
        for t in range(50):
            arm_pose = {
                'right_shoulder_pitch_joint': -0.8,
                'right_elbow_joint': 0.6,
                'right_wrist_roll_joint': 0.5
            }
            self.step_balance(steps=5, arm_offsets=arm_pose)
            time.sleep(0.002)
        self.print_telemetry("GRASP")

    def act_transport(self):
        print("\n[FSM] 📦 СОСТОЯНИЕ: TRANSPORT (Транспортировка объекта)...")
        for _ in range(80):
            arm_pose = {'right_shoulder_pitch_joint': -0.5}
            self.step_balance(steps=5, arm_offsets=arm_pose)
        self.print_telemetry("TRANSPORT")

    def act_release(self):
        print("\n[FSM] 🫳 СОСТОЯНИЕ: RELEASE (Возврат руки в стойку)...")
        for t in range(60):
            p = 1.0 - (t / 60.0)
            arm_pose = {'right_shoulder_pitch_joint': -0.8 * p}
            self.step_balance(steps=5, arm_offsets=arm_pose)
            time.sleep(0.002)
        self.print_telemetry("RELEASE")
        print("\n[FSM] ✅ Полный сценарий «Помощник» успешно завершен!")

def main():
    print("=" * 65)
    print("  UNITREE G1: BALANCED FSM + MUJOCO SIMULATION")
    print("=" * 65)

    walker = G1Walker()
    fsm = BalancedG1FSM(walker)

    # Запуск автомата
    fsm.start()
    fsm.found_object()
    fsm.at_target()
    fsm.arm_placed()
    fsm.grasped()
    fsm.delivered()
    fsm.done()

if __name__ == "__main__":
    main()
