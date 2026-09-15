#!/usr/bin/env python3
"""
Unitree G1 - Assistant Behavior State Machine (FSM)
Pipeline: Search -> Navigate -> Approach -> Grasp -> Transport -> Release
"""

import time
import sys
from transitions import Machine

class AssistantRobot(object):
    # Определение состояний
    states = ['idle', 'search', 'navigate', 'approach', 'grasp', 'transport', 'release', 'error_recovery']

    def __init__(self, name="Unitree_G1"):
        self.name = name
        self.target_detected = False
        self.grasp_force_ok = False
        self.battery_ok = True

        # Инициализация FSM
        self.machine = Machine(
            model=self, 
            states=AssistantRobot.states, 
            initial='idle',
            send_event=True
        )

        # Определение переходов между состояниями
        self.machine.add_transition(trigger='start_mission', source='idle', dest='search', before='on_start')
        self.machine.add_transition(trigger='object_found', source='search', dest='navigate', after='on_navigate')
        self.machine.add_transition(trigger='reached_target_area', source='navigate', dest='approach', after='on_approach')
        self.machine.add_transition(trigger='in_position', source='approach', dest='grasp', after='on_grasp')
        self.machine.add_transition(trigger='grasp_success', source='grasp', dest='transport', after='on_transport')
        self.machine.add_transition(trigger='reached_delivery_zone', source='transport', dest='release', after='on_release')
        self.machine.add_transition(trigger='mission_done', source='release', dest='idle', after='on_complete')
        
        # Аварийные переходы
        self.machine.add_transition(trigger='fail', source=['search', 'navigate', 'approach', 'grasp', 'transport'], dest='error_recovery')
        self.machine.add_transition(trigger='recover', source='error_recovery', dest='idle')

    # ── Колбэки действий (Behavior Primitives) ──
    def on_start(self, event):
        print(f"\n[FSM] 🎯 Старт миссии. Переход в режим поиска объекта...")

    def on_navigate(self, event):
        print("[FSM] 🚶 Объект обнаружен в поле зрения камеры (HSV-фильтр: OK, (x=1.8m, y=0.3m)).")
        print("[FSM]    Вызов Nav2: планирование траектории и движение...")
        time.sleep(1.0)

    def on_approach(self, event):
        print("[FSM] 🎯 Робот подошел к целевой точке. Позиционирование манипулятора (IK)...")
        time.sleep(1.0)

    def on_grasp(self, event):
        print("[FSM] ✊ Активация кисти Dex3-1 (силовой захват с контролем усилия)...")
        time.sleep(1.0)

    def on_transport(self, event):
        print("[FSM] 📦 Объект зафиксирован. Включение походки с грузом (Carry Walk)...")
        print("[FSM]    Перемещение в зону выгрузки (Target Zone: x=0.0m, y=0.0m)...")
        time.sleep(1.2)

    def on_release(self, event):
        print("[FSM] 🫳 Достигнута зона доставки. Разжатие кисти Dex3-1 и отпускание объекта...")
        time.sleep(1.0)

    def on_complete(self, event):
        print("[FSM] ✅ Миссия успешно завершена! Робот возвращен в режим ожидания (Idle).")


def run_demo():
    print("=" * 60)
    print("      UNITREE G1: FSM BEHAVIOR EXECUTION ENGINE")
    print("=" * 60)

    robot = AssistantRobot()
    print(f"Текущее состояние: [{robot.state.upper()}]")

    # Симуляция выполнения полного цикла автомата
    try:
        robot.start_mission()
        print(f"Текущее состояние: [{robot.state.upper()}]")
        
        # Имитация работы сенсорной подсистемы
        print("  ... Обработка RGB-D потока ...")
        time.sleep(0.8)
        robot.object_found()
        print(f"Текущее состояние: [{robot.state.upper()}]")

        robot.reached_target_area()
        print(f"Текущее состояние: [{robot.state.upper()}]")

        robot.in_position()
        print(f"Текущее состояние: [{robot.state.upper()}]")

        robot.grasp_success()
        print(f"Текущее состояние: [{robot.state.upper()}]")

        robot.reached_delivery_zone()
        print(f"Текущее состояние: [{robot.state.upper()}]")

        robot.mission_done()
        print(f"Финальное состояние: [{robot.state.upper()}]")

    except Exception as e:
        print(f"[ERROR] Ошибка выполнения FSM: {e}")
        robot.fail()
        print(f"Аварийное состояние: [{robot.state.upper()}]")

    print("=" * 60)

if __name__ == "__main__":
    run_demo()
