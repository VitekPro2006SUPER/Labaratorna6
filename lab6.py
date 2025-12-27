import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QFrame, QMessageBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ODESolverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Чисельне розв'язування ДР (Ейлер vs Рунге-Кутта)")
        self.resize(950, 600)

        main_layout = QHBoxLayout(self)

        left_panel = QVBoxLayout()
        self.input_func = self.create_input("Рівняння y' =", "(1 - x**2) / (x * y)", left_panel)
        self.input_x0 = self.create_input("Початок (x0):", "1.0", left_panel) # x не може бути 0 для цього рівняння
        self.input_xn = self.create_input("Кінець (xn):", "2.6", left_panel)
        self.input_y0 = self.create_input("Умова (y0):", "2.0", left_panel)
        self.input_step = self.create_input("Крок (h):", "0.1", left_panel)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_panel.addWidget(line)

        # Кнопка
        self.btn_calc = QPushButton("Побудувати графік")
        self.btn_calc.setMinimumHeight(50)
        self.btn_calc.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; font-size: 14px;")
        self.btn_calc.clicked.connect(self.on_plot_clicked)
        left_panel.addWidget(self.btn_calc)
        
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

    def create_input(self, label_text, default_val, layout):
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
        inp = QLineEdit(default_val)
        layout.addWidget(lbl)
        layout.addWidget(inp)
        return inp

    # === Логіка Методів ===
    def solve_euler(self, f, x0, y0, h, xn):
        xs, ys = [x0], [y0]
        x, y = x0, y0
        while x < xn - 1e-9:
            try:
                y += h * f(x, y)
                x += h
                xs.append(x)
                ys.append(y)
            except (ZeroDivisionError, ValueError):
                break # Зупиняємо, якщо ділення на 0 або вихід за межі
        return xs, ys

    def solve_rk4(self, f, x0, y0, h, xn):
        xs, ys = [x0], [y0]
        x, y = x0, y0
        while x < xn - 1e-9:
            try:
                k1 = h * f(x, y)
                k2 = h * f(x + 0.5*h, y + 0.5*k1)
                k3 = h * f(x + 0.5*h, y + 0.5*k2)
                k4 = h * f(x + h, y + k3)
                
                y += (k1 + 2*k2 + 2*k3 + k4) / 6.0
                x += h
                xs.append(x)
                ys.append(y)
            except (ZeroDivisionError, ValueError):
                break
        return xs, ys

    def on_plot_clicked(self):
        try:
            eq_str = self.input_func.text()
            x0 = float(self.input_x0.text())
            xn = float(self.input_xn.text())
            y0 = float(self.input_y0.text())
            h = float(self.input_step.text())

            # Створення функції з рядка
            def func(x, y):
                return eval(eq_str, {"np": np, "x": x, "y": y})

            # Обчислення
            x_eul, y_eul = self.solve_euler(func, x0, y0, h, xn)
            x_rk, y_rk   = self.solve_rk4(func, x0, y0, h, xn)

            # Малювання
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(x_eul, y_eul, label=f'Метод Ейлера (h={h})', 
                    color='red', linestyle='--', marker='o', markersize=3, alpha=0.7)
            
            ax.plot(x_rk, y_rk, label=f'Рунге-Кутта 4 (h={h})', 
                    color='blue', linewidth=2)

            ax.set_title(f"Розв'язок: $y' = {eq_str}$")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True, linestyle=':', color='gray')

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Помилка обчислень", f"Перевірте рівняння та дані!\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ODESolverApp()
    window.show()
    sys.exit(app.exec())