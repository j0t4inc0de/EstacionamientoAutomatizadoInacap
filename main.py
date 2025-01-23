# Camara
import cv2
from ultralytics import YOLO
from collections import deque
import time
# Interfaz Grafica
import sys
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QPushButton, QMenuBar, QMenu, QAction, QInputDialog, QWidget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from database.carga_de_datos import save_data, load_data
from PyQt5.QtGui import QPixmap

# Funciones para el uso de la camara/video
# Hilo para manejar la cámara
class CameraThread(QThread):
    frame_processed = pyqtSignal()
    vehicle_entered = pyqtSignal()  # Vehículo cruza línea de entrada
    vehicle_exited = pyqtSignal()   # Vehículo cruza línea de salida
    
    def __init__(self, video_path, yolo_model, left_line, right_line, frame_interval=3):
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.left_line = left_line
        self.right_line = right_line
        self.frame_interval = frame_interval  # Frecuencia de procesamiento de fotogramas (por defecto 3)
        self.frame_counter = 0  # Contador de fotogramas procesados
        self.running = True
        self.vehicle_tracks = deque(maxlen=100)  # Seguimiento limitado a los últimos 100 vehículos
        self.time_threshold = 1.0  # Tiempo mínimo entre detecciones para evitar duplicados
                # Añadir estados para seguimiento de vehículos
        self.vehicles_crossing_right = set()  # Para entrada
        self.vehicles_crossing_left = set()   # Para salida
        self.tracked_vehicles = {}  # Diccionario para seguimiento
        self.vehicle_id_counter = 0  # Contador para IDs únicos

    def resize_frame(self, frame, width=640):
        """Redimensiona el fotograma a una resolución específica."""
        height = int(frame.shape[0] * (width / frame.shape[1]))
        return cv2.resize(frame, (width, height))

    def check_line_crossing(self, center_x, center_y, vehicle_id, current_time):
        """Verifica el cruce de líneas con mejor manejo de estado"""
        
        # Verificar entrada (línea derecha)
        if (self.right_line[0][0] < center_x < self.right_line[1][0] and 
            self.right_line[0][1] - 5 <= center_y <= self.right_line[1][1] + 5):
            
            if vehicle_id not in self.vehicles_crossing_right:
                # Verificar si el vehículo no fue detectado recientemente
                recent_detection = False
                for track in self.vehicle_tracks:
                    if (abs(track['center_x'] - center_x) < 20 and 
                        abs(track['center_y'] - center_y) < 20 and 
                        current_time - track['timestamp'] < self.time_threshold):
                        recent_detection = True
                        break
                
                if not recent_detection:
                    self.vehicles_crossing_right.add(vehicle_id)
                    self.vehicle_tracks.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'timestamp': current_time
                    })
                    print("Vehículo detectado entrando")  # Debug
                    self.vehicle_entered.emit()
                    return "Entrada"
        else:
            self.vehicles_crossing_right.discard(vehicle_id)

        # Verificar salida (línea izquierda)
        if (self.left_line[0][0] < center_x < self.left_line[1][0] and 
            self.left_line[0][1] - 5 <= center_y <= self.left_line[1][1] + 5):
            
            if vehicle_id not in self.vehicles_crossing_left:
                recent_detection = False
                for track in self.vehicle_tracks:
                    if (abs(track['center_x'] - center_x) < 20 and 
                        abs(track['center_y'] - center_y) < 20 and 
                        current_time - track['timestamp'] < self.time_threshold):
                        recent_detection = True
                        break
                
                if not recent_detection:
                    self.vehicles_crossing_left.add(vehicle_id)
                    self.vehicle_tracks.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'timestamp': current_time
                    })
                    print("Vehículo detectado saliendo")  # Debug
                    self.vehicle_exited.emit()
                    return "Salida"
        else:
            self.vehicles_crossing_left.discard(vehicle_id)
        
        return None

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_counter += 1
            
            if self.frame_counter % self.frame_interval == 0:
                frame_resized = self.resize_frame(frame)
                results = self.yolo_model(frame_resized, conf=0.5)
                current_time = time.time()

                for result in results:
                    boxes = result.boxes.xyxy
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Asignar o recuperar ID del vehículo
                        vehicle_key = f"{center_x}_{center_y}"
                        if vehicle_key not in self.tracked_vehicles:
                            self.vehicle_id_counter += 1
                            self.tracked_vehicles[vehicle_key] = self.vehicle_id_counter
                        
                        vehicle_id = self.tracked_vehicles[vehicle_key]
                        
                        # Verificar cruce de líneas
                        crossing_status = self.check_line_crossing(
                            center_x, center_y, vehicle_id, current_time
                        )
                        
                        # Dibujar información en el frame
                        if crossing_status:
                            cv2.putText(frame_resized, crossing_status, 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 2)

                # Dibujar líneas
                cv2.line(frame_resized, self.left_line[0], self.left_line[1], (0, 0, 255), 2)
                cv2.line(frame_resized, self.right_line[0], self.right_line[1], (255, 0, 0), 2)

                cv2.imshow("Detección en tiempo real", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

# Interfaz grafica de conteo de autos
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sistema de estacionamiento - INACAP')
        self.setGeometry(100, 100, 800, 400)

        # Variables para mantener el estado de ocupación
        self.total_normal = 214
        self.ocupados_normal = 0
        self.ocupados_ejecutivo = 0
        self.ocupados_reservas = 0
        self.ocupados_discapacitados = 0
        self.ocupados_mecanica = 0
        self.ocupados_ambulancia = 0

        # Líneas para detección
        self.left_line = [(190, 150), (339, 150)]  # Línea izquierda (salida)
        self.right_line = [(362, 150), (500, 150)]  # Línea derecha (entrada)

        # Modelo YOLO
        self.yolo_model = YOLO('./yolov8n.pt')

        # Hilo de cámara
        self.camera_thread = None

        # Horario administrativo predeterminado
        self.hora_inicio_administrativo = 8
        self.hora_fin_administrativo = 17

        # Cargar datos desde el archivo
        data = load_data()
        if data:
            self.ocupados_normal = data.get("ocupados_normal", 0)
            self.ocupados_ejecutivo = data.get("ocupados_ejecutivo", 0)
            self.ocupados_reservas = data.get("ocupados_reservas", 0)
            self.ocupados_discapacitados = data.get("ocupados_discapacitados", 0)
            self.ocupados_mecanica = data.get("ocupados_mecanica", 0)
            self.ocupados_ambulancia = data.get("ocupados_ambulancia", 0)
            self.hora_inicio_administrativo = data.get("hora_inicio_administrativo", 8)
            self.hora_fin_administrativo = data.get("hora_fin_administrativo", 17)

        # Diseño principal
        main_layout = QVBoxLayout()

        # Crear un diseño horizontal para el encabezado
        header_layout = QHBoxLayout()

        # Etiqueta para el título
        header_label = QLabel("Estacionamientos:")
        header_label.setAlignment(Qt.AlignLeft)
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; color: #000000;")
        header_layout.addWidget(header_label)

        # Etiqueta para la imagen
        logo_label = QLabel(self)
        pixmap = QPixmap("fablab.jpeg")  # Logo FabLab

        if pixmap.isNull():
            print("Error: No se pudo cargar la imagen. Verifica la ruta o el archivo.")
        else:
            # Nuevo tamaño más grande (ancho x alto)
            nuevo_ancho = 100  # Aumenté de 100 a 200
            nuevo_alto = 100   # Aumenté de 50 a 100
            scaled_pixmap = pixmap.scaled(
                nuevo_ancho, 
                nuevo_alto, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setFixedSize(nuevo_ancho, nuevo_alto)  # Fija el tamaño del contenedor

        header_layout.addWidget(logo_label, alignment=Qt.AlignRight)

        # Añadir el diseño del encabezado al diseño principal
        main_layout.addLayout(header_layout)

        # Diseño en cuadrícula para las secciones principales
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)

        # Añadir secciones
        self.disponibles_label = self.create_section(grid_layout, 0, 0, "Disponibles", f"{self.total_normal}")
        self.ocupados_label = self.create_section(grid_layout, 0, 1, "Ocupados", f"{self.ocupados_normal}/{self.total_normal}")
        self.ejecutivo_label = self.create_section_with_buttons(grid_layout, 0, 2, "Ejecutivos", f"{self.ocupados_ejecutivo}/14", "ejecutivo", 14)

        self.hora_label = self.create_section(grid_layout, 0, 3, "Hora Actual", self.get_current_time())

        self.reservas_label = self.create_section_with_buttons(grid_layout, 1, 0, "Reservados", f"{self.ocupados_reservas}/10", "reservas", 10)
        self.discapacitados_label = self.create_section_with_buttons(grid_layout, 1, 1, "Discapacitados", f"{self.ocupados_discapacitados}/7", "discapacitados", 7)
        self.mecanica_label = self.create_section_with_buttons(grid_layout, 1, 2, "Mecánica", f"{self.ocupados_mecanica}/2", "mecanica", 2)
        self.ambulancia_label = self.create_section_with_buttons(grid_layout, 1, 3, "Ambulancia", f"{self.ocupados_ambulancia}/1", "ambulancia", 1)

        main_layout.addLayout(grid_layout)

        # Configurar el timer para actualizar la hora
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dynamic_data)
        self.timer.start(1000)  # Actualización cada segundo

        # Ajustar la disponibilidad inicial según el horario
        self.set_initial_availability()

        # Crear barra de menú
        self.create_menu()

        # Widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        central_widget.setStyleSheet("background-color: #ffffff;") 
        self.setCentralWidget(central_widget)

    def create_menu(self):
        menu_bar = self.menuBar()  # Utilizamos menuBar() ya que ahora es un QMainWindow

        # Menú "Configuración"
        config_menu = menu_bar.addMenu('Configuración')

        # Acción para cambiar el horario administrativo
        horario_action = QAction('Modificar horario administrativo', self)
        horario_action.triggered.connect(self.modify_horario_administrativo)
        config_menu.addAction(horario_action)

        # Acción para abrir la cámara
        camera_action = QAction('Abrir cámara', self)
        camera_action.triggered.connect(self.start_camera)
        config_menu.addAction(camera_action)

    def modify_horario_administrativo(self):
        """
        Método que se llama cuando se selecciona la opción de modificar el horario administrativo
        """
        # Obtener las nuevas horas de inicio y fin del horario administrativo
        inicio, ok_inicio = QInputDialog.getInt(self, "Modificar Hora de Inicio", "Horario AM administrativo (0-23):", self.hora_inicio_administrativo, 0, 23)
        if ok_inicio:
            fin, ok_fin = QInputDialog.getInt(self, "Modificar Hora de Fin", "Horario PM administrativo (0-23):", self.hora_fin_administrativo, 0, 23)
            if ok_fin:
                # Actualizar el horario administrativo
                self.hora_inicio_administrativo = inicio
                self.hora_fin_administrativo = fin
                print(f"Nuevo horario administrativo: {self.hora_inicio_administrativo}:00 - {self.hora_fin_administrativo}:00")

                # Reajustar disponibilidad de los espacios según el nuevo horario
                self.set_initial_availability()

    def set_initial_availability(self):
        """
        Ajusta la disponibilidad inicial según el horario actual y el horario administrativo configurado.
        """
        current_time = datetime.now().hour
        if self.hora_inicio_administrativo <= current_time < self.hora_fin_administrativo:
            self.ocupados_ejecutivo = 14
        else:
            self.ocupados_ejecutivo = 0

        self.update_section_labels()

    def update_dynamic_data(self):
        """
        Método que se llama cada segundo con el QTimer para actualizar la disponibilidad en tiempo real.
        """
        current_time = datetime.now().hour
        # Actualizar la hora actual
        if hasattr(self, 'hora_label'):
            self.hora_label.setText(self.get_current_time())
            
        if self.hora_inicio_administrativo <= current_time < self.hora_fin_administrativo:
            if self.ocupados_ejecutivo != 14:
                self.ocupados_ejecutivo = 14
                print(f"Horario administrativo: {self.hora_inicio_administrativo}:00 - {self.hora_fin_administrativo}:00 - 14 espacios ejecutivos ocupados.\nEstacionamientos Ocupados:{self.ocupados_normal}")

        # Actualizar etiquetas visuales y totales
        print("---"*20, f"\nOcupadosEjecutivo:{self.ocupados_ejecutivo}\nOcupadosDiscapacitados:{self.ocupados_discapacitados}\nOcupadosMecanica:{self.ocupados_mecanica}\nOcupadosAmbulancia:{self.ocupados_ambulancia}")
        self.update_section_labels()

    def update_section_labels(self):
        """
        Actualiza las etiquetas visuales de ocupación y disponibilidad en tiempo real.
        """
        # Actualizar las etiquetas de "Ocupados"
        if hasattr(self, 'ocupados_normal_label'):
            self.ocupados_normal_label.setText(f'{self.ocupados_normal + self.ocupados_ejecutivo}/{self.total_normal}')
        
        if hasattr(self, 'disponibles_normal_label'):
            self.disponibles_normal_label.setText(f'{self.total_normal - (self.ocupados_normal + self.ocupados_ejecutivo)}')

        # Actualizar la etiqueta de "Ejecutivo"
        ejecutivo_label = self.findChild(QLabel, "ejecutivo_label")
        if ejecutivo_label:
            ejecutivo_label.setText(f"{self.ocupados_ejecutivo}/14")

    def update_count(self, label, change, min_val, max_val, section):
        current_count = int(label.text().split('/')[0])
        new_count = current_count + change
        if min_val <= new_count <= max_val:
            label.setText(f"{new_count}/{max_val}")

            # Actualizar solo la sección correspondiente sin duplicar
            if section == "ejecutivo":
                self.ocupados_ejecutivo += change
                self.update_section_labels()
            else:
                self.update_total(section, change)

    def update_total(self, section, change):
        """
        Actualiza el total de ocupados en la sección específica y en el total general de "Normal".
        """
        # Actualiza los contadores de ocupados por sección
        if section == "ejecutivo":
            self.ocupados_ejecutivo += change
        elif section == "reservas":
            self.ocupados_reservas += change
        elif section == "discapacitados":
            self.ocupados_discapacitados += change
        elif section == "mecanica":
            self.ocupados_mecanica += change
        elif section == "ambulancia":
            self.ocupados_ambulancia += change

        # Actualiza el total general de ocupados
        self.ocupados_normal += change
        # Actualiza las etiquetas visuales
        self.ocupados_label.setText(f"{self.ocupados_normal}/{self.total_normal}")
        self.disponibles_label.setText(f"{self.total_normal - self.ocupados_normal}")
        # Guardamos los datos actualizados
        self.update_section_labels()

    def save_data_on_exit(self):
        """
        Método para guardar los datos antes de salir del programa.
        """
        data = {
            "ocupados_normal": self.ocupados_normal,
            "ocupados_ejecutivo": self.ocupados_ejecutivo,
            "ocupados_reservas": self.ocupados_reservas,
            "ocupados_discapacitados": self.ocupados_discapacitados,
            "ocupados_mecanica": self.ocupados_mecanica,
            "ocupados_ambulancia": self.ocupados_ambulancia,
            "hora_inicio_administrativo": self.hora_inicio_administrativo,
            "hora_fin_administrativo": self.hora_fin_administrativo,
        }
        save_data(data)  # Guardamos los datos en el archivo

    def get_current_time(self):
        return datetime.now().strftime("%H:%M")

    def create_section(self, layout, row, col, title, value):
        """
        Crear una sección sin botones.
        """
        section_layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 40px; 
            color: #FFFFFF;
            font-weight: bold;
        """)

        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("""
            font-size: 50px;
            color: #000000;
            font-weight: bold;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 5px;
        """)

        section_layout.addWidget(title_label)
        section_layout.addWidget(value_label)

        container = QWidget()
        container.setLayout(section_layout)
        container.setStyleSheet("""
            background-color: #eb5f14;
            border: 2px solid #000000;
            border-radius: 15px;
            padding: 20px;
        """)

        layout.addWidget(container, row, col)

        # Asignamos el value_label a una variable de instancia
        if title == "Ocupados":
            self.ocupados_normal_label = value_label
        elif title == "Disponibles":
            self.disponibles_normal_label = value_label

        return value_label

    def create_section_with_buttons(self, layout, row, col, title, value, section, max_val):
        """
        Crear una sección con botones de incremento y decremento.
        """
        section_layout = QVBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 35px;
            color: #FFFFFF;
            font-weight: semi-bold;
        """)

        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("font-size: 50px; color: #000000; background-color: #ffffff;")
        value_label.setObjectName(f"{section}_label")

        if section == "ejecutivo":
            self.ocupados_ejecutivo_label = value_label  # Asignamos la etiqueta ejecutiva

        btn_layout = QHBoxLayout()
        btn_incr = QPushButton('+')
        btn_decr = QPushButton('-')

        # Estilo para botones blancos
        button_style = """
            QPushButton {
                font-size: 30px;
                color: #000000;
                background-color: #ffffff;
                border: 2px solid #FFFFFF;
                border-radius: 8px;
                min-width: 30px;
                min-height: 30px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #c2c2c2;
            }
            QPushButton:pressed {
                background-color: #8f8f8f;
            }
        """
        btn_incr.setStyleSheet(button_style)
        btn_decr.setStyleSheet(button_style)
        btn_incr.clicked.connect(lambda: self.update_count(value_label, 1, 0, max_val, section))
        btn_decr.clicked.connect(lambda: self.update_count(value_label, -1, 0, max_val, section))

        btn_layout.addWidget(btn_decr)
        btn_layout.addWidget(btn_incr)

        section_layout.addWidget(title_label)
        section_layout.addWidget(value_label)
        section_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(section_layout)
        container.setStyleSheet("""
            background-color: #eb5f14;
            border: 2px solid #000000;
            border-radius: 15px;
            padding: 15px;
        """)

        layout.addWidget(container, row, col)
        return value_label
    
    def vehicle_entered(self):
        """
        Disminuye los estacionamientos disponibles y aumenta los ocupados
        cuando un vehículo entra (cruza la línea de entrada).
        """
        if self.total_normal - self.ocupados_normal > 0:  # Verifica si hay espacio disponible
            self.ocupados_normal += 1
            self.update_section_labels()

    def vehicle_exited(self):
        """
        Aumenta los estacionamientos disponibles y disminuye los ocupados
        cuando un vehículo sale (cruza la línea de salida).
        """
        if self.ocupados_normal > 0:  # Verifica si hay autos ocupando espacios
            self.ocupados_normal -= 1
            self.update_section_labels()
    
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            print("La cámara ya está en ejecución.")
            return
        self.camera_thread = CameraThread("rtsp://admin:admin2024@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0", self.yolo_model, self.left_line, self.right_line)
        # self.camera_thread = CameraThread("videoCAR.MOV", self.yolo_model, self.left_line, self.right_line)

        # Conectar señales para actualizar contadores
        self.camera_thread.vehicle_entered.connect(self.vehicle_entered)
        self.camera_thread.vehicle_exited.connect(self.vehicle_exited)

        self.camera_thread.start()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()

    # Guardar datos al cerrar el programa
    app.aboutToQuit.connect(window.save_data_on_exit)

    window.show()
    sys.exit(app.exec_())