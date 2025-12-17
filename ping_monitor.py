#!/usr/bin/env python3
import subprocess
import re
import signal
from datetime import datetime

class PingMonitor:
    def __init__(self, target="8.8.8.8", duration=10):
        """
        Монитор ping для Linux
        
        Args:
            target (str): Целевой хост или IP
            duration (int): Время выполнения в секундах
        """
        self.target = target
        self.duration = duration
        self.process = None
        self.running = False
        
        # Данные для хранения
        self.packets = []  # Полные данные пакетов
        self.times = []    # Только значения времени
        self.total_sent = 0
        self.total_received = 0
        self.min_time = None
        self.max_time = None
        self.avg_time = None
        
    def parse_ping_line(self, line):
        """Парсинг строки вывода ping для Linux"""
        pattern = r'(\d+) bytes from .*?: icmp_seq=(\d+) ttl=(\d+) time=([\d.]+) ms'
        match = re.search(pattern, line.strip())
        
        if match:
            packet_data = {
                'bytes': int(match.group(1)),
                'seq': int(match.group(2)),
                'ttl': int(match.group(3)),
                'time': float(match.group(4)),
                'timestamp': datetime.now()
            }
            # Сохраняем только значение времени
            self.times.append(float(match.group(4)))
            return packet_data
        return None
    
    def time_iterator(self):
        """
        Итератор, который возвращает текущее значение времени ping
        
        Yields:
            float: Текущее значение времени в миллисекундах
        """
        # Команда ping для Linux
        command = ['ping', '-n', '-O', self.target]
        
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Устанавливаем таймер завершения
            def timeout_handler(signum, frame):
                if self.process:
                    self.process.terminate()
                self.running = False
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.duration)
            
            self.running = True
            
            # Читаем вывод построчно
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                    
                packet_data = self.parse_ping_line(line.strip())
                if packet_data:
                    self.packets.append(packet_data)
                    self.total_received += 1
                    
                    # Возвращаем текущее значение времени
                    yield packet_data['time']
            
            # Ждем завершения процесса
            if self.process:
                self.process.wait()
            
            # Отключаем таймер
            signal.alarm(0)
            
        except Exception as e:
            raise RuntimeError(f"Ошибка в итераторе ping: {e}")
        finally:
            self.running = False
            self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Расчет статистики на основе собранных данных"""
        if not self.times:
            return
            
        self.min_time = min(self.times)
        self.max_time = max(self.times)
        self.avg_time = sum(self.times) / len(self.times)
        
        # Оцениваем количество отправленных пакетов
        if self.packets:
            max_seq = max(p['seq'] for p in self.packets)
            self.total_sent = max_seq
    
    def get_statistics(self):
        """
        Возвращает статистику после завершения работы
        
        Returns:
            dict: Словарь со статистикой
        """
        return {
            'target': self.target,
            'duration': self.duration,
            'total_sent': self.total_sent,
            'total_received': self.total_received,
            'packet_loss': self.total_sent - self.total_received,
            'packet_loss_percent': ((self.total_sent - self.total_received) / 
                                   max(self.total_sent, 1)) * 100,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'times_count': len(self.times),
            'times': self.times
        }
    
    def save(self, filename=None, append=False):
        """
        Сохраняет только значения времени в файл
        
        Args:
            filename (str, optional): Имя файла. Если None, генерируется автоматически
            append (bool): Если True, добавляет к существующему файлу
            
        Returns:
            str: Имя сохраненного файла
        """
        if not self.times:
            raise ValueError("Нет данных о времени для сохранения")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ping_times_{self.target}_{timestamp}.log"
        
        mode = 'a' if append else 'w'
        
        with open(filename, mode) as f:
            if not append:
                f.write(f"# Время ping для {self.target}\n")
                f.write(f"# Начало: {datetime.now()}\n")
                f.write(f"# Всего записей: {len(self.times)}\n")
                f.write("# Время (ms)\n")
            
            for t in self.times:
                f.write(f"{t:.2f}\n")
        
        return filename
    
    def stop(self):
        """Остановка работы монитора"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process = None

