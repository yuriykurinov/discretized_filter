import subprocess
import re
import json
import threading
import time
from datetime import datetime

class PingMonitor:
    def __init__(self, target="8.8.8.8", duration=10):
        """
        Монитор ping для Linux с фиксированной длительностью
        
        Args:
            target (str): Целевой хост или IP
            duration (int): Время выполнения в секундах
        """
        self.target = target
        self.duration = duration
        self.process = None
        self.running = False
        self.thread = None
        self.packets = []
        self.total_sent = 0
        self.total_received = 0
        self.min_time = None
        self.max_time = None
        self.avg_time = None
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
        
    def parse_ping_line(self, line):
        """Парсинг строки вывода ping"""
        line = line.strip()
        
        # Паттерн для успешного ответа ping
        success_pattern = r'(\d+) bytes from .*: icmp_seq=(\d+) ttl=(\d+) time=([\d.]+) ms'
        # Паттерн для потерянного пакета (если используем -O)
        timeout_pattern = r'no answer yet for icmp_seq=(\d+)'
        # Альтернативный паттерн для ping без -O
        timeout_pattern2 = r'Request timeout for icmp_seq (\d+)'
        
        success_match = re.search(success_pattern, line)
        timeout_match = re.search(timeout_pattern, line) or re.search(timeout_pattern2, line)
        
        if success_match:
            return {
                'type': 'response',
                'bytes': int(success_match.group(1)),
                'seq': int(success_match.group(2)),
                'ttl': int(success_match.group(3)),
                'time': float(success_match.group(4)),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        elif timeout_match:
            seq = int(timeout_match.group(1))
            return {
                'type': 'timeout',
                'seq': seq,
                'timestamp': datetime.now().isoformat(),
                'status': 'timeout'
            }
        
        # Также парсим статистику в конце
        stats_pattern = r'(\d+) packets transmitted, (\d+) received'
        stats_match = re.search(stats_pattern, line)
        if stats_match:
            self.total_sent = int(stats_match.group(1))
            self.total_received = int(stats_match.group(2))
        
        return None
    
    def _run_monitor(self):
        """Внутренний метод для запуска мониторинга"""
        try:
            # Используем -c для ограничения количества пакетов или -w для таймаута
            # Вместо -O используем более стандартный подход
            command = ['ping', '-n', '-i', '0.5', '-w', str(self.duration), self.target]
            
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.start_time = datetime.now().isoformat()
            
            # Читаем вывод построчно
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                    
                packet_data = self.parse_ping_line(line)
                if packet_data:
                    with self.lock:
                        self.packets.append(packet_data)
            
            # Ждем завершения процесса
            if self.process:
                self.process.wait(timeout=self.duration + 2)
                
        except subprocess.TimeoutExpired:
            # Принудительно завершаем если завис
            if self.process:
                self.process.kill()
        except Exception as e:
            print(f"Ошибка в мониторинге ping для {self.target}: {e}")
        finally:
            self.running = False
            self.end_time = datetime.now().isoformat()
            self._calculate_statistics()
    
    def start(self):
        """Запускает мониторинг"""
        if self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._run_monitor)
        self.thread.start()
        return True
    
    def wait(self, timeout=None):
        """Ожидает завершения мониторинга"""
        if not self.thread:
            return True
        
        # Если timeout не указан, используем duration + 5 секунд
        if timeout is None:
            timeout = self.duration + 5
        
        self.thread.join(timeout=timeout)
        
        # Если поток все еще работает, принудительно останавливаем
        if self.thread.is_alive():
            self.stop()
            return False
        
        return True
    
    def stop(self):
        """Останавливает мониторинг"""
        if not self.running:
            return False
        
        self.running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        return True
    
    def _calculate_statistics(self):
        """Расчет статистики"""
        with self.lock:
            successful_packets = [p for p in self.packets if p.get('status') == 'success']
            
            if not successful_packets:
                return
            
            times = [p['time'] for p in successful_packets]
            if times:
                self.min_time = min(times)
                self.max_time = max(times)
                self.avg_time = sum(times) / len(times)
            
            if self.packets:
                seq_numbers = [p.get('seq', 0) for p in self.packets if p.get('seq')]
                if seq_numbers:
                    self.total_sent = max(seq_numbers)
    
    def get_statistics(self):
        """Возвращает статистику"""
        with self.lock:
            successful_packets = [p for p in self.packets if p.get('status') == 'success']
            lost_packets = [p for p in self.packets if p.get('status') == 'timeout']
            
            total_received = len(successful_packets)
            total_lost = len(lost_packets)
            total_sent = self.total_sent if self.total_sent > 0 else (total_received + total_lost)
            
            packet_loss_percent = 0
            if total_sent > 0:
                packet_loss_percent = (total_lost / total_sent) * 100
            
            return {
                'target': self.target,
                'duration': self.duration,
                'running': self.running,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'total_sent': total_sent,
                'total_received': total_received,
                'total_lost': total_lost,
                'packet_loss_percent': packet_loss_percent,
                'min_time': self.min_time,
                'max_time': self.max_time,
                'avg_time': self.avg_time,
                'total_packets': len(self.packets),
                'successful_packets': len(successful_packets),
                'lost_packets': len(lost_packets)
            }
    
    def save_all_data(self, filename=None, format='json'):
        """Сохраняет все данные"""
        with self.lock:
            if not self.packets:
                print(f"Нет данных для сохранения ({self.target})")
                return None
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ping_data_{self.target}_{timestamp}"
            
            try:
                if format == 'json':
                    filename = f"{filename}.json"
                    self._save_json(filename)
                elif format == 'csv':
                    filename = f"{filename}.csv"
                    self._save_csv(filename)
                else:
                    raise ValueError(f"Неподдерживаемый формат: {format}")
                
                print(f"Данные сохранены в {filename}")
                return filename
            except Exception as e:
                print(f"Ошибка при сохранении данных для {self.target}: {e}")
                return None
    
    def _save_json(self, filename):
        """Сохраняет в JSON"""
        data = {
            'metadata': {
                'target': self.target,
                'duration': self.duration,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'statistics': self.get_statistics()
            },
            'packets': self.packets
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_csv(self, filename):
        """Сохраняет в CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Метаданные
            writer.writerow(['# Ping Data Export'])
            writer.writerow(['# Target:', self.target])
            writer.writerow(['# Duration (s):', self.duration])
            writer.writerow(['# Start Time:', self.start_time])
            writer.writerow(['# End Time:', self.end_time])
            writer.writerow([])
            
            # Статистика
            stats = self.get_statistics()
            writer.writerow(['# Statistics'])
            for key, value in stats.items():
                writer.writerow(['#', key, value])
            writer.writerow([])
            
            # Данные пакетов
            if self.packets:
                # Определяем все возможные заголовки
                all_headers = set()
                for packet in self.packets:
                    all_headers.update(packet.keys())
                headers = sorted(all_headers)
                
                writer.writerow(headers)
                for packet in self.packets:
                    row = [packet.get(header, '') for header in headers]
                    writer.writerow(row)
    
    def save_times_only(self, filename=None):
        """Сохраняет только времена (для обратной совместимости)"""
        with self.lock:
            successful_times = [p['time'] for p in self.packets if p.get('status') == 'success']
            
            if not successful_times:
                print(f"Нет данных о времени для сохранения ({self.target})")
                return None
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ping_times_{self.target}_{timestamp}.log"
            
            try:
                with open(filename, 'w') as f:
                    f.write(f"# Ping times for {self.target}\n")
                    f.write(f"# Duration: {self.duration}s\n")
                    f.write(f"# Start: {self.start_time}\n")
                    f.write(f"# End: {self.end_time}\n")
                    f.write(f"# Total records: {len(successful_times)}\n")
                    f.write("# Time (ms)\n")
                    
                    for t in successful_times:
                        f.write(f"{t:.2f}\n")
                
                print(f"Времена сохранены в {filename}")
                return filename
            except Exception as e:
                print(f"Ошибка при сохранении времен для {self.target}: {e}")
                return None