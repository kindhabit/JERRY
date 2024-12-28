import psutil
import os
import logging

logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.initial_usage = self.get_memory_usage()
        self.peak_usage = self.initial_usage['rss']
        
    def get_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / (1024 * 1024),  # MB
            'vms': memory_info.vms / (1024 * 1024),  # MB
            'percent': process.memory_percent()
        }
        
    def check_memory_threshold(self) -> bool:
        usage = self.get_memory_usage()
        self.peak_usage = max(self.peak_usage, usage['rss'])
        return usage['rss'] > self.threshold_mb
        
    def log_memory_usage(self):
        """현재 메모리 사용량 로깅"""
        usage = self.get_memory_usage()
        increase = usage['rss'] - self.initial_usage['rss']
        logger.info(
            f"메모리 사용량: "
            f"RSS: {usage['rss']:.2f}MB, "
            f"VMS: {usage['vms']:.2f}MB, "
            f"사용률: {usage['percent']:.1f}%, "
            f"최대: {self.peak_usage:.2f}MB, "
            f"증가량: {increase:.2f}MB"
        ) 