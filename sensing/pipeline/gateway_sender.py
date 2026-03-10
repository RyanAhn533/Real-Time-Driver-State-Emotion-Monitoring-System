"""
Gateway Sender
===============
8-byte USB 패킷을 차량 ECU로 전송.

PacketEncoder (multimodal_dms/gateway/packet_encoder.py)가 생성한
패킷을 pyserial로 전송한다.

연결 실패 시 graceful fallback (로그만, 크래시 안 함).

Usage:
    sender = GatewaySender(port="/dev/ttyUSB0", baudrate=115200)
    if sender.connect():
        sender.send(packet_bytes)
    sender.close()
"""

import sys
import time
from pathlib import Path
from typing import Optional

# sensing/ 디렉토리를 sys.path에 추가 (직접 실행 지원)
_SENSING_DIR = str(Path(__file__).resolve().parent.parent)
if _SENSING_DIR not in sys.path:
    sys.path.insert(0, _SENSING_DIR)

from core.logger import get_logger

logger = get_logger("kmer.gateway.sender")


class GatewaySender:
    """
    USB Serial 패킷 전송기.

    연결 실패/전송 실패 시 에러 로그만 남기고 계속 동작.
    주기적으로 재연결 시도.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        retry_interval: float = 5.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retry_interval = retry_interval

        self._serial = None
        self._last_retry = 0.0
        self._connected = False
        self._send_count = 0
        self._error_count = 0

    def connect(self) -> bool:
        """
        시리얼 포트 연결 시도.

        Returns:
            True if connected successfully
        """
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            self._connected = True
            self._last_retry = time.time()
            logger.info("Gateway connected: %s @ %d", self.port, self.baudrate)
            return True
        except ImportError:
            logger.warning("pyserial not installed. Gateway disabled.")
            self._connected = False
            return False
        except Exception as e:
            logger.warning("Gateway connect failed (%s): %s", self.port, e)
            self._connected = False
            self._last_retry = time.time()
            return False

    def send(self, packet: bytes) -> bool:
        """
        8-byte 패킷 전송.

        Args:
            packet: 8-byte USB packet (from PacketEncoder.encode())

        Returns:
            True if sent successfully
        """
        if not self._connected:
            # 주기적 재연결 시도
            now = time.time()
            if now - self._last_retry >= self.retry_interval:
                self.connect()
            return False

        try:
            if self._serial is not None and self._serial.is_open:
                self._serial.write(packet)
                self._serial.flush()
                self._send_count += 1
                return True
            else:
                self._connected = False
                return False
        except Exception as e:
            self._error_count += 1
            if self._error_count % 100 == 1:
                logger.warning("Gateway send error (%d total): %s",
                               self._error_count, e)
            self._connected = False
            return False

    def close(self):
        """시리얼 포트 닫기."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False
        logger.info("Gateway closed (sent=%d, errors=%d)",
                    self._send_count, self._error_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "port": self.port,
            "send_count": self._send_count,
            "error_count": self._error_count,
        }


class NullGatewaySender(GatewaySender):
    """
    Gateway 비활성화 시 사용하는 no-op sender.

    패킷을 전송하지 않고 로그만 남김.
    """

    def __init__(self):
        super().__init__()
        self._connected = False

    def connect(self) -> bool:
        logger.info("Gateway disabled (NullSender)")
        return False

    def send(self, packet: bytes) -> bool:
        return False

    def close(self):
        pass
