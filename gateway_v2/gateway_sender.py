"""
Gateway Sender v2 (모트렉스 13-byte 프로토콜)
==============================================
13-byte 패킷을 USB Serial 또는 TCP 소켓으로 전송.

Usage:
    # USB Serial
    sender = GatewaySender(port="/dev/ttyUSB0", baudrate=115200)

    # TCP (네트워크 테스트)
    sender = TcpGatewaySender(host="192.168.1.10", port=9000)

    if sender.connect():
        sender.send(packet_bytes)
    sender.close()
"""

import time
import socket
import logging
from typing import Optional

from .packet_encoder import PACKET_SIZE, STX, ETX

logger = logging.getLogger("kmer.gateway.sender")


class GatewaySender:
    """USB Serial 패킷 전송기 (모트렉스 13-byte)."""

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
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout,
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
        if len(packet) != PACKET_SIZE:
            return False
        if packet[0] != STX or packet[-1] != ETX:
            return False

        if not self._connected:
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
                logger.warning("Gateway send error (%d): %s", self._error_count, e)
            self._connected = False
            return False

    def close(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False
        logger.info("Gateway closed (sent=%d, errors=%d)", self._send_count, self._error_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "transport": "serial",
            "port": self.port,
            "send_count": self._send_count,
            "error_count": self._error_count,
        }


class TcpGatewaySender:
    """TCP 소켓 패킷 전송기 (네트워크 테스트용). 패킷 내용은 USB와 동일."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, retry_interval: float = 5.0):
        self.host = host
        self.port = port
        self.retry_interval = retry_interval

        self._sock: Optional[socket.socket] = None
        self._last_retry = 0.0
        self._connected = False
        self._send_count = 0
        self._error_count = 0

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(3.0)
            self._sock.connect((self.host, self.port))
            self._connected = True
            self._last_retry = time.time()
            logger.info("TCP Gateway connected: %s:%d", self.host, self.port)
            return True
        except Exception as e:
            logger.warning("TCP connect failed (%s:%d): %s", self.host, self.port, e)
            self._connected = False
            self._last_retry = time.time()
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
            return False

    def send(self, packet: bytes) -> bool:
        if len(packet) != PACKET_SIZE:
            return False
        if packet[0] != STX or packet[-1] != ETX:
            return False

        if not self._connected:
            now = time.time()
            if now - self._last_retry >= self.retry_interval:
                self.connect()
            return False

        try:
            self._sock.sendall(packet)
            self._send_count += 1
            return True
        except Exception as e:
            self._error_count += 1
            if self._error_count % 100 == 1:
                logger.warning("TCP send error (%d): %s", self._error_count, e)
            self._connected = False
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            return False

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False
        logger.info("TCP Gateway closed (sent=%d, errors=%d)", self._send_count, self._error_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "transport": "tcp",
            "host": self.host,
            "port": self.port,
            "send_count": self._send_count,
            "error_count": self._error_count,
        }


class NullGatewaySender:
    """Gateway 비활성화 시 사용하는 no-op sender."""

    def __init__(self):
        self._connected = False
        self._send_count = 0
        self._error_count = 0

    def connect(self) -> bool:
        return False

    def send(self, packet: bytes) -> bool:
        return False

    def close(self):
        pass

    @property
    def is_connected(self) -> bool:
        return False

    @property
    def stats(self) -> dict:
        return {"connected": False, "transport": "null", "send_count": 0, "error_count": 0}
