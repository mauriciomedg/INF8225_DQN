# net_io.py
import json
import socket
from typing import Any, Dict, Generator, Optional, Tuple

class JsonLineClient:
    """
    TCP client that sends/receives newline-delimited JSON objects.
    One JSON object per line.
    """
    def __init__(self, host: str, port: int, tcp_nodelay: bool = True):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._buf = b""
        self.tcp_nodelay = tcp_nodelay

    def connect(self) -> None:
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        if self.tcp_nodelay:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
                self._buf = b""

    def send(self, obj: Dict[str, Any]) -> None:
        if self.sock is None:
            raise RuntimeError("Socket not connected")
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        self.sock.sendall(payload)

    def iter_messages(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields decoded JSON dicts as they arrive.
        """
        if self.sock is None:
            raise RuntimeError("Socket not connected")

        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("socket closed")

            self._buf += chunk

            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                raw = raw.strip(b"\r")
                if not raw:
                    continue

                s = raw.decode("utf-8", errors="replace")
                try:
                    yield json.loads(s)
                except json.JSONDecodeError:
                    # Keep it loud + helpful but don't crash training.
                    print("BAD LINE (repr):", repr(s))