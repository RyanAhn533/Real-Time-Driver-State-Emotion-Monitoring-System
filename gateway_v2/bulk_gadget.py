#!/usr/bin/env python3
"""
K-MER USB Bulk Gadget — FunctionFS
====================================
Jetson USB-C를 Bulk Transfer 디바이스로 설정하고 패킷 전송.
커널 6.8 FunctionFS v2 디스크립터 규격.
"""

import struct
import os
import sys
import time
import signal

EP0_PATH = "/dev/usb-ffs/bulk0/ep0"
UDC = "a808670000.usb"

# FunctionFS constants
FUNCTIONFS_DESCRIPTORS_MAGIC_V2 = 2
FUNCTIONFS_STRINGS_MAGIC = 2
FUNCTIONFS_HAS_FS = 1
FUNCTIONFS_HAS_HS = 2
FUNCTIONFS_HAS_SS = 4

# USB descriptor types
USB_DT_INTERFACE = 4
USB_DT_ENDPOINT = 5
USB_DT_SS_ENDPOINT_COMP = 48  # 0x30


def make_interface_desc(num_endpoints=2, iface_class=0xFF, subclass=0x01, protocol=0x01):
    """9-byte interface descriptor."""
    return struct.pack("BBBBBBBBB",
        9,                  # bLength
        USB_DT_INTERFACE,   # bDescriptorType
        0,                  # bInterfaceNumber
        0,                  # bAlternateSetting
        num_endpoints,      # bNumEndpoints
        iface_class,        # bInterfaceClass (0xFF = Vendor Specific)
        subclass,           # bInterfaceSubClass
        protocol,           # bInterfaceProtocol
        1,                  # iInterface (string index)
    )


def make_endpoint_desc(addr, max_packet_size):
    """7-byte endpoint descriptor."""
    return struct.pack("<BBBBHB",
        7,              # bLength
        USB_DT_ENDPOINT,# bDescriptorType
        addr,           # bEndpointAddress
        0x02,           # bmAttributes (Bulk)
        max_packet_size,# wMaxPacketSize
        0,              # bInterval
    )


def make_ss_companion_desc(max_burst=0):
    """6-byte SuperSpeed endpoint companion descriptor."""
    return struct.pack("<BBBHB",
        6,                      # bLength
        USB_DT_SS_ENDPOINT_COMP,# bDescriptorType
        max_burst,              # bMaxBurst
        0,                      # bmAttributes
        0,                      # wBytesPerInterval (padding to 6)
    )


def build_descriptors():
    """Build FunctionFS v2 descriptor blob."""
    intf = make_interface_desc()

    # FS: Full Speed (64B)
    fs_ep_out = make_endpoint_desc(0x01, 64)
    fs_ep_in  = make_endpoint_desc(0x81, 64)
    fs_descs = intf + fs_ep_out + fs_ep_in

    # HS: High Speed (512B)
    hs_ep_out = make_endpoint_desc(0x01, 512)
    hs_ep_in  = make_endpoint_desc(0x81, 512)
    hs_descs = intf + hs_ep_out + hs_ep_in

    # SS: SuperSpeed (1024B) + companion
    ss_ep_out = make_endpoint_desc(0x01, 1024)
    ss_ep_out_comp = make_ss_companion_desc(0)
    ss_ep_in  = make_endpoint_desc(0x81, 1024)
    ss_ep_in_comp = make_ss_companion_desc(0)
    ss_descs = intf + ss_ep_out + ss_ep_out_comp + ss_ep_in + ss_ep_in_comp

    flags = FUNCTIONFS_HAS_FS | FUNCTIONFS_HAS_HS | FUNCTIONFS_HAS_SS
    fs_count = 3   # 1 intf + 2 eps
    hs_count = 3
    ss_count = 5   # 1 intf + 2 eps + 2 companions

    # Header: magic(4) + length(4) + flags(4)
    # Then per-speed counts: fs_count(4) + hs_count(4) + ss_count(4)
    # Then descriptors
    counts = struct.pack("<III", fs_count, hs_count, ss_count)
    payload = counts + fs_descs + hs_descs + ss_descs
    total_length = 12 + len(payload)  # header(12) + payload

    header = struct.pack("<III",
        FUNCTIONFS_DESCRIPTORS_MAGIC_V2,
        total_length,
        flags,
    )

    return header + payload


def build_strings():
    """Build FunctionFS strings blob."""
    s = b"KMER-Bulk\0"
    # Header: magic(4) + length(4) + str_count(4)
    # Then: lang(2) + string data
    lang_data = struct.pack("<H", 0x0409) + s
    total_length = 12 + len(lang_data)
    header = struct.pack("<III",
        FUNCTIONFS_STRINGS_MAGIC,
        total_length,
        1,  # str_count
    )
    return header + lang_data


def main():
    print("[BULK] Building descriptors...", flush=True)
    desc = build_descriptors()
    strings = build_strings()

    print(f"[BULK] Descriptor: {len(desc)}B, Strings: {len(strings)}B", flush=True)

    # Open EP0 and write descriptors
    fd = os.open(EP0_PATH, os.O_RDWR)
    try:
        os.write(fd, desc)
        print("[BULK] Descriptors written OK", flush=True)
    except OSError as e:
        print(f"[BULK] Descriptor write FAILED: {e}", flush=True)
        os.close(fd)
        return False

    try:
        os.write(fd, strings)
        print("[BULK] Strings written OK", flush=True)
    except OSError as e:
        print(f"[BULK] Strings write FAILED: {e}", flush=True)
        os.close(fd)
        return False

    # Activate UDC
    print("[BULK] Activating UDC...", flush=True)
    os.system(f"echo {UDC} | sudo tee /sys/kernel/config/usb_gadget/l4t/UDC > /dev/null 2>&1")
    time.sleep(3)

    # Check endpoints
    try:
        eps = os.listdir("/dev/usb-ffs/bulk0/")
    except PermissionError:
        eps = os.popen("sudo ls /dev/usb-ffs/bulk0/").read().split()
    print(f"[BULK] Endpoints: {eps}", flush=True)

    if "ep1" not in eps or "ep2" not in eps:
        print("[BULK] Endpoints not created — host may not be connected", flush=True)
        # Keep EP0 open and wait
        print("[BULK] Waiting for host connection... (Ctrl+C to stop)", flush=True)
        try:
            while True:
                try:
                    eps = os.popen("sudo ls /dev/usb-ffs/bulk0/").read().split()
                except:
                    pass
                if "ep1" in eps and "ep2" in eps:
                    print("[BULK] Host connected! Endpoints available.", flush=True)
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            os.close(fd)
            return False

    # Send packets via EP2 (Bulk IN = device → host)
    print("[BULK] Opening EP2 (Bulk IN)...", flush=True)
    try:
        ep2_fd = os.open("/dev/usb-ffs/bulk0/ep2", os.O_WRONLY)
    except PermissionError:
        os.system("sudo chmod 666 /dev/usb-ffs/bulk0/ep1 /dev/usb-ffs/bulk0/ep2")
        ep2_fd = os.open("/dev/usb-ffs/bulk0/ep2", os.O_WRONLY)

    packets = [
        (b'\x02\x01\x00\x00\x10\x03\x00\xF0\x00\xFC\x1C\x03', '공포'),
        (b'\x02\x01\x00\x00\x10\x03\x00\xF0\x01\xFC\x1D\x03', '놀람'),
        (b'\x02\x01\x00\x00\x10\x03\x00\xF0\x02\xFC\x1E\x03', '분노'),
        (b'\x02\x01\x00\x00\x10\x03\x00\xF0\x03\xFC\x1F\x03', '슬픔'),
        (b'\x02\x01\x00\x00\x10\x03\x00\xF0\x04\xFC\x18\x03', '행복'),
        (b'\x02\x01\x00\x00\x10\x03\x00\xF0\x05\xFC\x19\x03', '혐오'),
    ]

    print(f"[BULK] Sending {len(packets)} packets (10s interval)...", flush=True)
    for i, (pkt, name) in enumerate(packets):
        try:
            n = os.write(ep2_fd, pkt)
            print(f"[{i+1}/{len(packets)}] {name} TX OK ({n}B) | {' '.join(f'{b:02X}' for b in pkt)}", flush=True)
        except BlockingIOError:
            print(f"[{i+1}/{len(packets)}] {name} WOULD BLOCK", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(packets)}] {name} TX FAIL: {e}", flush=True)

        if i < len(packets) - 1:
            time.sleep(10)

    os.close(ep2_fd)
    os.close(fd)
    print("=== done ===", flush=True)
    return True


if __name__ == "__main__":
    main()
