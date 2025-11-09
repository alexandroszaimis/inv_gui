#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverter GUI (Qt / PySide6)

- Shared serial port (reader thread) -> UI can send *while* logging
- Protocol:
    * Normal frames: HEADER | CMD | LEN | PAYLOAD | CRC16-CCITT
    * Logger stream: HEADER | CMD_LOG_PAYLOAD | LEN | PAYLOAD    (NO CRC)
- Tabs:
    1) Settings
    2) Control & Monitor (periodic send, highlights, decoded status, LOG DATA, numeric monitor split L/R)
    3) Graph (two windows using pyqtgraph)

Tested with Python 3.13 + PySide6.
"""

import os, sys, time, json, struct
from collections import deque, defaultdict
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import serial
from serial.tools import list_ports
import numpy as np
import pandas as pd

# Favor speed over prettiness
pg.setConfigOptions(antialias=False)      # fastest line rendering
pg.setConfigOptions(useOpenGL=False)      # True can help sometimes, but buggy on some GPUs
pg.setConfigOptions(useNumba=False)       # leave False unless you install numba

# ----------------------- Protocol & constants -----------------------
HEADER = 0x788D3478
LE = "<"
HEADER_BYTES = struct.pack(LE + "I", HEADER)

# Commands (keep IDs consistent with your firmware)
CMD_SEND_DATA        = 0
CMD_STOP_DATA        = 1
CMD_RECEIVE_SETTINGS = 2
CMD_SEND_SETTINGS    = 3    # payload size padded on-wire to FRAME_SIZE_SETTINGS
CMD_SAVE_TO_FLASH    = 4
CMD_RESET_DEFAULTS   = 5
CMD_SEND_CONTROL     = 6
CMD_DIAG_STATUS      = 7
CMD_TEMP_DATA        = 8
CMD_LOG_PAYLOAD      = 9   # Streamed (no CRC) logger payload (superframes or per-frame payload)
CMD_REQUEST_DATA     = 10

# Settings payload size on wire
FRAME_SIZE_SETTINGS  = 16384  # bytes

# Logger defaults
LOG_N_CHANNELS    = 12
LOG_BATCH_SAMPLES = 1000
LOG_PAYLOAD_U16   = LOG_N_CHANNELS * LOG_BATCH_SAMPLES
LOG_PAYLOAD_SIZE  = 2 * LOG_PAYLOAD_U16

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------- CRC16-CCITT -----------------------
def crc16_ccitt(data: bytes, poly=0x1021, init=0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def build_packet(cmd: int, payload: bytes, with_crc: bool = True) -> bytes:
    lb = struct.pack(LE + "H", len(payload))
    if with_crc:
        crc = crc16_ccitt(bytes([cmd]) + lb + payload)
        pkt = HEADER_BYTES + bytes([cmd]) + lb + payload + struct.pack(LE + "H", crc)
    else:
        pkt = HEADER_BYTES + bytes([cmd]) + lb + payload
    return pkt

# ----------------------- Settings schema (names/types/units) -----------------------
# Regular settings (non-log related)
FIELDS = [
    ("motor_type","u8",""),
    ("sensor","u8",""),
    ("motor_connection","u8",""),
    ("sampling_timing","u8",""),
    ("mode","u8",""),
    ("sensored_control","u8",""),
    ("Fs_trq","u16","KHz"),
    ("Fs_speed","u16","KHz"),
    ("Fswitching","u16","KHz"),
    ("zero_sequence","u8",""),

    ("Imax_ph","f32","A_peak"), ("Inom_ph","f32","A_peak"), ("I_o_ph","f32","A_peak"),
    ("overload_interval","f32","s"), ("overload_limiter","f32",""), ("t_drop","f32","s"),

    ("motor_temp_derate_thres","f32","°C"), ("motor_temp_max_thres","f32","°C"),
    ("igbt_temp_derate_thres","f32","°C"), ("igbt_temp_max_thres","f32","°C"),

    ("v_base","f32","Vpeak"), ("I_base","f32","A_peak"),
    ("no_load_rpm","i32","rpm"), ("nom_rpm","i32","rpm"), ("max_rpm","i32","rpm"), ("min_rpm","i32","rpm"),

    ("Rs","f32","Ω"), ("Rr","f32","Ω"), ("Lls","f32","H"), ("Llr","f32","H"),
    ("Lm","f32","H"), ("Lsig","f32","H"),
    ("psi_nom","f32","Wb"), ("psi_min","f32","Wb"),
    ("J","f32","kg·m²"), ("b","f32","N·m·s/rad"), ("np","f32","pairs"),
    ("Ld","f32","H"), ("Lq","f32","H"),

    ("max_trq_num","u32",""),
    ("cc_trise","f32","s"), ("spdc_trise","f32","s"),
    ("mcm_trise","f32","s"), ("scvm_trise","f32","s"), ("fw_trise","f32","s"),

    ("current_ramp_time","f32","ms"), ("velocity_ramp_time","f32","ms"),
    ("spd_ref_offset","f32","rpm"), ("low_rpm_thres","f32","rpm"), ("min_freq","f32","Hz"),

    ("max_power","f32","KW"), ("min_power","f32","KW"),

    ("voltage_trise","f32","s"), ("voltage_tfall","f32","s"),

    ("ocd_thres","f32","A_peak"), ("overvoltage_thres","f32","V"), ("undervoltage_thres","f32","V"),

    ("ma1max","f32","—"), ("ma3","f32","—"),

    ("mcu_max_temp","f32","°C"),

    ("dtclut_flux_hyst","f32","Wb"),
    ("dtclut_trq_hyst","f32","Nm"),
    ("dtclut_trq_hyst_prcnt","u8","%"),
    ("dtclut_trq_hyst_max","f32","Nm"),("dtclut_trq_hyst_min","f32","Nm"),
    ("dtclut_use_prcnt_trq_hyst","u8","—"),
    ("dtcsvm_Kp_flux","f32","—"),("dtcsvm_Ki_flux","f32","—"),
    ("dtcsvm_Kp_trq","f32","—"),("dtcsvm_Ki_trq","f32","—"),
    ("dtcsvm_use_i_x_r","u8","0/1"),("dtcsvm_use_dec","u8","0/1"),
    ("dtc_lamda","f32","0-1"),

    ("wh1_timeout_thres","u32","us"), ("warning_interval","u32","ms"),
    ("error_reset_interval","u32","ms"), ("startup_time","u32","ms"), ("mcu_reset_interval","u32","ms")
    
]

# Log settings (separate from regular settings)
LOG_FIELDS = [
    ("data_logger_ts_div","u32","s/Fs_trq"),
    ("log_channels","u8","—"),
    ("log_scales[100]","f32","—"),  # array of 100 f32
    ("log_slot[LOG_N_CHANNELS]","u8","—"),
    ("params_init_key","u32","—")  # Last member of struct
]

# Combined fields for payload building (order matters for struct packing)
ALL_FIELDS = FIELDS + LOG_FIELDS

# Helper functions for array field parsing
def parse_array_field(name):
    """Parse array field name like 'log_scales[100]' -> ('log_scales', 100)"""
    if '[' in name and ']' in name:
        base = name[:name.index('[')]
        size_str = name[name.index('[')+1:name.index(']')]
        try:
            if size_str == "LOG_N_CHANNELS":
                size = LOG_N_CHANNELS
            else:
                size = int(size_str)
            return (base, size)
        except:
            return (name, 0)
    return (name, 0)

def is_array_field(name):
    """Check if field name represents an array"""
    return '[' in name and ']' in name

# Enums
ENUM_MOTOR_TYPE = {0: "PMSM", 1: "INDUCTION"}
ENUM_SENSOR     = {0: "none", 1: "resolver", 2: "abs_encoder", 3: "inc_encoder"}
ENUM_CONN       = {0: "DELTA", 1: "STAR"}
ENUM_ZERO_SEQ   = {0: "NONE", 1: "3RD_HARMONIC_INJ", 2: "MIN_MAX_INJ"}
ENUM_SAMPLING   = {0: "SYNC", 1: "ASYNC"}
ENUM_MODE       = {0: "FOC", 1: "V_F", 2: "DTC_LUT", 3: "DTC_SVM"}

ENUM_MAP_BY_NAME = {
    "motor_type":       ENUM_MOTOR_TYPE,
    "sensor":           ENUM_SENSOR,
    "motor_connection": ENUM_CONN,
    "sampling_timing":  ENUM_SAMPLING,
    "mode":             ENUM_MODE,
    "zero_sequence":    ENUM_ZERO_SEQ,
}

# LOG_TABLE variables (from firmware LOG_TABLE array, in order)
LOG_TABLE_VARIABLES = [
    "TIME_16US",      # 0
    "TIME_16MS",      # 1
    "Id_ref",         # 2
    "Iq_ref",         # 3
    "Id",             # 4
    "Iq",             # 5
    "psi_ref",        # 6
    "trq_ref",        # 7
    "trq_actual",     # 8
    "psi_actual",     # 9
    "psi_ref",        # 10 (duplicate)
    "trq_ref",        # 11 (duplicate)
    "trq_actual",     # 12 (duplicate)
    "sector",         # 13
    "vector",         # 14
    "Ia_ph",          # 15
    "Ib_ph",          # 16
    "pwr_ref",        # 17
    "pwr_actual",     # 18
    "motor_rpm",      # 19
    "Vdc",            # 20
]

# DIAG fields (name, type). Keep aligned with firmware order.
DIAG_FIELDS = [
    ("critical_hw_status","u16"),
    ("aux_hw_status","u8"),
    ("last_error","u8"),
    ("sys_status","u8"), ("handler_status","u8"),
    ("latched_error","u8"), ("actual_state","u8"), ("control_mode_actual","u8"),
    ("limiter_flags","u8"),

    ("Id_ref","f32"), ("Iq_ref","f32"),
    ("Id_filt","f32"), ("Iq_filt","f32"),
    ("trq_ref","f32"), ("trq_actual","f32"),
    ("flux_ref","f32"), ("flux_actual","f32"),
    ("pwr_ref","f32"), ("pwr_actual","f32"),

    ("motor_rpm","i16"),
    ("Vdc","f32"),
    ("Vdc_max","f32"),
    ("num_max_trq","i16"),
    ("vcu_max_velocity","u16"),
    ("Imax_total","f32"), ("Imin_total","f32"),
    ("Iq_ref_request","f32"),

    ("max_velocity","i16"), ("min_velocity","i16"),

    ("motor_temp","f32"),
    ("igbt_temp","f32"),
    ("nominal_rpm","f32"),
    ("I1","f32"), ("I2","f32"), ("I3","f32"),
    ("I1max","f32"), ("I2max","f32"), ("I3max","f32"),
    ("ambient_temp","f32"), ("ambient_humidity","f32"),
    ("cc_time","f32"), ("cc_ts","f32"),
    ("sp_time","f32"), ("sp_ts","f32"),
]

# Units for DIAG fields used in the numeric monitor
UNITS = {
    "motor_rpm": "rpm",
    "trq_ref": "Nm",
    "trq_actual": "Nm",
    "Vdc": "V",
    "motor_temp": "°C",
    "igbt_temp": "°C",
    "I1": "A", "I2": "A", "I3": "A",
    "I1max": "A", "I2max": "A", "I3max": "A",
    "ambient_temp": "°C", "ambient_humidity": "%",
}

# Fields that are already summarized in the decoded Status box
STATUS_BOX_FIELDS = {
    "critical_hw_status",
    "aux_hw_status",
    "last_error",
    "sys_status",
    "handler_status",
    "latched_error",
    "actual_state",
    "control_mode_actual",
}

# ----- Temp-data (MCU post-log) binary layout -----
# Order is exactly as in capture_local_monitor()
TEMP_CTRL_FIELDS = [
    ("num_max_trq",        "h"),  # i16
    ("num_min_trq",        "h"),  # i16
    ("vcu_max_velocity",   "h"),  # i16 (or 'H' if unsigned)
    ("vcu_min_velocity",   "h"),  # i16
    ("velocity_req",       "h"),  # i16
    ("torque_req",         "h"),  # i16  (num_trq_req)
    ("mode",               "B"),  # u8   (control_mode_command)
    ("inv_en",             "B"),  # u8   (invstatus.vcu_command: 0/1)
]
# Build a struct fmt for the DIAG part straight from DIAG_FIELDS
_DIAG_FMT = "<" + "".join({
    "u8":"B","u16":"H","i16":"h","i32":"i","u32":"I","f32":"f"
}[t] for _, t in DIAG_FIELDS)
_TEMP_CTRL_FMT = "<" + "".join(t for _, t in TEMP_CTRL_FIELDS)
TEMP_ROW_SIZE = struct.calcsize(_TEMP_CTRL_FMT) + struct.calcsize(_DIAG_FMT)

ERRORS_MAP = {
    0:"INVERTER_OK",1:"OCD_FAULT",2:"INVERTED_PHASE_POLARITY",3:"HW_FAULT",
    4:"OVERVOLTAGE",5:"UNDERVOLTAGE",6:"IGBT_OVERTEMP",7:"MOTOR_OVERTEMP",
    8:"ADC_ERROR",9:"ADC_INIT_ERROR",10:"SPI_ERROR",11:"ENCODER_READ_ERROR",
    12:"ENDAT_INIT_ERROR",13:"VELOCITY_EXCEEDS_MAXIMUM",14:"PWM_OVERMODULATION",
    15:"INVERSE_TRANSFORM_TIMEOUT",16:"TIMER_INIT_ERROR",17:"PWM_START_FAILURE",
    18:"PWM_STOP_FAILURE",19:"INVALID_CONTROL_MODE",20:"CONFIGURATION_ERROR",
    21:"WHILE1_TIMEOUT",22:"CANRX_ERROR",23:"CANTX_ERROR",24:"COMMUNICATION_TIMEOUT_ERROR",
    25:"SUPPLY_3V3_ERROR",26:"SUPPLY_5V0_ERROR",27:"SUPPLY_5V6_ERROR",
    28:"SUPPLY_12V0_ERROR",29:"VREF_ERROR",30:"NO_LV_SUPPLY",31:"MCU_OVERTEMP",
    32:"MCU_DISABLE_IMPLAUSIBILITY",33:"MCU_ENABLE_IMPLAUSIBILITY",34:"GD_DISABLE_IMPLAUSIBILITY",
}
ERRORTYPE_MAP = {0:"NO_ERRORS",1:"FATAL_ERROR",2:"CRITICAL_ERROR",3:"WARNING",4:"INITIALIZING"}
LATCHED_MAP    = {0:"ERROR_RESET",1:"FATAL_ERROR_EXISTING",2:"CRITICAL_ERROR_EXISTING",3:"WARNING_EXISTING"}
STATE_MAP      = {0:"INVERTER_DISABLE", 1:"INVERTER_ENABLE"}
CTRL_MODE_MAP  = {0:"FAULT_MODE",1:"SPEED_CONTROL",2:"TORQUE_CONTROL"}

LIMITER_FLAG_BITS = [
    (5, "SpeedLimiter"),
    (4, "PowerLimiter"),
    (3, "StallLimiter"),
    (2, "I2tLimiter"),
    (1, "MotorTemp"),
    (0, "IGBTTemp"),
]

def enum_name(d, v): return d.get(int(v), "?")

def fmt_bits(mask: int, table):
    out = [name for bit,name in table if (mask & (1<<bit))]
    return " | ".join(out) if out else "0"

def fmt_crit(mask: int):
    base = "HW_INVERTER_DISABLED" if (mask & 1) else "HW_INVERTER_ENABLED"
    flags = [name for bit,name in [
        (1,"HW_OCD_A"), (2,"HW_OCD_B"), (3,"HW_OCD_C"),
        (4,"HW_INV_ERROR"), (5,"HW_OVERVOLTAGE"), (6,"HW_FAULT_LATCHED"),
        (7,"HW_MCU_DISABLE"), (8,"HW_GD_FLT"), (9,"HW_GD_RDY_ERROR"),
        (10,"HW_SENSOR_A_DC"), (11,"HW_SENSOR_B_DC"), (12,"HW_SENSOR_C_DC"),
    ] if (mask & (1<<bit))]
    return " | ".join([base] + flags) if flags else base

# ----------------------- Serial Reader Thread -----------------------
class SerialReader(QtCore.QThread):
    sig_frame  = QtCore.Signal(int, bytes)     # (cmd, payload) for CRC'd frames
    sig_logger = QtCore.Signal(bytes)          # payload for CMD_LOG_PAYLOAD (no CRC)
    sig_status = QtCore.Signal(str)

    def __init__(self, ser: serial.Serial, parent=None):
        super().__init__(parent)
        self._ser = ser
        self._running = True
        self._buf = bytearray()

    def stop(self):
        self._running = False

    def run(self):
        read = self._ser.read
        buf = self._buf
        find = buf.find
        while self._running:
            try:
                data = read(max(self._ser.in_waiting, 4096) or 1024)
                if data:
                    buf.extend(data)

                # Try to parse frames
                while True:
                    idx = find(HEADER_BYTES)
                    if idx == -1:
                        if len(buf) > 3:
                            del buf[:len(buf)-3]
                        break
                    if idx > 0:
                        del buf[:idx]
                    if len(buf) < 4 + 1 + 2:
                        break
                    cmd = buf[4]
                    pay_len = struct.unpack_from(LE + "H", buf, 5)[0]

                    if cmd == CMD_LOG_PAYLOAD:
                        total = 4 + 1 + 2 + pay_len
                        if len(buf) < total:
                            break
                        payload = bytes(buf[7:7+pay_len])
                        del buf[:total]
                        self.sig_logger.emit(payload)
                        continue
                    else:
                        total = 4 + 1 + 2 + pay_len + 2
                        if len(buf) < total:
                            break
                        payload = bytes(buf[7:7+pay_len])
                        rx_crc = struct.unpack_from(LE + "H", buf, 7 + pay_len)[0]
                        calc_crc = crc16_ccitt(bytes([cmd]) + struct.pack(LE+"H", pay_len) + payload)
                        del buf[:total]
                        if rx_crc != calc_crc:
                            self.sig_status.emit(f"CRC mismatch on CMD {cmd}")
                            continue
                        self.sig_frame.emit(cmd, payload)
                        continue

            except Exception as e:
                self.sig_status.emit(f"Serial read error: {e}")
                time.sleep(0.05)

class DataLoggerQt:
    """
    File-based logger for the shared-port Qt GUI.
    - Collects raw CMD_LOG_PAYLOAD payloads into a .bin file
    - Builds a CSV that matches csv_write_f16.py scaling
    - Injects SETTINGS (vertical) starting at column P
    - Optionally appends a MONITOR table (control + diag snapshot rows)
    """

    # ----- scaling (mirror csv_write_f16.py) -----
    SHIFT_BITS   = 8
    SCALE_FLOATS = True
    SCALE_CH1    = False
    CHANNEL_MULTIPLIERS = [
        256,        # CH1 (counter; often left raw)
        256,   # CH2
        256,   # CH3
        256,   # CH4
        256,   # CH5
        256,     # CH6
        256,     # CH7
        256,   # CH8
        256,   # CH9
        256,   # CH10
        256,   # CH11
        256    # CH12
    ]

    def __init__(self, app, n_channels=12, data_dir="data"):
        self.app = app
        self.NC = int(n_channels)
        self.DATA_DIR = data_dir
        os.makedirs(self.DATA_DIR, exist_ok=True)

        # column P (16th) anchor
        self.COL_P = 16
        self._bin = None
        self.bin_path = None
        self.csv_path = None

        # optional extra rows (control/monitor) to append after stop
        self.monitor_rows = []   # list of dicts

    # ---------- helpers ----------
    @staticmethod
    def _u16_to_s16(u: int) -> int:
        return u - 0x10000 if u >= 0x8000 else u

    def _mult(self, i: int) -> float:
        if i < len(self.CHANNEL_MULTIPLIERS):
            m = self.CHANNEL_MULTIPLIERS[i]
            return float(m) if m not in (0, 0.0) else 1.0
        return 1.0

    def _decode_ch_num(self, i: int, u16_val: int) -> float:
        if not self.SCALE_FLOATS:
            return float(int(u16_val))
        if i == 0 and not self.SCALE_CH1:
            return float(int(u16_val))
        s16 = self._u16_to_s16(int(u16_val))
        return (s16 << self.SHIFT_BITS) / self._mult(i)

    def add_monitor_rows(self, rows: list[dict]):
        for r in rows or []:
            self.add_monitor_row(r)

    # ---------- lifecycle ----------
    def start(self, basename: str):
        self.bin_path = os.path.join(self.DATA_DIR, basename + ".bin")
        self.csv_path = os.path.join(self.DATA_DIR, basename + ".csv")
        self._bin = open(self.bin_path, "wb", buffering=1024*1024)

    def write_payload(self, payload: bytes):
        if self._bin:
            self._bin.write(payload)

    def close_bin(self):
        try:
            if self._bin:
                self._bin.flush()
                self._bin.close()
        except Exception:
            pass
        self._bin = None

    def add_monitor_row(self, row_dict: dict):
        """Append one control/monitor snapshot row (keys: ctrl + DIAG_FIELDS names)."""
        self.monitor_rows.append(dict(row_dict))

    # ---------- CSV build ----------
    def _settings_sidepanel_lines(self) -> list[str]:
        # Prefer MCU-received values
        settings_dict = dict(self.app._recv_values) if getattr(self.app, "_recv_values", None) else {}

        # Fallback to current send widgets if something is missing
        for (name, typ, _unit) in FIELDS:
            if name not in settings_dict:
                if name in ENUM_MAP_BY_NAME:
                    combo = self.app.enum_senders.get(name)
                    if combo:
                        settings_dict[name] = int(combo.currentData())
                elif name == "sensored_control":
                    cb = self.app.send_vars.get(name)
                    if cb:
                        settings_dict[name] = 1 if cb.isChecked() else 0
                else:
                    w = self.app.send_vars.get(name)
                    if isinstance(w, QtWidgets.QLineEdit):
                        txt = w.text().strip()
                        try:
                            if   typ in ("u8","u16","u32","i32"): settings_dict[name] = int(txt or "0")
                            elif typ == "f32":                   settings_dict[name] = float(txt or "0")
                        except Exception:
                            settings_dict[name] = 0

        lines = ["#SETTINGS,"]
        for (name, typ, _unit) in FIELDS:
            if name not in settings_dict: 
                continue
            val = settings_dict[name]
            if name in ENUM_MAP_BY_NAME:
                out = enum_name(ENUM_MAP_BY_NAME[name], int(val))
            elif name == "sensored_control":
                out = "ENABLED" if int(val) else "DISABLED"
            else:
                out = f"{val}"
            lines.append(f"{name},{out}")
        lines.append("")  # blank separator

        # MONITOR header
        ctrl_header = ["num_max_trq","num_min_trq","max_velocity","min_velocity",
                       "velocity_req","torque_req","mode","inv_en"]
        mon_header  = [n for (n, _t) in DIAG_FIELDS]
        lines.append("#MONITOR,")
        lines.append(",".join(ctrl_header + mon_header))

        # helper mappers
        def map_control_row(row):
            out = dict(row)
            if "mode" in out:
                out["mode"] = enum_name(CTRL_MODE_MAP, int(out["mode"]))
            if "inv_en" in out:
                out["inv_en"] = "ENABLED" if int(out["inv_en"]) else "DISABLED"
            return out

        def map_monitor_row(row):
            out = dict(row)
            if "critical_hw_status" in out:
                out["critical_hw_status"] = fmt_crit(int(out["critical_hw_status"]))
            if "aux_hw_status" in out:
                out["aux_hw_status"] = fmt_bits(int(out["aux_hw_status"]),
                                                [(b, name) for b, name in LIMITER_FLAG_BITS])  # reuse bits if you wish
            for k in ("last_error","sys_status"):
                if k in out:
                    out[k] = enum_name(ERRORS_MAP, int(out[k]))
            if "handler_status" in out:
                out["handler_status"] = enum_name(ERRORTYPE_MAP, int(out["handler_status"]))
            if "latched_error" in out:
                out["latched_error"] = enum_name(LATCHED_MAP, int(out["latched_error"]))
            if "actual_state" in out:
                out["actual_state"] = enum_name(STATE_MAP, int(out["actual_state"]))
            if "control_mode_actual" in out:
                out["control_mode_actual"] = enum_name(CTRL_MODE_MAP, int(out["control_mode_actual"]))
            if "limiter_flags" in out:
                out["limiter_flags"] = fmt_bits(int(out["limiter_flags"]), LIMITER_FLAG_BITS)
            return out

        # monitor rows (if any)
        for row in self.monitor_rows:
            cvals = map_control_row({k: row.get(k, "") for k in ctrl_header})
            mvals = map_monitor_row({k: row.get(k, "") for k in mon_header})
            lines.append(",".join([str(cvals[k]) for k in ctrl_header] +
                                  [str(mvals[k]) for k in mon_header]))

        return lines

    def build_csv(self):
        if not (self.bin_path and os.path.isfile(self.bin_path)):
            raise RuntimeError("BIN not found for CSV build")

        # side panel strings and padding
        side_lines = self._settings_sidepanel_lines()
        pad_along = "," * max(0, (self.COL_P - 1 - self.NC))

        # robust stream decode (fixes the 'line 1000' drift):
        raw = np.fromfile(self.bin_path, dtype="<u2")
        usable = (raw.size // self.NC) * self.NC
        if usable != raw.size:
            raw = raw[:usable]
        arr = raw.reshape(-1, self.NC)

        with open(self.csv_path, "w", encoding="utf-8", newline="") as fc:
            # header
            fc.write(",".join([f"CH{i+1}" for i in range(self.NC)]) + pad_along + ",#SETTINGS,\n")

            side_idx = 0
            # write rows
            for k in range(arr.shape[0]):
                row_u16 = arr[k]
                fields = [f"{self._decode_ch_num(i, int(row_u16[i])):.9g}" for i in range(self.NC)]
                if side_idx < len(side_lines):
                    fc.write(",".join(fields) + pad_along + "," + side_lines[side_idx] + "\n")
                    side_idx += 1
                else:
                    fc.write(",".join(fields) + "\n")

        # remove BIN to keep folder tidy
        try:
            os.remove(self.bin_path)
        except Exception:
            pass

    def set_post_lines(self, lines):
        """Provide raw post-stop MCU lines to be dumped at column P."""
        self._post_lines = list(lines or [])

    def _write_post_lines_block(self, fc, pad_along):
        """Emit a '#POST_LOG_MCU' block at column P."""
        pls = getattr(self, "_post_lines", [])
        if not pls:
            return
        fc.write(pad_along + ",#POST_LOG_MCU,\n")
        for ln in pls:
            # write each MCU-provided line verbatim in the side panel
            fc.write(pad_along + "," + ln + "\n")
        fc.write("\n")

# ----------------------- Utility widgets -----------------------
class Led(QtWidgets.QLabel):
    def __init__(self, diameter=14, on_color="#36c275", off_color="#b5b5b5", parent=None):
        super().__init__(parent)
        self._d = diameter
        self._on = QtGui.QColor(on_color)
        self._off = QtGui.QColor(off_color)
        self._state = False
        self.setFixedSize(diameter, diameter)

    def setState(self, state: bool):
        self._state = bool(state)
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        color = self._on if self._state else self._off
        p.setBrush(QtGui.QBrush(color))
        p.setPen(QtGui.QPen(QtGui.QColor("#777777")))
        r = QtCore.QRectF(1,1,self._d-2,self._d-2)
        p.drawEllipse(r)
        p.end()

class BitIndicator(QtWidgets.QFrame):
    """Small rectangular indicator with text label."""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self._active = False
        self._on_col = QtGui.QColor("#cc3333")  # red when active

        self.setFixedSize(90, 24)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)

        self.text_lbl = QtWidgets.QLabel(label)
        self.text_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.text_lbl.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.text_lbl.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self.text_lbl, alignment=QtCore.Qt.AlignCenter)

        self._apply_theme_colors()  # initial (inactive)

    def _apply_theme_colors(self):
        """Apply theme-matching 'off' style (transparent bg, themed border/text)."""
        pal = self.palette()
        border = pal.color(QtGui.QPalette.Dark)
        text   = pal.color(QtGui.QPalette.Text)

        # transparent background so it inherits the container (light or dark)
        self.setStyleSheet(
            f"QFrame{{background: transparent; border:1px solid {border.name()}; border-radius:5px;}}"
        )
        self.text_lbl.setStyleSheet(
            f"color:{text.name()}; background: transparent; border: none;"
        )

    def setActive(self, state: bool):
        """Turn indicator on/off (red when active)."""
        self._active = bool(state)
        if self._active:
            self.setStyleSheet(
                "QFrame{background:#cc3333; border:1px solid #990000; border-radius:5px;}"
            )
            self.text_lbl.setStyleSheet("color:white; background: transparent; border: none;")
        else:
            self._apply_theme_colors()


# ---------- Multi-select combobox for plot signal selection ----------
class MultiSelectCombo(QtWidgets.QComboBox):
    checkedChanged = QtCore.Signal()

    def __init__(self, items=None, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.setMaxVisibleItems(20)
        self._model = QtGui.QStandardItemModel(self)
        self.setModel(self._model)
        if items:
            self.set_items(items)
        self.view().pressed.connect(self._on_index_pressed)

    def set_items(self, items):
        self._model.clear()
        for name in items:
            it = QtGui.QStandardItem(name)
            it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            it.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
            self._model.appendRow(it)
        self._refresh_text()

    def _on_index_pressed(self, index):
        item = self._model.itemFromIndex(index)
        st = QtCore.Qt.Checked if item.checkState() == QtCore.Qt.Unchecked else QtCore.Qt.Unchecked
        item.setCheckState(st)
        self._refresh_text()
        self.checkedChanged.emit()

    def checked_items(self):
        return [self._model.item(i).text()
                for i in range(self._model.rowCount())
                if self._model.item(i).checkState() == QtCore.Qt.Checked]

    def _refresh_text(self):
        txt = ", ".join(self.checked_items())
        self.lineEdit().setText(txt if txt else "— select —")

# ----------------------- Plot widgets (pyqtgraph) -----------------------
class TwoWindowPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- single, persistent layout ----
        self._vbox = QtWidgets.QVBoxLayout(self)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(6)

        # ---- plots (create first!) ----
        self.plot1 = pg.PlotWidget()
        self.plot2 = pg.PlotWidget()
        for pw in (self.plot1, self.plot2):
            pw.showGrid(x=True, y=True)
            if pw.plotItem.legend is None:
                pw.addLegend()
        self._vbox.addWidget(self.plot1)
        self._vbox.addWidget(self.plot2)

        for pw in (self.plot1, self.plot2):
            pw.setDownsampling(auto=True)      # let pg decimate while drawing
            pw.setClipToView(True)
            pw.plotItem.setMouseEnabled(x=True, y=True)

        # state
        self._curves = {}            # keys: (win_id, name) -> PlotDataItem
        self._palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#ff33a6", "#33ff77"
        ]
        self._ch_pen = {}            # name -> QPen

        # safe defaults
        self.set_groups(
            win1_groups=[("CH2","CH3"), ("CH4","CH5"), ("CH6","CH7")],
            win2_groups=[("CH8","CH9"), ("CH10","CH11","CH12")],
            win1_titles=["Flux","Torque","Sector/Vector"],
            win2_titles=["Ia/Ib","CH10-12"],
            link_x=True
        )

    # ---------- public API ----------
    def set_groups(
        self,
        win1_groups=None,
        win2_groups=None,
        win1_titles=None,
        win2_titles=None,
        link_x=True,
        **kwargs
    ):
        """Rebuild the two windows with the provided channel groups."""
        groups1 = win1_groups or kwargs.get("groups1") or []
        groups2 = win2_groups or kwargs.get("groups2") or []

        # Flatten group tuples into a unique ordered list per window
        def _flatten(groups):
            seen, out = set(), []
            for tup in groups:
                for name in tup:
                    if name not in seen:
                        seen.add(name); out.append(name)
            return out

        names1 = _flatten(groups1)
        names2 = _flatten(groups2)

        # Clear old curves only (do NOT re-add widgets/legends)
        for (win_id, name), curve in list(self._curves.items()):
            if ((win_id == 1 and name not in names1) or
                (win_id == 2 and name not in names2)):
                try:
                    (self.plot1 if win_id == 1 else self.plot2).removeItem(curve)
                except Exception:
                    pass
                del self._curves[(win_id, name)]

        # Ensure required curves exist
        def _ensure(win_id, plotw, names):
            for name in names:
                key = (win_id, name)
                if key not in self._curves:
                    curve = plotw.plot(name=name, pen=self._pen_for(name))
                    self._curves[key] = curve

        _ensure(1, self.plot1, names1)
        _ensure(2, self.plot2, names2)

        # Titles (accept list or str)
        if isinstance(win1_titles, (list, tuple)) and len(win1_titles) > 0:
            self.plot1.setTitle("; ".join(map(str, win1_titles)))
        elif isinstance(win1_titles, str):
            self.plot1.setTitle(win1_titles)

        if isinstance(win2_titles, (list, tuple)) and len(win2_titles) > 0:
            self.plot2.setTitle("; ".join(map(str, win2_titles)))
        elif isinstance(win2_titles, str):
            self.plot2.setTitle(win2_titles)

        # X-linking
        if link_x:
            self.plot2.setXLink(self.plot1)
        else:
            self.plot2.setXLink(None)

    def set_zoom_mode(self, mode: str):
        vb1 = self.plot1.getViewBox()
        vb2 = self.plot2.getViewBox()
        if mode == "x":
            for vb in (vb1, vb2):
                vb.setMouseMode(pg.ViewBox.RectMode)
                vb.disableAutoRange(axis=pg.ViewBox.YAxis)
                vb.enableAutoRange(axis=pg.ViewBox.XAxis)
        elif mode == "y":
            for vb in (vb1, vb2):
                vb.setMouseMode(pg.ViewBox.RectMode)
                vb.disableAutoRange(axis=pg.ViewBox.XAxis)
                vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            for vb in (vb1, vb2):
                vb.enableAutoRange(axis=pg.ViewBox.XYAxes)
                vb.autoRange()

    def zoom_fit(self):
        """Reset both plots to show all data (not cumulative unzoom)."""
        for plot in (self.plot1, self.plot2):
            vb = plot.getViewBox()
            # 1) disable auto range so we can set manually
            vb.disableAutoRange()
            # 2) compute visible bounds from all curves
            xmins, xmaxs, ymins, ymaxs = [], [], [], []
            for item in plot.listDataItems():
                if item is None or item.xData is None or item.yData is None:
                    continue
                x, y = item.xData, item.yData
                if len(x) == 0:
                    continue
                xmins.append(np.nanmin(x))
                xmaxs.append(np.nanmax(x))
                ymins.append(np.nanmin(y))
                ymaxs.append(np.nanmax(y))
            if xmins:
                xmin, xmax = min(xmins), max(xmaxs)
                ymin, ymax = min(ymins), max(ymaxs)
                if xmax == xmin:
                    xmax += 1.0
                if ymax == ymin:
                    ymax += 1.0
                vb.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0.05)
            # 3) re-enable autorange for next user zooms
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes)

    def clear(self):
        # Remove curves but keep widgets
        for curve in list(self._curves.values()):
            try:
                curve.getViewBox().removeItem(curve)
            except Exception:
                pass
        self._curves.clear()
        self.plot1.clear(); 
        if self.plot1.plotItem.legend is None: self.plot1.addLegend()
        self.plot1.showGrid(x=True, y=True)
        self.plot2.clear();
        if self.plot2.plotItem.legend is None: self.plot2.addLegend()
        self.plot2.showGrid(x=True, y=True)

    def plot_csv(self, csv_path: str, groups1, groups2, titles1=None, titles2=None, link_x=True):
        # Load only CH* columns, tolerate ragged rows
        try:
            df = pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines="skip",
                usecols=lambda c: str(c).startswith("CH")
            )
        except Exception:
            cols = [f"CH{i}" for i in range(1, 13)]
            data = []
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()
                for line in f:
                    parts = [p.strip() for p in line.split(",")]
                    row = []
                    for p in parts[:12]:
                        try: row.append(float(p))
                        except: row.append(np.nan)
                    if any(np.isfinite(row)):
                        data.append(row)
            df = pd.DataFrame(data, columns=cols)

        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if not any(col.startswith("CH") for col in df.columns):
            raise ValueError("No CH* columns found in CSV.")

        # (A) Configure groups/curves first
        self.set_groups(groups1, groups2, titles1, titles2, link_x)

        # (B) Pre-build a *shared* X and use coarse decimation for speed
        n = len(df.index)
        # target ~20k points per curve max (tweakable)
        ds = max(1, n // 20000)
        x_full = np.arange(n, dtype=np.float32)
        x = x_full[::ds]

        # (C) Push data
        for (win_id, name), curve in self._curves.items():
            if name in df.columns:
                y_col = df[name].to_numpy(np.float32)
                # slice once; skipFiniteCheck=True assumes no NaNs after dropna
                curve.setData(x=x, y=y_col[::ds], connect='finite', skipFiniteCheck=True)


    # ---------- helpers ----------
    def _pen_for(self, name: str):
        if name not in self._ch_pen:
            idx = len(self._ch_pen) % len(self._palette)
            self._ch_pen[name] = pg.mkPen(self._palette[idx], width=1.2)
        return self._ch_pen[name]


# ----------------------- Main Window -----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inverter GUI — PySide6")
        self.resize(1400, 900)

        self._recv_values = {}

        self._postlog_lines = []   # lines received from MCU after STOP (one big message)
        self._postlog_rows = []   # list of dict rows parsed from CMD_TEMP_DATA
        self._awaiting_postlog = False
        self._staged_pc_row = None
        self._stop_timeout = QtCore.QTimer(self)
        self._stop_timeout.setSingleShot(True)
        self._stop_timeout.timeout.connect(self._finalize_logging)

        self.dlogger = None
        self._last_diag_values = {}   # store most recent DIAG_STATUS dict

        # ============ Live plot buffers ============
        self.live_max_samples = 500  # default window size (adjustable)
        self.live_time = deque(maxlen=self.live_max_samples)
        self.live_t0 = time.monotonic()
        self.live_buf = defaultdict(lambda: deque(maxlen=self.live_max_samples))
        self.plots_running = True     # (5) start/stop plots
        self._color_map = {}          # stable color per signal name

        # === Status bar ===
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # === Central tabs ===
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # ============ Serial state ============
        self.ser = None
        self.reader = None
        self.desired_port = None
        self.baud = 12_000_000

        # ============ Logger state ============
        self.logging = False
        self.log_bin_file = None
        self.log_basename = None
        self.SCALE_FLOATS = True
        self.SHIFT_BITS = 8
        self.SCALE_CH1 = False
        self.CHANNEL_MULTIPLIERS = [1.0] + [100000.0]*11

        self._ctrl_vals = {
            "num_max_trq": 0,
            "num_min_trq": 0,
            "max_velocity": 0,
            "min_velocity": 0,
            "velocity_req": 0,
            "torque_req": 0,
            "mode": 1,   # SPEED_CONTROL
            "inv_en": 0,
        }

        # ============ Build UI ============
        self._build_toolbar()
        self._build_settings_tab()
        self._build_ctrlmon_tab()
        self._build_graph_tab()

        # Theme
        QtWidgets.QApplication.setStyle("fusion")
        self.setPalette(self.palette())

        self._build_menus()
        self._apply_theme("inherited")

        # === single QTimer for periodic control ===
        self.periodic_timer = QtCore.QTimer(self)
        self.periodic_timer.setSingleShot(False)
        self.periodic_timer.timeout.connect(self._send_control_once)

        # === separate QTimer for periodic monitor request ===
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.setSingleShot(False)
        self.monitor_timer.timeout.connect(self._send_request_data)

    # ---------------- Toolbar / top controls ----------------
    def _build_toolbar(self):
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        self.usb_led = Led()
        tb.addWidget(self.usb_led)
        self.usb_lbl = QtWidgets.QLabel("USB: Disconnected")
        tb.addWidget(self.usb_lbl)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel("Port:"))
        self.port_combo = QtWidgets.QComboBox()
        tb.addWidget(self.port_combo)
        btn_scan = QtWidgets.QToolButton(); btn_scan.setText("Refresh")
        btn_scan.clicked.connect(self._refresh_ports)
        tb.addWidget(btn_scan)

        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Baud:"))
        self.baud_edit = QtWidgets.QLineEdit(str(self.baud)); self.baud_edit.setFixedWidth(100)
        tb.addWidget(self.baud_edit)

        btn_conn = QtWidgets.QToolButton(); btn_conn.setText("Connect / Reconnect")
        btn_conn.clicked.connect(self._user_connect)
        tb.addWidget(btn_conn)

        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel("View:"))
        self.view_combo = QtWidgets.QComboBox()
        self._refresh_csv_list()
        tb.addWidget(self.view_combo)
        btn_view_refresh = QtWidgets.QToolButton(); btn_view_refresh.setText("⟳")
        btn_view_refresh.clicked.connect(self._refresh_csv_list)
        tb.addWidget(btn_view_refresh)
        btn_view_open = QtWidgets.QToolButton(); btn_view_open.setText("VIEW DATA")
        btn_view_open.clicked.connect(self._open_csv_in_graph_tab)
        tb.addWidget(btn_view_open)

        self.options_btn = QtWidgets.QToolButton()
        self.options_btn.setToolTip("Options")
        self.options_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.options_btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.options_btn.setIconSize(QtCore.QSize(16, 16))

        # Try to load a proper “settings/gear” icon; fall back to ⚙ if not available
        ico = QtGui.QIcon.fromTheme("preferences-system")
        if ico.isNull():
            ico = QtGui.QIcon.fromTheme("settings")
        if ico.isNull():
            # fallback: show unicode gear if we couldn't find a theme icon
            self.options_btn.setText("⚙")
        else:
            self.options_btn.setIcon(ico)

        # menu will be attached later in _build_menus()
        tb.addWidget(self.options_btn)

        self._refresh_ports()

    def _set_status(self, s: str):
        self.status.showMessage(s, 5000)

    def _refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        if ports:
            if self.desired_port and self.desired_port in ports:
                self.port_combo.setCurrentText(self.desired_port)
            else:
                self.port_combo.setCurrentIndex(0)
        self._set_status(f"Ports: {ports}")

    def _user_connect(self):
        self.desired_port = self.port_combo.currentText()
        try:
            b = int(self.baud_edit.text().strip())
            if b > 0:
                self.baud = b
        except ValueError:
            pass
        self._close_serial()
        self._open_serial()

    def _open_serial(self):
        if not self.desired_port:
            self._set_status("No port selected")
            return
        try:
            self.ser = serial.Serial(self.desired_port, self.baud, timeout=0.02, write_timeout=0)
            self._set_status("Connected ✅")
            self.usb_led.setState(True)
            self.usb_lbl.setText("USB: Connected")
            self.reader = SerialReader(self.ser)
            self.reader.sig_frame.connect(self._handle_frame)
            self.reader.sig_logger.connect(self._handle_log_payload)
            self.reader.sig_status.connect(self._set_status)
            self.reader.start()
            # Automatically request settings on connection
            QtCore.QTimer.singleShot(100, self._read_settings)  # Small delay to ensure connection is ready
        except Exception as e:
            self.ser = None
            self.usb_led.setState(False)
            self.usb_lbl.setText("USB: Disconnected")
            self._set_status(f"Open failed: {e}")

    def _close_serial(self):
        try:
            if self.reader:
                self.reader.stop()
                self.reader.wait(500)
        except Exception:
            pass
        self.reader = None
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.usb_led.setState(False)
        self.usb_lbl.setText("USB: Disconnected")
        self.periodic_timer.stop()  # ensure periodic stops when link closes
        self._set_status("Disconnected ⛔")
        self.periodic_timer.stop()
        self.monitor_timer.stop()   # NEW

    # ---------------- Settings tab ----------------
    def _build_settings_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        actions = QtWidgets.QHBoxLayout()
        btn_prog = QtWidgets.QPushButton("Program Settings → MCU"); btn_prog.clicked.connect(self._send_settings)
        btn_read = QtWidgets.QPushButton("Read Settings ← MCU");   btn_read.clicked.connect(self._read_settings)
        btn_saveflash = QtWidgets.QPushButton("Save to Flash");    btn_saveflash.clicked.connect(lambda: self._send_simple(CMD_SAVE_TO_FLASH, b"", "Save-to-Flash sent ✅"))
        btn_resetdef  = QtWidgets.QPushButton("Reset to Defaults");btn_resetdef.clicked.connect(lambda: self._send_simple(CMD_RESET_DEFAULTS, b"", "Reset-Defaults sent ✅"))
        for b in (btn_prog, btn_read):
            actions.addWidget(b)
        actions.addSpacing(20)
        for b in (btn_saveflash, btn_resetdef):
            actions.addWidget(b)
        actions.addStretch(1)

        btn_load_json = QtWidgets.QPushButton("Load (.json)"); btn_load_json.clicked.connect(self._load_json)
        btn_save_json = QtWidgets.QPushButton("Save (.json)"); btn_save_json.clicked.connect(self._save_json)
        actions.addWidget(btn_load_json); actions.addWidget(btn_save_json)

        btn_copy = QtWidgets.QPushButton("Copy MCU → Current"); btn_copy.clicked.connect(self._copy_recv_to_send)
        actions.addWidget(btn_copy)

        v.addLayout(actions)

        # Create sub-tabs for General Settings and Log Settings
        sub_tabs = QtWidgets.QTabWidget()
        
        # General Settings tab
        general_tab = self._build_general_settings_tab()
        sub_tabs.addTab(general_tab, "General Settings")
        
        # Log Settings tab
        log_tab = self._build_log_settings_tab()
        sub_tabs.addTab(log_tab, "Log Settings")
        
        v.addWidget(sub_tabs)

        grp_hex = QtWidgets.QGroupBox("Packet Preview (HEX)"); grp_hex.setCheckable(True); grp_hex.setChecked(False)
        hv = QtWidgets.QVBoxLayout(grp_hex)
        self.hex_box = QtWidgets.QPlainTextEdit(); self.hex_box.setReadOnly(True); self.hex_box.setMinimumHeight(60); self.hex_box.setMaximumHeight(120)
        self.hex_box.setWordWrapMode(QtGui.QTextOption.NoWrap)
        def _hex_toggle(checked): self.hex_box.setVisible(checked)
        grp_hex.toggled.connect(_hex_toggle); self.hex_box.setVisible(False)
        hv.addWidget(self.hex_box); v.addWidget(grp_hex)

        self.tabs.addTab(w, "Settings")

    def _build_general_settings_tab(self):
        """Build the general settings tab with regular settings fields."""
        w = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(4)

        if not hasattr(self, 'send_vars'):
            self.send_vars = {}
        if not hasattr(self, 'recv_vars'):
            self.recv_vars = {}
        if not hasattr(self, 'enum_senders'):
            self.enum_senders = {}
        if not hasattr(self, 'enum_receivers'):
            self.enum_receivers = {}

        COLUMNS = 4
        row_per_col = (len(FIELDS) + COLUMNS - 1) // COLUMNS

        def add_row(row, col, name, typ, unit):
            basec = col*5
            r = row
            grid.addWidget(QtWidgets.QLabel(name), r, basec + 0)
            if name in ENUM_MAP_BY_NAME:
                combo = QtWidgets.QComboBox()
                for k, txt in ENUM_MAP_BY_NAME[name].items():
                    combo.addItem(f"{k} - {txt}", k)
                self.enum_senders[name] = combo
                grid.addWidget(combo, r, basec + 1)
                recv = QtWidgets.QLabel("—"); self.enum_receivers[name] = recv
                grid.addWidget(recv, r, basec + 2)
            elif name == "sensored_control":
                cb = QtWidgets.QCheckBox(); self.send_vars[name] = cb
                grid.addWidget(cb, r, basec + 1)
                recv = QtWidgets.QCheckBox(); recv.setEnabled(False); self.recv_vars[name] = recv
                grid.addWidget(recv, r, basec + 2)
            else:
                e = QtWidgets.QLineEdit(); e.setFixedWidth(110); self.send_vars[name] = e
                grid.addWidget(e, r, basec + 1)
                rv = QtWidgets.QLabel("—"); self.recv_vars[name] = rv
                grid.addWidget(rv, r, basec + 2)
            grid.addWidget(QtWidgets.QLabel(unit), r, basec + 3)

        r = [0]*COLUMNS
        for idx, (name, typ, unit) in enumerate(FIELDS):
            c = min(idx // row_per_col, COLUMNS-1)
            add_row(r[c], c, name, typ, unit)
            r[c] += 1

        scroll.setWidget(inner)
        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(scroll)
        return w

    def _build_log_settings_tab(self):
        """Build the log settings tab with dynamic channel selection and log slots."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        
        # Initialize log-specific storage
        if not hasattr(self, 'log_send_vars'):
            self.log_send_vars = {}
        if not hasattr(self, 'log_recv_vars'):
            self.log_recv_vars = {}
        if not hasattr(self, 'log_slot_widgets'):
            self.log_slot_widgets = []  # List of lists: [send_widget, recv_widget] for each slot
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        
        # Add regular log fields (data_logger_ts_div, log_channels)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        
        row = 0
        for name, typ, unit in LOG_FIELDS:
            if name in ("log_scales[100]", "log_slot[LOG_N_CHANNELS]", "log_channels"):
                continue  # Handle arrays and log_channels separately
            
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            e = QtWidgets.QLineEdit()
            e.setFixedWidth(110)
            self.log_send_vars[name] = e
            grid.addWidget(e, row, 1)
            rv = QtWidgets.QLabel("—")
            self.log_recv_vars[name] = rv
            grid.addWidget(rv, row, 2)
            grid.addWidget(QtWidgets.QLabel(unit), row, 3)
            row += 1
        
        inner_layout.addLayout(grid)
        
        # Separator
        inner_layout.addWidget(QtWidgets.QLabel(""))
        inner_layout.addWidget(QtWidgets.QLabel("Channel Configuration:"))
        
        # Channel selection dropdown
        channel_layout = QtWidgets.QHBoxLayout()
        channel_layout.addWidget(QtWidgets.QLabel("Number of Channels:"))
        self.log_channel_combo = QtWidgets.QComboBox()
        self.log_channel_combo.setEnabled(True)  # Ensure it's enabled
        # Initially populate with default range (1 to LOG_N_CHANNELS) so it's usable
        # This will be updated when settings are received
        for i in range(1, LOG_N_CHANNELS + 1):
            self.log_channel_combo.addItem(str(i), i)
        # Connect to update log_channels when changed
        def on_channel_count_changed(index):
            self._on_log_channels_changed(index)
            # Update log_channels value
            num_ch = self.log_channel_combo.currentData()
            if num_ch is not None:
                self._recv_values["log_channels"] = num_ch
        
        self.log_channel_combo.currentIndexChanged.connect(on_channel_count_changed)
        channel_layout.addWidget(self.log_channel_combo)
        # Display MCU log_channels value next to dropdown
        channel_layout.addWidget(QtWidgets.QLabel("(MCU: —)"))
        self.log_channels_mcu_label = QtWidgets.QLabel("—")
        channel_layout.addWidget(self.log_channels_mcu_label)
        channel_layout.addStretch()
        inner_layout.addLayout(channel_layout)
        
        # Container for log slot inputs (will be populated dynamically)
        self.log_slots_container = QtWidgets.QWidget()
        self.log_slots_layout = QtWidgets.QGridLayout(self.log_slots_container)
        self.log_slots_layout.setHorizontalSpacing(10)
        self.log_slots_layout.setVerticalSpacing(4)
        inner_layout.addWidget(self.log_slots_container)
        
        inner_layout.addStretch()
        
        scroll.setWidget(inner)
        v.addWidget(scroll)
        
        return w
    
    def _on_log_channels_changed(self, index):
        """Update log slot inputs when channel count changes."""
        # Clear existing slot widgets and scale widgets
        for widgets in self.log_slot_widgets:
            for w in widgets:
                w.setParent(None)
        self.log_slot_widgets.clear()
        # Clear scale widgets dictionary
        if hasattr(self, 'log_scales_widgets'):
            self.log_scales_widgets.clear()
        
        # Get selected number of channels
        num_channels = self.log_channel_combo.currentData()
        if num_channels is None:
            return
        
        # Get available log variables from LOG_TABLE_VARIABLES
        available_vars = LOG_TABLE_VARIABLES
        
        # Initialize log_scales_widgets if not exists
        if not hasattr(self, 'log_scales_widgets'):
            self.log_scales_widgets = {}
        
        # Create slot dropdowns for each channel with associated log scale
        # Channel 1 uses index 0, Channel 2 uses index 1, etc.
        for i in range(num_channels):
            label = QtWidgets.QLabel(f"Channel {i+1} Slot:")
            send_combo = QtWidgets.QComboBox()
            send_combo.setFixedWidth(200)
            # Add all available variables with their index from LOG_TABLE (no "None" option)
            for idx, var_name in enumerate(available_vars):
                send_combo.addItem(f"{idx} - {var_name}", idx)
            
            # When slot selection changes, update the scale label immediately
            def make_slot_changed_handler(channel_idx, combo_widget):
                def slot_changed(index):
                    slot_idx = combo_widget.currentData()
                    if slot_idx is not None:
                        scale_label = self.log_scales_widgets.get(f"label_{channel_idx}")
                        if scale_label:
                            scale_label.setText(f"log_scale[{slot_idx}]:")
                        # Also update MCU scale if available
                        self._update_mcu_scale_display(channel_idx, slot_idx)
                return slot_changed
            
            recv_label = QtWidgets.QLabel("—")
            
            # Log scale input for this channel (program/send)
            scale_label = QtWidgets.QLabel("log_scale[0]:")
            scale_label.setFixedWidth(100)
            scale_edit = QtWidgets.QLineEdit()
            scale_edit.setFixedWidth(100)
            scale_edit.setText("0.0")
            
            # MCU scale display (receive)
            scale_recv_label = QtWidgets.QLabel("—")
            scale_recv_label.setFixedWidth(100)
            
            scale_key = f"scale_{i}"
            self.log_scales_widgets[scale_key] = scale_edit
            self.log_scales_widgets[f"label_{i}"] = scale_label
            self.log_scales_widgets[f"recv_{i}"] = scale_recv_label
            
            # Connect handler after widgets are stored
            send_combo.currentIndexChanged.connect(make_slot_changed_handler(i, send_combo))
            # Set initial scale label based on default selection (index 0 = TIME_16US)
            initial_slot_idx = send_combo.currentData() if send_combo.count() > 0 else 0
            if initial_slot_idx is not None:
                scale_label.setText(f"log_scale[{initial_slot_idx}]:")
                self._update_mcu_scale_display(i, initial_slot_idx)
            
            self.log_slots_layout.addWidget(label, i, 0)
            self.log_slots_layout.addWidget(send_combo, i, 1)
            self.log_slots_layout.addWidget(recv_label, i, 2)
            self.log_slots_layout.addWidget(QtWidgets.QLabel("u8"), i, 3)
            self.log_slots_layout.addWidget(scale_label, i, 4)
            self.log_slots_layout.addWidget(scale_edit, i, 5)
            self.log_slots_layout.addWidget(QtWidgets.QLabel("(MCU:"), i, 6)
            self.log_slots_layout.addWidget(scale_recv_label, i, 7)
            self.log_slots_layout.addWidget(QtWidgets.QLabel(")"), i, 8)
            
            # Store widgets for later access
            slot_name = f"log_slot[{i}]"  # Channel 1 uses index 0, Channel 2 uses index 1, etc.
            self.log_send_vars[slot_name] = send_combo
            self.log_recv_vars[slot_name] = recv_label
            self.log_slot_widgets.append([label, send_combo, recv_label, scale_label, scale_edit, scale_recv_label])

    def _build_control_payload(self) -> bytes:
        fmt = LE + "hhhhhhBB"
        return struct.pack(
            fmt,
            int(self._ctrl_vals.get("num_max_trq", 0)),
            int(self._ctrl_vals.get("num_min_trq", 0)),
            int(self._ctrl_vals.get("max_velocity", 0)),
            int(self._ctrl_vals.get("min_velocity", 0)),
            int(self._ctrl_vals.get("velocity_req", 0)),
            int(self._ctrl_vals.get("torque_req", 0)),
            int(self._ctrl_vals.get("mode", 1)),
            int(self._ctrl_vals.get("inv_en", 0)),
        )

    def _send_settings(self):
        if not self._ensure_open(): return
        payload = self._build_settings_payload_1024()
        pkt = build_packet(CMD_SEND_SETTINGS, payload, with_crc=True)
        self._write_hex_preview(pkt)
        try:
            self.ser.write(pkt)
            self._set_status("Settings sent ✅")
        except Exception as e:
            self._set_status(f"I/O error: {e}")
            self._close_serial()

    def _read_settings(self):
        if not self._ensure_open(): return
        try:
            self.ser.write(build_packet(CMD_RECEIVE_SETTINGS, b"", with_crc=True))
            self._set_status("Requested settings…")
        except Exception as e:
            self._set_status(f"I/O error: {e}")
            self._close_serial()

    def _send_simple(self, cmd, payload: bytes, ok_msg: str):
        if not self._ensure_open(): return
        try:
            self.ser.write(build_packet(cmd, payload, with_crc=True))
            self._set_status(ok_msg)
        except Exception as e:
            self._set_status(f"I/O error: {e}")
            self._close_serial()

    def _build_settings_payload_1024(self) -> bytes:
        p = bytearray()
        for (name, typ, _) in ALL_FIELDS:
            if is_array_field(name):
                base_name, array_size = parse_array_field(name)
                # Handle arrays
                if base_name == "log_scales":
                    # log_scales[100] - array of 100 f32
                    # Ensure array_size is exactly 100
                    if array_size != 100:
                        array_size = 100
                    # Initialize all to 0.0
                    scales_array = [0.0] * array_size
                    # Get values from per-channel scale inputs
                    scales_widgets = getattr(self, 'log_scales_widgets', {})
                    num_channels = getattr(self, 'log_channel_combo', None)
                    if num_channels and num_channels.currentData() is not None:
                        num_ch = num_channels.currentData()
                        for i in range(num_ch):
                            # Get the slot index for this channel
                            slot_name = f"log_slot[{i}]"
                            slot_widget = getattr(self, 'log_send_vars', {}).get(slot_name)
                            if slot_widget and isinstance(slot_widget, QtWidgets.QComboBox):
                                slot_idx = slot_widget.currentData()
                                if slot_idx is not None and 0 <= slot_idx < array_size:
                                    # Get the scale value for this channel
                                    scale_key = f"scale_{i}"
                                    scale_widget = scales_widgets.get(scale_key)
                                    if scale_widget and isinstance(scale_widget, QtWidgets.QLineEdit):
                                        try:
                                            scale_val = float(scale_widget.text().strip() or "0.0")
                                        except:
                                            scale_val = 0.0
                                    else:
                                        scale_val = 0.0
                                    scales_array[slot_idx] = scale_val
                    # Pack exactly 100 float values (400 bytes total)
                    assert len(scales_array) == 100, f"log_scales array must be exactly 100 elements, got {len(scales_array)}"
                    for val in scales_array:
                        p += struct.pack(LE + "f", val)
                elif base_name == "log_slot":
                    # log_slot[LOG_N_CHANNELS] - array of u8
                    # Get values from log slot widgets (now QComboBox)
                    num_channels = getattr(self, 'log_channel_combo', None)
                    if num_channels and num_channels.currentData() is not None:
                        num_ch = num_channels.currentData()
                        for i in range(num_ch):
                            slot_name = f"log_slot[{i}]"
                            w = getattr(self, 'log_send_vars', {}).get(slot_name)
                            if w and isinstance(w, QtWidgets.QComboBox):
                                val = int(w.currentData() or 0) & 0xFF
                            elif w and isinstance(w, QtWidgets.QLineEdit):
                                try:
                                    val = int(w.text().strip() or "0") & 0xFF
                                except:
                                    val = 0
                            else:
                                val = 0
                            p += struct.pack(LE + "B", val)
                        # Pad remaining slots with zeros
                        for _ in range(array_size - num_ch):
                            p += struct.pack(LE + "B", 0)
                    else:
                        # No channels selected, pack zeros
                        for _ in range(array_size):
                            p += struct.pack(LE + "B", 0)
            elif name in ENUM_MAP_BY_NAME:
                combo = self.enum_senders.get(name)
                val = int(combo.currentData()) if combo else 0
                p += struct.pack(LE + "B", val & 0xFF)
            elif name == "sensored_control":
                val = 1 if self.send_vars[name].isChecked() else 0
                p += struct.pack(LE + "B", val & 0xFF)
            else:
                # Special handling for log_channels (no widget, use received value or 0)
                if name == "log_channels":
                    # Use received value if available, otherwise 0
                    val = self._recv_values.get(name, 0)
                    p += struct.pack(LE + "B", int(val) & 0xFF)
                # Check if it's a log field
                elif name in [f[0] for f in LOG_FIELDS if not is_array_field(f[0])]:
                    w = getattr(self, 'log_send_vars', {}).get(name)
                    s = w.text().strip() if isinstance(w, QtWidgets.QLineEdit) else "0"
                    try:
                        if   typ == "u8":  p += struct.pack(LE + "B", int(s) & 0xFF)
                        elif typ == "u16": p += struct.pack(LE + "H", int(s) & 0xFFFF)
                        elif typ == "u32": p += struct.pack(LE + "I", int(s) & 0xFFFFFFFF)
                        elif typ == "i32": p += struct.pack(LE + "i", int(s))
                        elif typ == "f32": p += struct.pack(LE + "f", float(s))
                    except Exception:
                        if   typ == "u8":  p += struct.pack(LE + "B", 0)
                        elif typ == "u16": p += struct.pack(LE + "H", 0)
                        elif typ == "u32": p += struct.pack(LE + "I", 0)
                        elif typ == "i32": p += struct.pack(LE + "i", 0)
                        elif typ == "f32": p += struct.pack(LE + "f", 0.0)
                else:
                    w = self.send_vars.get(name)
                    s = w.text().strip() if isinstance(w, QtWidgets.QLineEdit) else "0"
                    try:
                        if   typ == "u8":  p += struct.pack(LE + "B", int(s) & 0xFF)
                        elif typ == "u16": p += struct.pack(LE + "H", int(s) & 0xFFFF)
                        elif typ == "u32": p += struct.pack(LE + "I", int(s) & 0xFFFFFFFF)
                        elif typ == "i32": p += struct.pack(LE + "i", int(s))
                        elif typ == "f32": p += struct.pack(LE + "f", float(s))
                    except Exception:
                        if   typ == "u8":  p += struct.pack(LE + "B", 0)
                        elif typ == "u16": p += struct.pack(LE + "H", 0)
                        elif typ == "u32": p += struct.pack(LE + "I", 0)
                        elif typ == "i32": p += struct.pack(LE + "i", 0)
                        elif typ == "f32": p += struct.pack(LE + "f", 0.0)
        if len(p) > FRAME_SIZE_SETTINGS:
            raise ValueError(f"Settings struct exceeds {FRAME_SIZE_SETTINGS} payload bytes")
        return bytes(p) + b"\x00"*(FRAME_SIZE_SETTINGS - len(p))

    def _write_hex_preview(self, pkt: bytes):
        self.hex_box.setPlainText(" ".join(f"{b:02X}" for b in pkt))

    def _load_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Settings JSON", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            settings = payload.get("settings", payload)
            # Load regular fields
            for (name, typ, unit) in FIELDS:
                if name not in settings: continue
                val = settings[name]
                if name in ENUM_MAP_BY_NAME:
                    combo = self.enum_senders.get(name)
                    if combo:
                        for i in range(combo.count()):
                            if combo.itemData(i) == int(val):
                                combo.setCurrentIndex(i); break
                elif name == "sensored_control":
                    self.send_vars[name].setChecked(bool(int(val)))
                else:
                    e = self.send_vars.get(name)
                    if isinstance(e, QtWidgets.QLineEdit):
                        e.setText(str(val))
            # Load log fields (non-array)
            for (name, typ, unit) in LOG_FIELDS:
                if is_array_field(name) or name == "log_channels":
                    continue  # Arrays and log_channels handled separately
                if name not in settings: continue
                val = settings[name]
                e = getattr(self, 'log_send_vars', {}).get(name)
                if isinstance(e, QtWidgets.QLineEdit):
                    e.setText(str(val))
            # Load log_channels (update MCU label and dropdown)
            if "log_channels" in settings:
                log_channels_val = int(settings["log_channels"])
                self._recv_values["log_channels"] = log_channels_val
                self._update_log_channels_ui(log_channels_val)
                mcu_label = getattr(self, 'log_channels_mcu_label', None)
                if mcu_label:
                    mcu_label.setText(str(log_channels_val))
            # Load log slots if present
            if "log_slot[LOG_N_CHANNELS]" in settings:
                slot_values = settings["log_slot[LOG_N_CHANNELS]"]
                if isinstance(slot_values, list):
                    for i, val in enumerate(slot_values):
                        slot_name = f"log_slot[{i}]"
                        w = getattr(self, 'log_send_vars', {}).get(slot_name)
                        slot_val = int(val)
                        if isinstance(w, QtWidgets.QComboBox):
                            # Find the item with matching data value
                            for j in range(w.count()):
                                if w.itemData(j) == slot_val:
                                    w.setCurrentIndex(j)
                                    break
                        elif isinstance(w, QtWidgets.QLineEdit):
                            w.setText(str(slot_val))
            # Load log scales if present
            if "log_scales[100]" in settings:
                scales_values = settings["log_scales[100]"]
                if isinstance(scales_values, list):
                    self._recv_values["log_scales[100]"] = scales_values
                    # Update per-channel scale inputs
                    self._update_log_scales_from_recv()
            self._set_status("Loaded JSON ✅")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load JSON", f"Failed to load:\n{e}")

    def _save_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Settings JSON", "", "JSON (*.json)")
        if not path: return
        try:
            settings_dict = {}
            # Save regular fields
            for (name, typ, unit) in FIELDS:
                if name in ENUM_MAP_BY_NAME:
                    combo = self.enum_senders.get(name)
                    val = int(combo.currentData()) if combo else 0
                elif name == "sensored_control":
                    val = 1 if self.send_vars[name].isChecked() else 0
                else:
                    e = self.send_vars.get(name)
                    val = e.text().strip() if isinstance(e, QtWidgets.QLineEdit) else "0"
                    if typ in ("u8","u16","u32","i32"):
                        try: val = int(val)
                        except: val = 0
                    elif typ == "f32":
                        try: val = float(val)
                        except: val = 0.0
                settings_dict[name] = val
            # Save log fields (non-array)
            for (name, typ, unit) in LOG_FIELDS:
                if is_array_field(name) or name == "log_channels":
                    continue  # Arrays and log_channels handled separately
                e = getattr(self, 'log_send_vars', {}).get(name)
                if isinstance(e, QtWidgets.QLineEdit):
                    val = e.text().strip() or "0"
                    if typ in ("u8","u16","u32","i32"):
                        try: val = int(val)
                        except: val = 0
                    elif typ == "f32":
                        try: val = float(val)
                        except: val = 0.0
                    settings_dict[name] = val
            # Save log_channels from received value
            if "log_channels" in self._recv_values:
                settings_dict["log_channels"] = int(self._recv_values["log_channels"])
            # Save log slots
            if hasattr(self, 'log_channel_combo') and self.log_channel_combo.currentData() is not None:
                num_ch = self.log_channel_combo.currentData()
                slot_values = []
                for i in range(num_ch):
                    slot_name = f"log_slot[{i}]"
                    w = getattr(self, 'log_send_vars', {}).get(slot_name)
                    if isinstance(w, QtWidgets.QComboBox):
                        val = int(w.currentData() or 0)
                    elif isinstance(w, QtWidgets.QLineEdit):
                        try:
                            val = int(w.text().strip() or "0")
                        except:
                            val = 0
                    else:
                        val = 0
                    slot_values.append(val)
                # Pad to LOG_N_CHANNELS
                while len(slot_values) < LOG_N_CHANNELS:
                    slot_values.append(0)
                settings_dict["log_slot[LOG_N_CHANNELS]"] = slot_values
            # Save log scales (build array from per-channel inputs)
            scales_array = [0.0] * 100
            scales_widgets = getattr(self, 'log_scales_widgets', {})
            if hasattr(self, 'log_channel_combo') and self.log_channel_combo.currentData() is not None:
                num_ch = self.log_channel_combo.currentData()
                for i in range(num_ch):
                    slot_name = f"log_slot[{i}]"
                    slot_widget = getattr(self, 'log_send_vars', {}).get(slot_name)
                    if slot_widget and isinstance(slot_widget, QtWidgets.QComboBox):
                        slot_idx = slot_widget.currentData()
                        if slot_idx is not None and 0 <= slot_idx < 100:
                            scale_key = f"scale_{i}"
                            scale_widget = scales_widgets.get(scale_key)
                            if scale_widget and isinstance(scale_widget, QtWidgets.QLineEdit):
                                try:
                                    val = float(scale_widget.text().strip() or "0.0")
                                except:
                                    val = 0.0
                            else:
                                val = 0.0
                            scales_array[slot_idx] = val
            settings_dict["log_scales[100]"] = scales_array

            meta = {"saved_at": datetime.now().isoformat(timespec="seconds"),
                    "app": "Inverter GUI — PySide6",
                    "schema": {"frame_size": FRAME_SIZE_SETTINGS, "endianness": LE}}
            blob = {"settings": settings_dict, "meta": meta}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)
            self._set_status("Saved JSON ✅")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save JSON", f"Failed to save:\n{e}")

    def _apply_settings_recv(self, raw: bytes):
        pos = 0
        for (name, typ, unit) in ALL_FIELDS:
            try:
                if is_array_field(name):
                    base_name, array_size = parse_array_field(name)
                    # Handle arrays
                    if base_name == "log_scales":
                        # log_scales[100] - array of 100 f32
                        # Store array values
                        arr_vals = []
                        for i in range(array_size):
                            try:
                                val = struct.unpack_from(LE+"f", raw, pos)[0]
                                pos += 4
                            except:
                                val = 0.0
                                pos += 4
                            arr_vals.append(val)
                        # Store as list in _recv_values
                        self._recv_values[name] = arr_vals
                        # Update per-channel scale inputs based on slot selections
                        self._update_log_scales_from_recv()
                        continue
                    elif base_name == "log_slot":
                        # log_slot[LOG_N_CHANNELS] - array of u8
                        # Read all slots, but only display up to log_channels
                        arr_vals = []
                        for _ in range(array_size):
                            try:
                                val = raw[pos]
                                pos += 1
                            except:
                                val = 0
                                pos += 1
                            arr_vals.append(val)
                        self._recv_values[name] = arr_vals
                        continue
                elif name == "log_channels":
                    # Special handling for log_channels: read u8
                    val = raw[pos]; pos += 1
                elif typ == "u8":  val = raw[pos]; pos += 1
                elif typ == "u16": val = struct.unpack_from(LE+"H", raw, pos)[0]; pos += 2
                elif typ == "u32": val = struct.unpack_from(LE+"I", raw, pos)[0]; pos += 4
                elif typ == "i32": val = struct.unpack_from(LE+"i", raw, pos)[0]; pos += 4
                elif typ == "f32": val = struct.unpack_from(LE+"f", raw, pos)[0]; pos += 4
                else: val = 0
            except Exception:
                val = 0

            self._recv_values[name] = val

            # Update UI widgets
            if name in ENUM_MAP_BY_NAME:
                recv = self.enum_receivers.get(name)
                if recv: recv.setText(f"{int(val)} - {enum_name(ENUM_MAP_BY_NAME[name], val)}")
            elif name == "sensored_control":
                rc = self.recv_vars.get(name)
                if rc: rc.setChecked(bool(val))
            elif name == "log_channels":
                # Special handling: update dropdown, slot widgets, and MCU label
                log_channels_val = int(val)
                self._update_log_channels_ui(log_channels_val)
                # Update MCU label next to dropdown
                mcu_label = getattr(self, 'log_channels_mcu_label', None)
                if mcu_label:
                    mcu_label.setText(str(log_channels_val))
            elif name in [f[0] for f in LOG_FIELDS if not is_array_field(f[0])]:
                # Other log fields
                rv = getattr(self, 'log_recv_vars', {}).get(name)
                if rv:
                    if isinstance(val, float):
                        rv.setText(f"{val:.6f}".rstrip("0").rstrip("."))
                    else:
                        rv.setText(str(val))
            else:
                rv = self.recv_vars.get(name)
                if rv:
                    if isinstance(val, float):
                        rv.setText(f"{val:.6f}".rstrip("0").rstrip("."))
                    else:
                        rv.setText(str(val))
        
        # Update log slot displays after all fields are processed
        self._update_log_slots_from_recv()
        # Update log scales after slots are set
        self._update_log_scales_from_recv()
    
    def _update_mcu_scale_display(self, channel_idx, slot_idx):
        """Update MCU scale display for a specific channel."""
        if "log_scales[100]" not in self._recv_values:
            return
        
        scales_values = self._recv_values["log_scales[100]"]
        if not isinstance(scales_values, list):
            return
        
        scales_widgets = getattr(self, 'log_scales_widgets', {})
        if slot_idx is not None and 0 <= slot_idx < len(scales_values):
            scale_recv_label = scales_widgets.get(f"recv_{channel_idx}")
            if scale_recv_label:
                scale_val = scales_values[slot_idx]
                scale_recv_label.setText(f"{scale_val:.6f}".rstrip("0").rstrip("."))
    
    def _update_log_scales_from_recv(self):
        """Update per-channel log scale inputs from received values."""
        if "log_scales[100]" not in self._recv_values:
            return
        
        scales_values = self._recv_values["log_scales[100]"]
        if not isinstance(scales_values, list):
            return
        
        # Get current number of channels
        if not hasattr(self, 'log_channel_combo') or self.log_channel_combo.currentData() is None:
            return
        
        num_ch = self.log_channel_combo.currentData()
        scales_widgets = getattr(self, 'log_scales_widgets', {})
        
        # Update scale inputs based on slot selections
        for i in range(num_ch):
            slot_name = f"log_slot[{i}]"  # Channel 1 uses index 0
            slot_widget = getattr(self, 'log_send_vars', {}).get(slot_name)
            if slot_widget and isinstance(slot_widget, QtWidgets.QComboBox):
                slot_idx = slot_widget.currentData()
                if slot_idx is not None and 0 <= slot_idx < len(scales_values):
                    # Update scale label
                    scale_label = scales_widgets.get(f"label_{i}")
                    if scale_label:
                        scale_label.setText(f"log_scale[{slot_idx}]:")
                    # Update scale value (program/send)
                    scale_key = f"scale_{i}"
                    scale_widget = scales_widgets.get(scale_key)
                    if scale_widget and isinstance(scale_widget, QtWidgets.QLineEdit):
                        scale_val = scales_values[slot_idx]
                        scale_widget.setText(f"{scale_val:.6f}".rstrip("0").rstrip("."))
                    # Update MCU scale display (receive)
                    self._update_mcu_scale_display(i, slot_idx)
    
    def _update_log_channels_ui(self, num_channels):
        """Update the log channels dropdown and slot widgets based on received value."""
        if not hasattr(self, 'log_channel_combo'):
            return
        
        # Always maintain options from 1 to LOG_N_CHANNELS
        # Only update the current selection if we have a valid number of channels
        if num_channels > 0:
            # Ensure dropdown has all options from 1 to LOG_N_CHANNELS
            current_count = self.log_channel_combo.count()
            if current_count < LOG_N_CHANNELS:
                # Add missing options
                for i in range(current_count + 1, LOG_N_CHANNELS + 1):
                    self.log_channel_combo.addItem(str(i), i)
            
            # Set current selection to received num_channels value
            for i in range(self.log_channel_combo.count()):
                if self.log_channel_combo.itemData(i) == num_channels:
                    self.log_channel_combo.setCurrentIndex(i)
                    break
            
            # Trigger slot widget update
            self._on_log_channels_changed(self.log_channel_combo.currentIndex())
        else:
            # If log_channels is 0 or invalid, keep default range but don't select anything
            # The dropdown will still be usable with the default range
            pass
    
    def _update_log_slots_from_recv(self):
        """Update log slot input fields from received values."""
        if "log_slot[LOG_N_CHANNELS]" not in self._recv_values:
            return
        
        slot_values = self._recv_values["log_slot[LOG_N_CHANNELS]"]
        if not isinstance(slot_values, list):
            return
        
        # Get current number of channels
        if not hasattr(self, 'log_channel_combo') or self.log_channel_combo.currentData() is None:
            return
        
        num_ch = self.log_channel_combo.currentData()
        scales_widgets = getattr(self, 'log_scales_widgets', {})
        
        # Update slot widgets
        for i in range(min(num_ch, len(slot_values))):
            slot_name = f"log_slot[{i}]"
            send_w = getattr(self, 'log_send_vars', {}).get(slot_name)
            recv_w = getattr(self, 'log_recv_vars', {}).get(slot_name)
            slot_val = int(slot_values[i])
            # Update QComboBox if it's a dropdown
            if send_w and isinstance(send_w, QtWidgets.QComboBox):
                # Find the item with matching data value
                for j in range(send_w.count()):
                    if send_w.itemData(j) == slot_val:
                        send_w.setCurrentIndex(j)
                        # Update scale label when slot is set
                        scale_label = scales_widgets.get(f"label_{i}")
                        if scale_label:
                            scale_label.setText(f"log_scale[{slot_val}]:")
                        # Update MCU scale display
                        self._update_mcu_scale_display(i, slot_val)
                        break
            elif send_w and isinstance(send_w, QtWidgets.QLineEdit):
                send_w.setText(str(slot_val))
            # Update receive label
            if recv_w:
                recv_w.setText(str(slot_val))

    def _copy_recv_to_send(self):
        if not self._recv_values:
            self._set_status("No MCU settings received yet"); return
        # Copy regular fields
        for (name, typ, _unit) in FIELDS:
            if name not in self._recv_values: continue
            val = self._recv_values[name]
            if name in ENUM_MAP_BY_NAME:
                combo = self.enum_senders.get(name)
                if combo is None: continue
                for i in range(combo.count()):
                    if int(combo.itemData(i)) == int(val):
                        combo.setCurrentIndex(i); break
            elif name == "sensored_control":
                cb = self.send_vars.get(name)
                if cb: cb.setChecked(bool(int(val)))
            else:
                e = self.send_vars.get(name)
                if isinstance(e, QtWidgets.QLineEdit):
                    e.setText(str(int(val)) if typ in ("u8","u16","u32","i32") else f"{float(val):.6f}".rstrip("0").rstrip("."))
        # Copy log fields (non-array)
        for (name, typ, _unit) in LOG_FIELDS:
            if is_array_field(name) or name == "log_channels":
                continue  # Arrays and log_channels handled separately
            if name not in self._recv_values: continue
            val = self._recv_values[name]
            e = getattr(self, 'log_send_vars', {}).get(name)
            if isinstance(e, QtWidgets.QLineEdit):
                e.setText(str(int(val)) if typ in ("u8","u16","u32","i32") else f"{float(val):.6f}".rstrip("0").rstrip("."))
        # Copy log slots
        if "log_slot[LOG_N_CHANNELS]" in self._recv_values:
            slot_values = self._recv_values["log_slot[LOG_N_CHANNELS]"]
            if isinstance(slot_values, list):
                for i, val in enumerate(slot_values):
                    slot_name = f"log_slot[{i}]"
                    w = getattr(self, 'log_send_vars', {}).get(slot_name)
                    slot_val = int(val)
                    if isinstance(w, QtWidgets.QComboBox):
                        # Find the item with matching data value
                        for j in range(w.count()):
                            if w.itemData(j) == slot_val:
                                w.setCurrentIndex(j)
                                break
                    elif isinstance(w, QtWidgets.QLineEdit):
                        w.setText(str(slot_val))
        # Copy log scales (update per-channel scale inputs)
        if "log_scales[100]" in self._recv_values:
            self._update_log_scales_from_recv()
        self._set_status("MCU → Current settings copied ✅")

    # ---------------- Periodic control ----------------
    def _on_periodic_toggled(self, checked: bool):
        if checked:
            # fixed 50 ms period
            fixed_period = 50
            self.periodic_timer.setInterval(fixed_period)

            if not self._ensure_open():
                self._user_connect()
                if not self._ensure_open():
                    self._set_status("Port not open; periodic send not started")
                    self.periodic_chk.setChecked(False)
                    return

            if not self.periodic_timer.isActive():
                self.periodic_timer.start()
            self._set_status(f"Periodic control send ON (fixed {fixed_period} ms)")
        else:
            if self.periodic_timer.isActive():
                self.periodic_timer.stop()
            self._set_status("Periodic control send OFF")

    def _on_period_changed(self, val: int):
        new_ms = max(1, int(val))
        if self.monitor_timer.isActive():
            self.monitor_timer.setInterval(new_ms)
            self._set_status(f"Monitor request period → {new_ms} ms")

    def _on_live_samples_changed(self, n: int):
        self._set_live_max_samples(int(n))

    def _set_live_max_samples(self, n: int):
        n = max(10, int(n))  # safety
        # rebuild time deque
        old_time = list(self.live_time)[-n:]
        self.live_time = deque(old_time, maxlen=n)
        # rebuild each signal deque
        for k in list(self.live_buf.keys()):
            old = list(self.live_buf[k])[-n:]
            self.live_buf[k] = deque(old, maxlen=n)

    # --- Menus / Theme handling -----------------------------------------------
    def _build_menus(self):
        # --- Build detached Options menu (no menubar entry) ---
        opt_menu = QtWidgets.QMenu("Options", self)
        self.options_menu = opt_menu

        # --- Theme submenu ---
        theme_menu = QtWidgets.QMenu("Theme", self)
        opt_menu.addMenu(theme_menu)

        self._theme_group = QtGui.QActionGroup(self)
        self._theme_group.setExclusive(True)

        def add_theme_action(text, key):
            act = QtGui.QAction(text, self, checkable=True)
            act.setData(key)
            theme_menu.addAction(act)
            self._theme_group.addAction(act)
            return act

        self.act_theme_inherited = add_theme_action("Inherited", "inherited")
        self.act_theme_light     = add_theme_action("Light", "light")
        self.act_theme_dark      = add_theme_action("Dark", "dark")
        self.act_theme_inherited.setChecked(True)

        self._theme_group.triggered.connect(self._on_theme_changed)

        # attach to the gear button
        if hasattr(self, "options_btn") and self.options_btn is not None:
            self.options_btn.setMenu(opt_menu)

    def _on_theme_changed(self, action: QtGui.QAction):
        mode = action.data()
        self._apply_theme(mode)

    def _apply_theme(self, mode: str):
        """mode: 'inherited' | 'light' | 'dark'"""
        self.theme_mode = mode
        app = QtWidgets.QApplication.instance()
        QtWidgets.QApplication.setStyle("fusion")  # keep Fusion for consistency

        if mode == "inherited":
            app.setPalette(app.style().standardPalette())
        elif mode == "light":
            app.setPalette(self._light_palette())   # <- use crisp light palette
        elif mode == "dark":
            app.setPalette(self._dark_palette())

        # re-style tiles & plots
        if hasattr(self, "_apply_highlight_base_style"):
            self._apply_highlight_base_style()
        if hasattr(self, "_apply_highlight_text_style"):
            self._apply_highlight_text_style()
        if hasattr(self, "_apply_plot_theme"):
            self._apply_plot_theme()

    def _apply_plot_theme(self):
        """Safely synchronize pyqtgraph plots with the current theme."""
        # Determine theme colors
        mode = getattr(self, "theme_mode", "inherited")

        if mode == "dark":
            bg_color = QtGui.QColor(30, 30, 30)
            fg_color = QtGui.QColor(230, 230, 230)
            grid_pen = pg.mkPen((90, 90, 90))
        elif mode == "light":
            bg_color = QtGui.QColor(255, 255, 255)
            fg_color = QtGui.QColor(20, 20, 20)
            grid_pen = pg.mkPen((200, 200, 200))
        else:  # inherited
            pal = self.palette()
            bg_color = pal.color(QtGui.QPalette.Base)
            fg_color = pal.color(QtGui.QPalette.Text)
            grid_pen = pg.mkPen(fg_color.lighter(180))

        # Apply global pyqtgraph config
        pg.setConfigOption("background", bg_color)
        pg.setConfigOption("foreground", fg_color)

        def restyle_plot(pw: pg.PlotWidget):
            if not isinstance(pw, pg.PlotWidget):
                return
            try:
                pw.setBackground(bg_color)
                pi = pw.getPlotItem()
                # Axes colors
                for axis in ("left", "bottom", "right", "top"):
                    ax = pi.getAxis(axis)
                    if ax:
                        ax.setPen(fg_color)
                        ax.setTextPen(fg_color)
                # Enable grid lines safely
                pi.showGrid(x=True, y=True, alpha=0.3)
                # Legend colors
                leg = getattr(pi, "legend", None)
                if leg:
                    leg.setBrush(QtGui.QBrush(bg_color))
                    try:
                        leg.setLabelTextColor(fg_color)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[plot_theme] restyle error: {e}")

        # Schedule style after GUI settles to avoid freeze
        QtCore.QTimer.singleShot(0, lambda: [
            restyle_plot(getattr(self, "live_plot1", None)),
            restyle_plot(getattr(self, "live_plot2", None))
        ])

    def _light_palette(self) -> QtGui.QPalette:
        p = QtGui.QPalette()
        white   = QtGui.QColor(255, 255, 255)
        off     = QtGui.QColor(246, 246, 246)   # panels/alternate rows
        button  = QtGui.QColor(240, 240, 240)
        text    = QtGui.QColor(20, 20, 20)
        link    = QtGui.QColor(33, 111, 219)
        highlight = QtGui.QColor(51, 153, 255)

        p.setColor(QtGui.QPalette.Window, white)
        p.setColor(QtGui.QPalette.Base, white)
        p.setColor(QtGui.QPalette.AlternateBase, off)
        p.setColor(QtGui.QPalette.Button, button)
        p.setColor(QtGui.QPalette.ToolTipBase, white)
        p.setColor(QtGui.QPalette.ToolTipText, text)
        p.setColor(QtGui.QPalette.Text, text)
        p.setColor(QtGui.QPalette.WindowText, text)
        p.setColor(QtGui.QPalette.ButtonText, text)
        p.setColor(QtGui.QPalette.Link, link)
        p.setColor(QtGui.QPalette.Highlight, highlight)
        p.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
        return p

    def _dark_palette(self) -> QtGui.QPalette:
        # Popular Qt Fusion dark palette
        p = QtGui.QPalette()
        gray1 = QtGui.QColor(53, 53, 53)
        gray2 = QtGui.QColor(35, 35, 35)
        gray3 = QtGui.QColor(60, 60, 60)
        text  = QtGui.QColor(220, 220, 220)
        blue  = QtGui.QColor(42, 130, 218)

        p.setColor(QtGui.QPalette.Window, gray1)
        p.setColor(QtGui.QPalette.WindowText, text)
        p.setColor(QtGui.QPalette.Base, gray2)
        p.setColor(QtGui.QPalette.AlternateBase, gray1)
        p.setColor(QtGui.QPalette.ToolTipBase, text)
        p.setColor(QtGui.QPalette.ToolTipText, text)
        p.setColor(QtGui.QPalette.Text, text)
        p.setColor(QtGui.QPalette.Button, gray1)
        p.setColor(QtGui.QPalette.ButtonText, text)
        p.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        p.setColor(QtGui.QPalette.Link, blue)
        p.setColor(QtGui.QPalette.Highlight, blue)
        p.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        return p

    def _highlight_base_css(self) -> str:
        pal = self.palette()
        win_col = pal.color(QtGui.QPalette.Window)
        is_dark = win_col.lightness() < 128
        base_bg = win_col
        base_border = win_col.lighter(140) if is_dark else win_col.darker(140)
        return f"QFrame{{background:{base_bg.name()};border:1px solid {base_border.name()};border-radius:6px;}}"

    def _apply_highlight_base_style(self):
        css = self._highlight_base_css()
        for f in getattr(self, "hl_frames", {}).values():
            f.setStyleSheet(css)

    def _apply_highlight_text_style(self):
        pal = self.palette()
        win_col = pal.color(QtGui.QPalette.Window)
        is_dark = win_col.lightness() < 128
        title_col = "#cfd3da" if is_dark else "#555555"
        unit_col  = "#9aa0a6" if is_dark else "#777777"
        value_col = pal.color(QtGui.QPalette.Text).name()
        for f in getattr(self, "hl_frames", {}).values():
            for lbl in f.findChildren(QtWidgets.QLabel):
                obj = lbl.objectName()
                if obj == "hl_title":
                    lbl.setStyleSheet(f"background: transparent; border: none; color:{title_col};")
                elif obj == "hl_unit":
                    lbl.setStyleSheet(f"background: transparent; border: none; color:{unit_col};")
                elif obj == "hl_val":
                    lbl.setStyleSheet(f"background: transparent; border: none; color:{value_col};")
                else:
                    lbl.setStyleSheet("background: transparent; border: none;")

    def changeEvent(self, e: QtCore.QEvent):
        super().changeEvent(e)
        if e.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.StyleChange):
            self._apply_highlight_base_style()
            self._apply_highlight_text_style()
            # refresh limiter indicators too
            for ind in getattr(self, "limiter_bits", {}).values():
                ind._apply_theme_colors()

    def _stage_control_from_widgets(self):
        """Copy current widget values into the staged dict (_ctrl_vals).
        Called on editingFinished for spinboxes, and on change for combo/checkbox.
        """
        self._ctrl_vals["num_max_trq"]  = int(self.ctrl_num_max_trq.value())
        self._ctrl_vals["num_min_trq"]  = int(self.ctrl_num_min_trq.value())
        self._ctrl_vals["max_velocity"] = int(self.ctrl_max_velocity.value())
        self._ctrl_vals["min_velocity"] = int(self.ctrl_min_velocity.value())
        self._ctrl_vals["velocity_req"] = int(self.ctrl_velocity_req.value())
        self._ctrl_vals["torque_req"]   = int(self.ctrl_torque_req.value())
        self._ctrl_vals["mode"]         = int(self.ctrl_mode.currentData())
        self._ctrl_vals["inv_en"]       = 1 if self.ctrl_inverter_en.isChecked() else 0

    def _stage_all_before_send(self):
        """Ensure we pick up any value the user just typed but didn't defocus from."""
        # For spinboxes, editingFinished may not have fired yet; force read once.
        self._ctrl_vals["num_max_trq"]  = int(self.ctrl_num_max_trq.value())
        self._ctrl_vals["num_min_trq"]  = int(self.ctrl_num_min_trq.value())
        self._ctrl_vals["max_velocity"] = int(self.ctrl_max_velocity.value())
        self._ctrl_vals["min_velocity"] = int(self.ctrl_min_velocity.value())
        self._ctrl_vals["velocity_req"] = int(self.ctrl_velocity_req.value())
        self._ctrl_vals["torque_req"]   = int(self.ctrl_torque_req.value())
        # Mode/enable are already staged on change, but we refresh anyway:
        self._ctrl_vals["mode"]   = int(self.ctrl_mode.currentData())
        self._ctrl_vals["inv_en"] = 1 if self.ctrl_inverter_en.isChecked() else 0

    def _on_monitor_req_toggled(self, checked: bool):
        # Ensure a sane interval (use same spinbox)
        try:
            period = int(self.period_ms.value())
        except Exception:
            period = 100
            self.period_ms.setValue(period)
        period = max(1, period)
        self.monitor_timer.setInterval(period)

        if checked:
            if not self._ensure_open():
                self._user_connect()
                if not self._ensure_open():
                    self._set_status("Port not open; monitor requests not started")
                    self.monreq_chk.setChecked(False)
                    return
            if not self.monitor_timer.isActive():
                self.monitor_timer.start()
            self._set_status(f"Monitor requests ON ({self.monitor_timer.interval()} ms)")
        else:
            if self.monitor_timer.isActive():
                self.monitor_timer.stop()
            self._set_status("Monitor requests OFF")

    def _send_request_data(self):
        """Send CMD_REQUEST_DATA (no payload) to ask MCU to reply with CMD_DIAG_STATUS."""
        if not self._ensure_open():
            # Auto-stop if link dropped
            if self.monreq_chk.isChecked():
                self.monreq_chk.setChecked(False)
            return
        try:
            self.ser.write(build_packet(CMD_REQUEST_DATA, b"", with_crc=True))
        except Exception as e:
            self._set_status(f"I/O error (REQUEST_DATA): {e}")
            # Stop the timer to avoid spamming errors
            if self.monreq_chk.isChecked():
                self.monreq_chk.setChecked(False)
            self._close_serial()


    # ---------------- Control & Monitor tab ----------------
    def _build_ctrlmon_tab(self):
        # --- Base container and layout ---
        w = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(w)   # single persistent layout
        w.setLayout(main_layout)

        # ===========================================================
        # LEFT PANEL — CONTROL / STATUS / MONITOR
        # ===========================================================
        left = QtWidgets.QGroupBox("Control (PC → MCU)")
        lv = QtWidgets.QVBoxLayout(left)

        # ---- Control spinboxes ----
        form = QtWidgets.QFormLayout()
        self.ctrl_num_max_trq  = QtWidgets.QSpinBox(); self.ctrl_num_max_trq.setRange(-32768, 32767)
        self.ctrl_num_min_trq  = QtWidgets.QSpinBox(); self.ctrl_num_min_trq.setRange(-32768, 32767)
        self.ctrl_max_velocity = QtWidgets.QSpinBox(); self.ctrl_max_velocity.setRange(-32768, 32767)
        self.ctrl_min_velocity = QtWidgets.QSpinBox(); self.ctrl_min_velocity.setRange(-32768, 32767)
        self.ctrl_velocity_req = QtWidgets.QSpinBox(); self.ctrl_velocity_req.setRange(-32768, 32767)
        self.ctrl_torque_req   = QtWidgets.QSpinBox(); self.ctrl_torque_req.setRange(-32768, 32767)

        for sb in (self.ctrl_num_max_trq, self.ctrl_num_min_trq,
                self.ctrl_max_velocity, self.ctrl_min_velocity,
                self.ctrl_velocity_req, self.ctrl_torque_req):
            sb.setKeyboardTracking(False)
            sb.editingFinished.connect(self._stage_control_from_widgets)

        self.ctrl_mode = QtWidgets.QComboBox()
        self.ctrl_mode.addItem("SPEED CONTROL", 1)
        self.ctrl_mode.addItem("TORQUE CONTROL", 2)
        self.ctrl_inverter_en = QtWidgets.QCheckBox("Enable inverter (GD)")
        self.ctrl_mode.currentIndexChanged.connect(self._stage_control_from_widgets)
        self.ctrl_inverter_en.stateChanged.connect(self._stage_control_from_widgets)
        QtCore.QTimer.singleShot(0, self._stage_control_from_widgets)

        form.addRow("num_max_trq", self.ctrl_num_max_trq)
        form.addRow("num_min_trq", self.ctrl_num_min_trq)
        form.addRow("max_velocity", self.ctrl_max_velocity)
        form.addRow("min_velocity", self.ctrl_min_velocity)
        form.addRow("velocity_request", self.ctrl_velocity_req)
        form.addRow("torque_request", self.ctrl_torque_req)
        form.addRow("control mode", self.ctrl_mode)
        form.addRow(self.ctrl_inverter_en)
        lv.addLayout(form)

        # ---- Periodic send ----
        pv = QtWidgets.QHBoxLayout()
        self.periodic_chk = QtWidgets.QCheckBox("Start periodic send(50ms)")
        self.periodic_chk.toggled.connect(self._on_periodic_toggled)
        self.period_ms = QtWidgets.QSpinBox(); self.period_ms.setRange(10, 10000)
        self.period_ms.setValue(100); self.period_ms.valueChanged.connect(self._on_period_changed)

        self.monreq_chk = QtWidgets.QCheckBox("Monitor data request")   # NEW
        self.monreq_chk.toggled.connect(self._on_monitor_req_toggled)   # NEW
        pv.addWidget(self.monreq_chk)                                   # NEW

        pv.addWidget(self.periodic_chk)
        pv.addSpacing(15)
        pv.addWidget(self.monreq_chk)  # monitor request box already created
        pv.addWidget(QtWidgets.QLabel("monitor period [ms]:"))
        pv.addWidget(self.period_ms)
        pv.addStretch(1)
        lv.addLayout(pv)

        btn_send_once = QtWidgets.QPushButton("Send Once Now")
        btn_send_once.clicked.connect(self._send_control_once)
        lv.addWidget(btn_send_once)

        self.log_btn = QtWidgets.QPushButton("LOG DATA")
        self.log_btn.clicked.connect(self._toggle_logging)
        lv.addWidget(self.log_btn)

        # ---- Status ----
        stat_grp = QtWidgets.QGroupBox("Status (decoded)")
        sh = QtWidgets.QHBoxLayout(stat_grp)
        self.status_text1 = QtWidgets.QPlainTextEdit(); self.status_text1.setReadOnly(True)
        self.status_text2 = QtWidgets.QPlainTextEdit(); self.status_text2.setReadOnly(True)
        sh.addWidget(self.status_text1); sh.addWidget(self.status_text2)
        lv.addWidget(stat_grp)

        # ---- Limiters ----
        lim_grp = QtWidgets.QGroupBox("Limiters")
        lim_layout = QtWidgets.QHBoxLayout(lim_grp)
        self.limiter_bits = {}
        for bit, name in LIMITER_FLAG_BITS:
            ind = BitIndicator(name)
            lim_layout.addWidget(ind)
            self.limiter_bits[name] = ind
        lim_layout.addStretch(1)
        lv.addWidget(lim_grp)

        # ---- Left numeric monitor ----
        left_monitor = QtWidgets.QGroupBox("Monitor (numeric) — L")
        lgrid = QtWidgets.QGridLayout(left_monitor)
        self.mon_num_vars = {}
        lv.addWidget(left_monitor)
        lv.addStretch(1)

        # ===========================================================
        # RIGHT PANEL — HIGHLIGHTS / PLOTS / MONITOR R
        # ===========================================================
        right_col = QtWidgets.QVBoxLayout()

        # ---- Highlights ----
        hi = QtWidgets.QGroupBox("Highlights")
        hiv = QtWidgets.QHBoxLayout(hi)

        def make_tile(title, unit):
            frame = QtWidgets.QFrame()
            v = QtWidgets.QVBoxLayout(frame)
            v.setAlignment(QtCore.Qt.AlignCenter)
            lblt = QtWidgets.QLabel(title); f = lblt.font(); f.setBold(True); lblt.setFont(f)
            val = QtWidgets.QLabel("—"); g = val.font(); g.setPointSize(18); g.setBold(True); val.setFont(g)
            lblu = QtWidgets.QLabel(unit)
            for w_ in (lblt, val, lblu):
                w_.setAlignment(QtCore.Qt.AlignCenter)
                w_.setStyleSheet("background: transparent; border: none;")
            v.addWidget(lblt); v.addWidget(val); v.addWidget(lblu)
            return frame, val

        t_rpm, v_rpm = make_tile("RPM", "rpm")
        t_trqref, v_trqref = make_tile("Torque Ref", "Nm")
        t_trqact, v_trqact = make_tile("Torque Actual", "Nm")
        t_vdc, v_vdc = make_tile("DC Bus", "V")
        t_mt, v_mt = make_tile("Motor Temp", "°C")
        t_it, v_it = make_tile("IGBT Temp", "°C")

        for tile in (t_rpm, t_trqref, t_trqact, t_vdc, t_mt, t_it):
            hiv.addWidget(tile)
        right_col.addWidget(hi)

        self.hl_frames = {"motor_rpm": t_rpm, "trq_ref": t_trqref, "trq_actual": t_trqact,
                        "Vdc": t_vdc, "motor_temp": t_mt, "igbt_temp": t_it}
        self.hl_vals = {"motor_rpm": v_rpm, "trq_ref": v_trqref, "trq_actual": v_trqact,
                        "Vdc": v_vdc, "motor_temp": v_mt, "igbt_temp": v_it}

        self._apply_highlight_base_style()
        self._apply_highlight_text_style()

        # ---------------- Real-time plots
        plots_grp = QtWidgets.QGroupBox("Real-time plots (from DIAG_STATUS)")
        pv2 = QtWidgets.QVBoxLayout(plots_grp)

        # Top row: live window + Freeze/Resume
        winh = QtWidgets.QHBoxLayout()
        winh.addWidget(QtWidgets.QLabel("Live window (samples):"))
        self.live_samples_spin = QtWidgets.QSpinBox()
        self.live_samples_spin.setRange(50, 200000)
        self.live_samples_spin.setValue(self.live_max_samples)
        self.live_samples_spin.valueChanged.connect(self._on_live_samples_changed)
        winh.addWidget(self.live_samples_spin)

        self.plot_toggle_btn = QtWidgets.QPushButton("Freeze")
        self.plot_toggle_btn.setCheckable(True)
        self.plot_toggle_btn.toggled.connect(self._toggle_plots)
        winh.addWidget(self.plot_toggle_btn)
        winh.addStretch(1)
        pv2.addLayout(winh)

        # Multi-select dropdowns for which DIAG vars to plot
        all_names = [n for (n, _t) in DIAG_FIELDS]
        self.combo_plot1 = MultiSelectCombo(all_names)
        self.combo_plot2 = MultiSelectCombo(all_names)
        # sensible defaults
        for want in ["Id_ref","Iq_ref","Id_filt","Iq_filt"]:
            if want in all_names:
                self.combo_plot1.model().item(all_names.index(want)).setCheckState(QtCore.Qt.Checked)
        for want in ["motor_rpm"]:
            if want in all_names:
                self.combo_plot2.model().item(all_names.index(want)).setCheckState(QtCore.Qt.Checked)

        selh = QtWidgets.QHBoxLayout()
        selh.addWidget(QtWidgets.QLabel("Plot1 vars:")); selh.addWidget(self.combo_plot1)
        selh.addWidget(QtWidgets.QLabel("Plot2 vars:")); selh.addWidget(self.combo_plot2)
        pv2.addLayout(selh)

        # The two live plots + legends
        self.live_plot1 = pg.PlotWidget()
        self.live_plot1.showGrid(x=True, y=True)
        self.live_plot1.setDownsampling(auto=True)
        self.live_plot1.setClipToView(True)
        self.live_plot1.setAntialiasing(True)
        self.live_plot1.addLegend()
        self.vb1 = self.live_plot1.getViewBox()
        self.vb1.enableAutoRange(x=True, y=True)

        self.live_plot2 = pg.PlotWidget()
        self.live_plot2.showGrid(x=True, y=True)
        self.live_plot2.setDownsampling(auto=True)
        self.live_plot2.setClipToView(True)
        self.live_plot2.setAntialiasing(True)
        self.live_plot2.addLegend()
        self.vb2 = self.live_plot2.getViewBox()
        self.vb2.enableAutoRange(x=True, y=True)

        pv2.addWidget(self.live_plot1)
        pv2.addWidget(self.live_plot2)

        # theme & curve dicts
        self._apply_plot_theme()
        self.live_curves1 = {}
        self.live_curves2 = {}

        right_col.addWidget(plots_grp)

        # Zoom buttons row (keep as you had)
        btns = QtWidgets.QHBoxLayout()
        self.zoomx_btn = QtWidgets.QPushButton("Zoom X")
        self.zoomy_btn = QtWidgets.QPushButton("Zoom Y")
        self.fit_btn   = QtWidgets.QPushButton("Fit View")
        for b in (self.zoomx_btn, self.zoomy_btn, self.fit_btn):
            b.setCheckable(True)
            btns.addWidget(b)
        btns.addStretch(1)
        pv2.addLayout(btns)
        self._connect_zoom_buttons()


        # ---- Right numeric monitor ----
        right_monitor = QtWidgets.QGroupBox("Monitor (numeric) — R")
        rgrid = QtWidgets.QGridLayout(right_monitor)
        right_col.addStretch(1)
        right_col.addWidget(right_monitor)

        # ===========================================================
        # FINAL ASSEMBLY
        # ===========================================================
        main_layout.addWidget(left, 0)
        main_layout.addLayout(right_col, 1)

        # ---- Monitors grid population (unchanged) ----
        exclude = {"motor_rpm", "trq_ref", "trq_actual", "Vdc", "motor_temp", "igbt_temp", "limiter_flags"} | STATUS_BOX_FIELDS
        names = [n for (n, _t) in DIAG_FIELDS if n not in exclude]
        half = (len(names) + 1) // 2
        left_names, right_names = names[:half], names[half:]

        def add_monitor_grid(names_list, grid, cols_per_side=4):
            rows = (len(names_list) + cols_per_side - 1) // cols_per_side
            for i, name in enumerate(names_list):
                c = i // rows; r = i % rows
                lbl = QtWidgets.QLabel(name)
                edit = QtWidgets.QLineEdit(); edit.setReadOnly(True); edit.setFixedWidth(120)
                edit.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                f = edit.font(); f.setFamily("Consolas"); edit.setFont(f)
                unit = QtWidgets.QLabel(UNITS.get(name, "")); unit.setStyleSheet("color:#9aa0a6;")
                base = c * 3
                grid.addWidget(lbl, r, base + 0)
                grid.addWidget(edit, r, base + 1)
                grid.addWidget(unit, r, base + 2)
                self.mon_num_vars[name] = edit

        add_monitor_grid(left_names, lgrid, 3)
        add_monitor_grid(right_names, rgrid, 4)

        self.tabs.addTab(w, "Control & Monitor")


    def _toggle_plots(self, checked):
        self.plots_running = not checked
        self.plot_toggle_btn.setText("Resume" if checked else "Freeze")
        # when resuming, immediately re-enable auto-follow
        if self.plots_running:
            self.vb1.enableAutoRange(x=True, y=True)
            self.vb2.enableAutoRange(x=True, y=True)

    def _send_control_once(self):
        if not self._ensure_open():
            self._user_connect()
            if not self._ensure_open():
                self._set_status("Port not open"); return
        try:
            self._stage_all_before_send()
            payload = self._build_control_payload()
            pkt = build_packet(CMD_SEND_CONTROL, payload, with_crc=True)
            self.ser.write(pkt)
            self._set_status("Control sent")
        except Exception as e:
            self._set_status(f"I/O error: {e}")
            self._close_serial()

    def _connect_zoom_buttons(self):
        self.zoomx_btn.clicked.connect(lambda: self._set_zoom_mode("x"))
        self.zoomy_btn.clicked.connect(lambda: self._set_zoom_mode("y"))
        self.fit_btn.clicked.connect(self._zoom_fit)

    def _set_zoom_mode(self, mode):
        self.zoomx_btn.setChecked(mode == "x")
        self.zoomy_btn.setChecked(mode == "y")

        vb1 = self.live_plot1.getViewBox()
        vb2 = self.live_plot2.getViewBox()

        if mode == "x":
            vb1.setMouseMode(pg.ViewBox.RectMode)
            vb2.setMouseMode(pg.ViewBox.RectMode)
            vb1.setAspectLocked(False)
            vb2.setAspectLocked(False)
            vb1.setLimits(yMin=None, yMax=None)
            vb2.setLimits(yMin=None, yMax=None)
            vb1.disableAutoRange(axis=pg.ViewBox.YAxis)
            vb2.disableAutoRange(axis=pg.ViewBox.YAxis)
        elif mode == "y":
            vb1.setMouseMode(pg.ViewBox.RectMode)
            vb2.setMouseMode(pg.ViewBox.RectMode)
            vb1.setAspectLocked(False)
            vb2.setAspectLocked(False)
            vb1.disableAutoRange(axis=pg.ViewBox.XAxis)
            vb2.disableAutoRange(axis=pg.ViewBox.XAxis)
        else:
            vb1.enableAutoRange(axis=pg.ViewBox.XYAxes)
            vb2.enableAutoRange(axis=pg.ViewBox.XYAxes)

    def _zoom_fit(self):
        self.zoomx_btn.setChecked(False)
        self.zoomy_btn.setChecked(False)
        for vb in (self.live_plot1.getViewBox(), self.live_plot2.getViewBox()):
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes)
            vb.autoRange()

    # ---------------- Graph tab ----------------
    def _build_graph_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        ctrl = QtWidgets.QHBoxLayout()
        self.grp1_edit = QtWidgets.QLineEdit("(CH2,CH3);(CH4,CH5);(CH6,CH7)")
        self.grp2_edit = QtWidgets.QLineEdit("(CH8,CH9);(CH10,CH11,CH12)")
        self.titles1_edit = QtWidgets.QLineEdit("Flux;Torque;Sector/Vector")
        self.titles2_edit = QtWidgets.QLineEdit("Ia/Ib;CH10-12")
        self.linkx_chk = QtWidgets.QCheckBox("Link X"); self.linkx_chk.setChecked(True)
        ctrl.addWidget(QtWidgets.QLabel("Win1 groups:")); ctrl.addWidget(self.grp1_edit)
        ctrl.addWidget(QtWidgets.QLabel("Win2 groups:")); ctrl.addWidget(self.grp2_edit)
        ctrl.addWidget(QtWidgets.QLabel("Titles1:")); ctrl.addWidget(self.titles1_edit)
        ctrl.addWidget(QtWidgets.QLabel("Titles2:")); ctrl.addWidget(self.titles2_edit)
        ctrl.addWidget(self.linkx_chk)
        btn_load_csv = QtWidgets.QPushButton("Load CSV…"); btn_load_csv.clicked.connect(self._pick_csv_into_graph)
        ctrl.addWidget(btn_load_csv); ctrl.addStretch(1)
        v.addLayout(ctrl)

        # --- Zoom buttons for GRAPH (CSV) plots ---
        graph_btns = QtWidgets.QHBoxLayout()
        self.graph_zoomx_btn = QtWidgets.QPushButton("Zoom X")
        self.graph_zoomy_btn = QtWidgets.QPushButton("Zoom Y")
        self.graph_fit_btn   = QtWidgets.QPushButton("Fit View")
        for b in (self.graph_zoomx_btn, self.graph_zoomy_btn, self.graph_fit_btn):
            b.setCheckable(True)
            graph_btns.addWidget(b)
        graph_btns.addStretch(1)
        v.addLayout(graph_btns)

        self.two_plot = TwoWindowPlot()
        self.graph_zoomx_btn.clicked.connect(
            lambda: (self.graph_zoomy_btn.setChecked(False),
                    self.two_plot.set_zoom_mode("x"))
        )
        self.graph_zoomy_btn.clicked.connect(
            lambda: (self.graph_zoomx_btn.setChecked(False),
                    self.two_plot.set_zoom_mode("y"))
        )
        self.graph_fit_btn.clicked.connect(
            lambda: (self.graph_zoomx_btn.setChecked(False),
                    self.graph_zoomy_btn.setChecked(False),
                    self.two_plot.zoom_fit(),
                    self.two_plot.set_zoom_mode("none"))
        )
        v.addWidget(self.two_plot)
        self.tabs.addTab(w, "Graph")

    def _parse_group_str(self, s: str):
        out = []
        for part in s.split(";"):
            part = part.strip()
            if not part: continue
            inner = part[1:-1] if (part.startswith("(") and part.endswith(")")) else part
            chans = [x.strip() for x in inner.split(",") if x.strip()]
            out.append(tuple(chans))
        return out

    def _build_csv_async(self):
        """Run CSV build in a background thread and report back to UI safely."""
        try:
            # Do the heavy work off the GUI thread
            if self.dlogger:
                self.dlogger.build_csv()
            QtCore.QTimer.singleShot(0, lambda: self._set_status("CSV built ✅"))
        except Exception as e:
            QtCore.QTimer.singleShot(
                0,
                lambda: QtWidgets.QMessageBox.critical(self, "Logger", f"Failed to build CSV:\n{e}")
            )

        QtCore.QTimer.singleShot(0, self._refresh_csv_list)


    def _open_csv_in_graph_tab(self):
        name = self.view_combo.currentText().strip()
        if not name:
            self._refresh_csv_list()
            name = self.view_combo.currentText().strip()
            if not name:
                QtWidgets.QMessageBox.information(self, "View", "No CSVs found in ./data")
                return
        self._plot_csv_path(os.path.join(DATA_DIR, name))

    def _pick_csv_into_graph(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick CSV", DATA_DIR, "CSV (*.csv)")
        if not path: return
        self._plot_csv_path(path)

    def _plot_csv_path(self, path: str):
        g1 = self._parse_group_str(self.grp1_edit.text())
        g2 = self._parse_group_str(self.grp2_edit.text())
        t1 = [t.strip() for t in self.titles1_edit.text().split(";") if t.strip()]
        t2 = [t.strip() for t in self.titles2_edit.text().split(";") if t.strip()]
        linkx = self.linkx_chk.isChecked()
        try:
            self.two_plot.plot_csv(path, g1, g2, t1, t2, linkx)
            self._set_status(f"Opened: {os.path.basename(path)} ✅")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV", f"Failed to plot:\n{e}")

    def _refresh_csv_list(self):
        try:
            files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
            files.sort(reverse=True)
            self.view_combo.clear(); self.view_combo.addItems(files)
        except Exception:
            self.view_combo.clear()

    # ---------------- Logging (shared port) ----------------
    def _toggle_logging(self):
        if not self.logging:#start logging
            if not self._ensure_open(): return
            try:
                self.ser.write(build_packet(CMD_SEND_DATA, b"", with_crc=True))
            except Exception as e:
                self._set_status(f"I/O error: {e}"); return

            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_basename = f"run_{stamp}"

            self._postlog_lines = []
            self._postlog_rows = []              # << reset so we never reuse previous batch
            self._staged_pc_row = None
            self._awaiting_postlog = False
            self._stop_timeout.stop()
            # start DataLoggerQt (file handling only)
            self.dlogger = DataLoggerQt(self, n_channels=LOG_N_CHANNELS, data_dir=DATA_DIR)
            self.dlogger.start(self.log_basename)

            self.logging = True
            self.log_btn.setText("STOP LOG")
            self._set_status("Logging… (shared port)")
        else:
            try:
                if self.ser:
                    self.ser.write(build_packet(CMD_STOP_DATA, b"", with_crc=True))
            except Exception as e:
                self._set_status(f"I/O error (STOP): {e}")

            # Stage the one PC snapshot row now; we'll append MCU rows when they arrive
            if self.dlogger:
                ctrl_row = dict(self._ctrl_vals)
                mon_row  = dict(self._last_diag_values)
                self._staged_pc_row = {**ctrl_row, **mon_row}

            # Wait for MCU post-log batch (CMD_TEMP_DATA); fallback after 1.5 s
            self._awaiting_postlog = True
            self._stop_timeout.start(1500)
            self._set_status("Stopping… waiting for MCU post-log batch")

            self.logging = False
            self.log_btn.setText("LOG DATA")


    def _handle_log_payload(self, payload: bytes):
        if self.logging and self.dlogger:
            try:
                self.dlogger.write_payload(payload)
            except Exception as e:
                self._set_status(f"BIN write error: {e}")

    # ---------------- Frame handling ----------------
    def _handle_frame(self, cmd: int, payload: bytes):
        if cmd == CMD_SEND_SETTINGS:
            raw = payload[:FRAME_SIZE_SETTINGS]
            self._apply_settings_recv(raw)
            self._set_status("Settings received ✅")
        elif cmd == CMD_DIAG_STATUS:
            self._handle_diag_status(payload)
        elif cmd == CMD_TEMP_DATA:
            # binary: repeated [TEMP_CTRL_FIELDS][DIAG_FIELDS] records
            rows = []
            off = 0
            total = len(payload)
            while off + TEMP_ROW_SIZE <= total:
                # control part
                ctrl_vals = struct.unpack_from(_TEMP_CTRL_FMT, payload, off); off += struct.calcsize(_TEMP_CTRL_FMT)
                ctrl = {name: ctrl_vals[i] for i, (name, _t) in enumerate(TEMP_CTRL_FIELDS)}
                # diag part
                diag_vals = struct.unpack_from(_DIAG_FMT, payload, off); off += struct.calcsize(_DIAG_FMT)
                diag = {name: diag_vals[i] for i, (name, _t) in enumerate(DIAG_FIELDS)}
                rows.append({**ctrl, **diag})
            self._postlog_rows = rows
            self._set_status(f"TEMP_DATA parsed: {len(rows)} rows")
            if self._awaiting_postlog:
                self._finalize_logging()
            print(f"DEBUG: TEMP_DATA rows received: {len(rows)}")

    def _finalize_logging(self):
        # Guard against double-calls
        if not self.dlogger:
            self._awaiting_postlog = False
            self._stop_timeout.stop()
            return

        self._stop_timeout.stop()
        self._awaiting_postlog = False

        # Append staged PC row (if any) and fresh MCU rows
        if self._staged_pc_row:
            try:
                self.dlogger.add_monitor_row(self._staged_pc_row)
            except Exception:
                pass
        if self._postlog_rows:
            try:
                self.dlogger.add_monitor_rows(self._postlog_rows)
            except Exception:
                pass

        # Build CSV off the GUI thread
        import threading
        def _close_and_build():
            try:
                self.dlogger.close_bin()
            except Exception:
                pass
            self._build_csv_async()   # calls dlogger.build_csv(), updates UI

        threading.Thread(target=_close_and_build, daemon=True).start()

        # Reset per-session containers
        self._postlog_rows = []
        self._staged_pc_row = None


    def _handle_diag_status(self, payload: bytes):
        values = {}
        pos = 0
        for name, typ in DIAG_FIELDS:
            try:
                if   typ == "u8":  v = payload[pos]; pos += 1
                elif typ == "u16": v = struct.unpack_from(LE+"H", payload, pos)[0]; pos += 2
                elif typ == "i16": v = struct.unpack_from(LE+"h", payload, pos)[0]; pos += 2
                elif typ == "i32": v = struct.unpack_from(LE+"i", payload, pos)[0]; pos += 4
                elif typ == "f32": v = struct.unpack_from(LE+"f", payload, pos)[0]; pos += 4
                else: v = 0
            except Exception:
                v = 0
            values[name] = v

        # Highlights values (plain text)
        def set_lbl(name, fmt=None):
            if name in self.hl_vals and name in values:
                v = values[name]
                self.hl_vals[name].setText(str(v) if fmt is None else fmt.format(v))
        set_lbl("motor_rpm")
        set_lbl("trq_ref", "{:.1f}")
        set_lbl("trq_actual", "{:.1f}")
        set_lbl("Vdc", "{:.1f}")
        set_lbl("motor_temp", "{:.1f}")
        set_lbl("igbt_temp", "{:.1f}")

        # (6) Highlight threshold logic → orange
        self._update_highlight_colors(values)

        # Numeric monitor cells (3-decimal formatting for floats)
        for k, w in self.mon_num_vars.items():
            if k not in values:
                continue

            val = values[k]
            try:
                if isinstance(val, (int, np.integer)):
                    txt = str(val)
                elif isinstance(val, float):
                    txt = f"{val:.3f}".rstrip("0").rstrip(".")
                else:
                    fv = float(val)
                    txt = f"{fv:.3f}".rstrip("0").rstrip(".")
            except Exception:
                txt = str(val)

            w.setText(txt)
            w.setCursorPosition(0)

        # Decoded text split into two columns
        lines = []
        if "critical_hw_status" in values: lines.append(f"critical_hw_status: {fmt_crit(int(values['critical_hw_status']))}")
        if "aux_hw_status" in values:      lines.append(f"aux_hw_status: {fmt_bits(int(values['aux_hw_status']), [(0,'HW_TSAL_OVER60'),(1,'HW_DISCHARGE_ENABLED'),(2,'HW_USB_CONNECTED'),(3,'HW_ST_LINK_CONNECTED')])}")
        if "last_error" in values:         lines.append(f"last_error: {enum_name(ERRORS_MAP, values['last_error'])}")
        if "sys_status" in values:         lines.append(f"sys_status: {enum_name(ERRORS_MAP, values['sys_status'])}")
        if "handler_status" in values:     lines.append(f"handler_status: {enum_name(ERRORTYPE_MAP, values['handler_status'])}")
        if "latched_error" in values:      lines.append(f"latched_error: {enum_name(LATCHED_MAP, values['latched_error'])}")
        if "actual_state" in values:       lines.append(f"actual_state: {enum_name(STATE_MAP, values['actual_state'])}")
        if "control_mode_actual" in values:lines.append(f"control_mode_actual: {enum_name(CTRL_MODE_MAP, values['control_mode_actual'])}")

        half = (len(lines)+1)//2
        self.status_text1.setPlainText("\n".join(lines[:half]) if lines else "")
        self.status_text2.setPlainText("\n".join(lines[half:]) if lines else "")

        # --- Update limiter flag indicators ---
        lf_val = int(values.get("limiter_flags", 0))
        for bit, name in LIMITER_FLAG_BITS:
            active = bool(lf_val & (1 << bit))
            if name in self.limiter_bits:
                self.limiter_bits[name].setActive(active)

        # Live plot buffers & draw
        self._update_live_plots(values)

        self._last_diag_values = dict(values)

    def _update_highlight_colors(self, values: dict):
        # --- Build styles based on current palette/theme ---
        pal = self.palette()
        win_col = pal.color(QtGui.QPalette.Window)
        is_dark = win_col.lightness() < 128

        # base tile = match UI theme
        base_bg = win_col
        base_border = (win_col.lighter(140) if is_dark else win_col.darker(140))
        def css_frame(bg: QtGui.QColor, border: QtGui.QColor) -> str:
            return f"QFrame{{background:{bg.name()};border:1px solid {border.name()};border-radius:6px;}}"

        base = css_frame(base_bg, base_border)

        # warn tile = orange regardless of theme (tuned per theme for contrast)
        warn_bg   = QtGui.QColor("#a86b00") if is_dark else QtGui.QColor("#ffe8cc")
        warn_bord = QtGui.QColor("#ffc766") if is_dark else QtGui.QColor("#a86b00")
        warn = css_frame(warn_bg, warn_bord)

        # ---- Threshold rules ----

        # rpm > max_velocity
        try:
            rpm = float(values.get("motor_rpm", 0))
            vmax = float(values.get("max_velocity", 0))
            self.hl_frames["motor_rpm"].setStyleSheet(warn if (vmax != 0 and rpm > vmax) else base)
        except Exception:
            self.hl_frames["motor_rpm"].setStyleSheet(base)

        # DC bus > 60
        try:
            vdc = float(values.get("Vdc", 0))
            self.hl_frames["Vdc"].setStyleSheet(warn if vdc > 60 else base)
        except Exception:
            self.hl_frames["Vdc"].setStyleSheet(base)

        # Temps vs derate thresholds (from received settings if available)
        mt = float(values.get("motor_temp", 0))
        it = float(values.get("igbt_temp", 0))
        mt_thr = float(self._recv_values.get("motor_temp_derate_thres", 0))
        it_thr = float(self._recv_values.get("igbt_temp_derate_thres", 0))
        self.hl_frames["motor_temp"].setStyleSheet(warn if (mt_thr and mt >= mt_thr) else base)
        self.hl_frames["igbt_temp"].setStyleSheet(warn if (it_thr and it >= it_thr) else base)

        # torque tiles → base (no thresholds)
        self.hl_frames["trq_ref"].setStyleSheet(base)
        self.hl_frames["trq_actual"].setStyleSheet(base)


    def _color_for(self, name: str):
        if name not in self._color_map:
            self._color_map[name] = pg.intColor(len(self._color_map))
        return self._color_map[name]

    def _update_live_plots(self, values: dict):
        # If plots not built yet, bail out gracefully
        if not hasattr(self, "combo_plot1") or self.combo_plot1 is None:
            return
        if not hasattr(self, "live_plot1") or self.live_plot1 is None:
            return
        if not self.plots_running:
            return

        now = time.monotonic() - getattr(self, "live_t0", time.monotonic())
        self.live_time.append(now)

        names1 = self.combo_plot1.checked_items()
        names2 = self.combo_plot2.checked_items()

        def ensure_curves(names, plotw, curves_dict):
            for k in list(curves_dict):
                if k not in names:
                    plotw.removeItem(curves_dict[k])
                    del curves_dict[k]
            for name in names:
                if name not in curves_dict:
                    pen = self._color_for(name)
                    curves_dict[name] = plotw.plot(name=name, pen=pen)

        ensure_curves(names1, self.live_plot1, self.live_curves1)
        ensure_curves(names2, self.live_plot2, self.live_curves2)

        for n in names1 + names2:
            if n in values:
                self.live_buf[n].append(values[n])

        x = np.fromiter(self.live_time, dtype=float)
        for n, curve in self.live_curves1.items():
            y = np.fromiter(self.live_buf[n], dtype=float) if n in self.live_buf else np.array([])
            if len(y) > 2:
                curve.setData(x[-len(y):], y)
        for n, curve in self.live_curves2.items():
            y = np.fromiter(self.live_buf[n], dtype=float) if n in self.live_buf else np.array([])
            if len(y) > 2:
                curve.setData(x[-len(y):], y)

        self.vb1.enableAutoRange(x=True, y=True)
        self.vb2.enableAutoRange(x=True, y=True)

    # ---------------- Helpers ----------------
    def _ensure_open(self) -> bool:
        return bool(self.ser and self.ser.is_open)

# ----------------------- main -----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
