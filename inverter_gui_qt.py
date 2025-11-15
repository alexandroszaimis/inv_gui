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
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

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
LOG_N_CHANNELS    = 16
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
    ("foc_lamda","f32","1-2"),

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
# Format: (name, type) where type is: "f32"/"f64" (scaled+signed), "i16" (signed, no scale), "u16" (unsigned, no scale)
LOG_TABLE_VARIABLES = [
    ("TIME_16US", "u16"),      
    ("TIME_16MS", "u16"),     
    ("msg_cnt", "u16"),
    ("msg_toggle", "u16"),
    ("Id_ref", "f32"),         
    ("Iq_ref", "f32"),        
    ("Id", "f32"),             
    ("Iq", "f32"),            
    ("foc_psi_ref", "f32"),      
    ("foc_trq_ref", "f32"),     
    ("foc_psi_actual", "f32"), 
    ("foc_trq_actual", "f32"), 
    ("dtc_psi_ref", "f32"),       
    ("dtc_trq_ref", "f32"),      
    ("dtc_psi_actual", "f32"),  
    ("dtc_trq_actual", "f32"),    
    ("dtc_sector", "u16"),        
    ("dtc_vector", "u16"),        
    ("Ia_ph", "f32"),         
    ("Ib_ph", "f32"),
    ("I1", "f32"), 
    ("I2", "f32"), 
    ("I3", "f32"), 
    ("V1", "f32"), 
    ("V2", "f32"),
    ("V3", "f32"),         
    ("pwr_ref", "f32"),       
    ("pwr_actual", "f32"), 
    ("velocity_request", "i16"),   
    ("motor_rpm", "i16"),
    ("motor_rpm_actual", "i16"),       
    ("Vdc", "f32"),
    ("TIM1_CCR1", "u16"),
    ("TIM1_CCR2", "u16"),
    ("TIM1_CCR3", "u16"),
    ("dtc_sw_table1", "u16"),
    ("dtc_sw_table2", "u16"),
    ("dtc_sw_table3", "u16")
   
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
    ("motor_rpm_actual","i16"),
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
        # Try to use log_scales from MCU settings
        if hasattr(self.app, "_recv_values") and self.app._recv_values:
            scales_values = self.app._recv_values.get("log_scales[100]")
            slot_values = self.app._recv_values.get("log_slot[LOG_N_CHANNELS]")
            
            if isinstance(scales_values, list) and isinstance(slot_values, list):
                # Get slot index for channel i
                if i < len(slot_values):
                    slot_idx = int(slot_values[i])
                    # Use log_scales[slot_idx] as the multiplier
                    if 0 <= slot_idx < len(scales_values):
                        m = scales_values[slot_idx]
                        return float(m) if m not in (0, 0.0) else 1.0
        
        # Fallback to CHANNEL_MULTIPLIERS if log_scales not available
        if i < len(self.CHANNEL_MULTIPLIERS):
            m = self.CHANNEL_MULTIPLIERS[i]
            return float(m) if m not in (0, 0.0) else 1.0
        return 1.0

    def _decode_ch_num(self, i: int, u16_val: int) -> float:
        # Get variable type from LOG_TABLE_VARIABLES
        var_type = None
        if hasattr(self.app, "_recv_values") and self.app._recv_values:
            slot_values = self.app._recv_values.get("log_slot[LOG_N_CHANNELS]")
            if isinstance(slot_values, list) and i < len(slot_values):
                slot_idx = int(slot_values[i])
                if 0 <= slot_idx < len(LOG_TABLE_VARIABLES):
                    var_name, var_type = LOG_TABLE_VARIABLES[slot_idx]
        
        # Decode based on type:
        # - f32/f64: scaled and signed
        # - i16: signed, no scaling
        # - u16: unsigned, no scaling
        if var_type == "u16":
            # Unsigned: no sign conversion, no scaling
            return float(int(u16_val))
        elif var_type == "i16":
            # Signed: convert to signed int16, no scaling
            s16 = self._u16_to_s16(int(u16_val))
            return float(s16)
        elif var_type in ("f32", "f64"):
            # Float: signed and scaled (always scale f32/f64 types, regardless of channel position)
            if not self.SCALE_FLOATS:
                return float(int(u16_val))
            s16 = self._u16_to_s16(int(u16_val))
            return (s16 << self.SHIFT_BITS) / self._mult(i)
        else:
            # Fallback: default behavior (signed and scaled)
            # Only apply SCALE_CH1 check for fallback case (when type is unknown)
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

        # Append LOG parameters (e.g., data_logger_ts_div) if available
        try:
            ts_div_val = None
            # Prefer send field from Log Settings
            if hasattr(self.app, "log_send_vars"):
                w = self.app.log_send_vars.get("data_logger_ts_div")
                if isinstance(w, QtWidgets.QLineEdit):
                    txt = w.text().strip()
                    if txt != "":
                        try:
                            ts_div_val = int(float(txt))
                        except Exception:
                            pass
            # Fallback to received values
            if ts_div_val is None and hasattr(self.app, "_recv_values"):
                rv = self.app._recv_values.get("data_logger_ts_div")
                if rv is not None:
                    try:
                        ts_div_val = int(float(rv))
                    except Exception:
                        pass
            if ts_div_val is not None:
                lines.append("#LOG,")
                lines.append(f"data_logger_ts_div,{ts_div_val}")
                lines.append("")
        except Exception:
            pass

        # MONITOR header
        ctrl_header = ["num_max_trq","num_min_trq","max_velocity","min_velocity",
                       "velocity_req","torque_req","mode","inv_en"]
        mon_header  = [n for (n, _t) in DIAG_FIELDS]
        lines.append("#MONITOR,")
        lines.append(",".join(ctrl_header + mon_header))

        # helper function to safely convert to int, defaulting to 0 for empty strings
        def safe_int(val, default=0):
            if val == "" or val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        # helper mappers
        def map_control_row(row):
            out = dict(row)
            if "mode" in out and out["mode"]:
                out["mode"] = enum_name(CTRL_MODE_MAP, safe_int(out["mode"]))
            if "inv_en" in out and out["inv_en"] != "":
                out["inv_en"] = "ENABLED" if safe_int(out["inv_en"]) else "DISABLED"
            return out

        def map_monitor_row(row):
            out = dict(row)
            if "critical_hw_status" in out and out["critical_hw_status"] != "":
                out["critical_hw_status"] = fmt_crit(safe_int(out["critical_hw_status"]))
            if "aux_hw_status" in out and out["aux_hw_status"] != "":
                out["aux_hw_status"] = fmt_bits(safe_int(out["aux_hw_status"]),
                                                [(b, name) for b, name in LIMITER_FLAG_BITS])  # reuse bits if you wish
            for k in ("last_error","sys_status"):
                if k in out and out[k] != "":
                    out[k] = enum_name(ERRORS_MAP, safe_int(out[k]))
            if "handler_status" in out and out["handler_status"] != "":
                out["handler_status"] = enum_name(ERRORTYPE_MAP, safe_int(out["handler_status"]))
            if "latched_error" in out and out["latched_error"] != "":
                out["latched_error"] = enum_name(LATCHED_MAP, safe_int(out["latched_error"]))
            if "actual_state" in out and out["actual_state"] != "":
                out["actual_state"] = enum_name(STATE_MAP, safe_int(out["actual_state"]))
            if "control_mode_actual" in out and out["control_mode_actual"] != "":
                out["control_mode_actual"] = enum_name(CTRL_MODE_MAP, safe_int(out["control_mode_actual"]))
            if "limiter_flags" in out and out["limiter_flags"] != "":
                out["limiter_flags"] = fmt_bits(safe_int(out["limiter_flags"]), LIMITER_FLAG_BITS)
            return out

        # monitor rows (if any)
        for row in self.monitor_rows:
            cvals = map_control_row({k: row.get(k, "") for k in ctrl_header})
            mvals = map_monitor_row({k: row.get(k, "") for k in mon_header})
            lines.append(",".join([str(cvals[k]) for k in ctrl_header] +
                                  [str(mvals[k]) for k in mon_header]))

        return lines

    def build_csv(self):
        if not self.bin_path:
            raise RuntimeError("BIN path not set for CSV build")
        if not os.path.isfile(self.bin_path):
            raise RuntimeError(f"BIN file not found: {self.bin_path}")
        if not self.csv_path:
            raise RuntimeError("CSV path not set for CSV build")
        
        # Check if BIN file has data
        bin_size = os.path.getsize(self.bin_path)
        if bin_size == 0:
            raise RuntimeError(f"BIN file is empty: {self.bin_path}")

        # Get log_channels value from MCU (number of channels to write)
        log_channels = self.NC  # default fallback to self.NC
        if hasattr(self.app, "_recv_values") and self.app._recv_values:
            log_channels_val = self.app._recv_values.get("log_channels")
            if log_channels_val is not None:
                log_channels = int(log_channels_val)

        # Get log_slot array to map channel indices to variable names
        log_slot_array = []
        if hasattr(self.app, "_recv_values") and self.app._recv_values:
            slot_values = self.app._recv_values.get("log_slot[LOG_N_CHANNELS]")
            if isinstance(slot_values, list):
                log_slot_array = slot_values

        # Build channel header names from log_slot mapping
        channel_names = []
        for i in range(log_channels):
            if i < len(log_slot_array):
                slot_idx = int(log_slot_array[i])
                if 0 <= slot_idx < len(LOG_TABLE_VARIABLES):
                    var_name, var_type = LOG_TABLE_VARIABLES[slot_idx]
                    channel_names.append(var_name)
                else:
                    channel_names.append(f"CH{i+1}")  # fallback if invalid index
            else:
                channel_names.append(f"CH{i+1}")  # fallback if no slot data

        # side panel strings and padding
        side_lines = self._settings_sidepanel_lines()
        pad_along = "," * max(0, (self.COL_P - 1 - log_channels))

        # robust stream decode (fixes the 'line 1000' drift):
        # The MCU sends log_channels uint16 values per sample, not LOG_N_CHANNELS
        raw = np.fromfile(self.bin_path, dtype="<u2")
        usable = (raw.size // log_channels) * log_channels
        if usable != raw.size:
            raw = raw[:usable]
        arr = raw.reshape(-1, log_channels)

        try:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as fc:
                # header - use variable names instead of CH1, CH2, etc.
                fc.write(",".join(channel_names) + pad_along + ",#SETTINGS,\n")

                side_idx = 0
                # write rows
                for k in range(arr.shape[0]):
                    row_u16 = arr[k]
                    # Only write the first log_channels columns
                    fields = [f"{self._decode_ch_num(i, int(row_u16[i])):.9g}" for i in range(log_channels)]
                    if side_idx < len(side_lines):
                        fc.write(",".join(fields) + pad_along + "," + side_lines[side_idx] + "\n")
                        side_idx += 1
                    else:
                        fc.write(",".join(fields) + "\n")

            # Verify CSV was written successfully
            if not os.path.isfile(self.csv_path):
                raise RuntimeError(f"CSV file was not created: {self.csv_path}")
            csv_size = os.path.getsize(self.csv_path)
            if csv_size == 0:
                raise RuntimeError(f"CSV file is empty: {self.csv_path}")
        except Exception as e:
            # If CSV write failed, don't delete BIN file
            raise RuntimeError(f"Failed to write CSV file: {e}")

        # remove BIN to keep folder tidy (only if CSV was created successfully)
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

# ----------------------- CSV Loader Thread -----------------------
class CSVLoaderThread(QtCore.QThread):
    """Background thread for loading CSV data without freezing the GUI."""
    sig_data_loaded = QtCore.Signal(object)  # Emits DataFrame when loaded
    sig_error = QtCore.Signal(str)  # Emits error message
    
    def __init__(self, csv_path, needed_channels, downsample_factor, cached_data, cached_downsample, selective_range, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.needed_channels = needed_channels
        self.downsample_factor = downsample_factor
        self.cached_data = cached_data
        self.cached_downsample = cached_downsample
        self.selective_range = selective_range
    
    def run(self):
        """Load CSV data in background thread."""
        try:
            df = self._load_csv_data()
            self.sig_data_loaded.emit(df)
            # Clear references immediately after emitting to free memory
            del df
            self.cached_data = None
            import gc
            gc.collect()
        except Exception as e:
            self.sig_error.emit(str(e))
    
    def _load_csv_data(self):
        """Load CSV data - extracted from plot_csv method."""
        import pandas as pd
        import numpy as np
        
        # Always use full CSV loading
        return self._load_full_csv()
    
    def _load_full_csv(self):
        """Load full CSV with downsampling using Polars (faster and more memory-efficient)."""
        import numpy as np
        
        # Use Polars if available, fallback to pandas
        if POLARS_AVAILABLE:
            #print("DEBUG: Using Polars for CSV loading")
            return self._load_full_csv_polars()
        else:
            print("DEBUG: Using Pandas for CSV loading (Polars not available)")
            return self._load_full_csv_pandas()
    
    def _load_full_csv_polars(self):
        """Load full CSV with downsampling using Polars."""
        import numpy as np
        
        # Read header to get column names
        try:
            df_header = pl.read_csv(
                self.csv_path,
                n_rows=0,
                ignore_errors=True,
                truncate_ragged_lines=True
            )
        except Exception as e:
            # Fallback to pandas if polars fails
            print(f"DEBUG: Polars failed ({e}), falling back to Pandas")
            return self._load_full_csv_pandas()
        
        all_channel_cols = [col for col in df_header.columns 
                           if col and not str(col).startswith("#") 
                           and not str(col).strip() == ""
                           and not str(col).startswith("Unnamed")
                           and not str(col).startswith("_duplicated_")]
        
        if not all_channel_cols:
            raise ValueError("No channel columns found in CSV.")
        
        cols_to_read = [col for col in self.needed_channels if col in all_channel_cols]
        if not cols_to_read:
            raise ValueError(f"None of the selected channels found in CSV. Available: {all_channel_cols[:10]}...")
        
        # Polars: use scan_csv for lazy evaluation (memory efficient) or read_csv for eager
        downsample = int(self.downsample_factor)
        
        if self.downsample_factor > 1:
            # Use lazy evaluation with scan_csv for chunked processing
            try:
                # Read CSV lazily and filter rows with explicit schema for numeric types
                schema_dict = {col: pl.Float32 for col in cols_to_read}
                lazy_df = pl.scan_csv(
                    self.csv_path,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    schema_overrides=schema_dict
                ).select(cols_to_read)
                
                # Add row number column for downsampling
                lazy_df = lazy_df.with_row_index("__row_idx")
                
                # Filter: keep rows where row_idx % downsample_factor == 0
                lazy_df = lazy_df.filter(
                    pl.col("__row_idx") % downsample == 0
                ).drop("__row_idx")
                
                # Collect (execute) and convert to pandas for compatibility
                df_polars = lazy_df.collect()
                
                # Convert to pandas DataFrame for compatibility with existing code
                df = df_polars.to_pandas()
                
                # Debug: verify downsample_factor is correct
                if len(df) > 0:
                    print(f"DEBUG: CSVLoaderThread.downsample_factor = {self.downsample_factor}, using downsample = {downsample}")
                    expected_pattern = list(range(0, min(20, len(df)), downsample))
                    print(f"DEBUG: Expected pattern for x{downsample}: {expected_pattern}")
                    actual_rows = df.index[:10].tolist()
                    print(f"DEBUG: Actual kept rows (first 10): {actual_rows}")
                
            except Exception as e:
                # Fallback to pandas if polars fails
                print(f"DEBUG: Polars failed ({e}), falling back to Pandas")
                return self._load_full_csv_pandas()
        else:
            # No downsampling - read normally
            try:
                # Read with explicit schema to ensure numeric types
                schema_dict = {col: pl.Float32 for col in cols_to_read}
                df_polars = pl.read_csv(
                    self.csv_path,
                    columns=cols_to_read,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    schema_overrides=schema_dict
                )
                # Convert to pandas DataFrame for compatibility
                df = df_polars.to_pandas()
            except Exception as e:
                # Fallback to pandas if polars fails
                print(f"DEBUG: Polars failed ({e}), falling back to Pandas")
                return self._load_full_csv_pandas()
        
        # Convert to numeric in-place to save memory (double-check all columns are numeric)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast='float')
        df = df.dropna(how="all")
        
        # Ensure all columns are float32 (downcast might not work for all)
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32', copy=False)
        
        return df
    
    def _load_full_csv_pandas(self):
        """Load full CSV with downsampling using pandas (fallback)."""
        import pandas as pd
        import numpy as np
        
        # Read header
        try:
            df_header = pd.read_csv(
                self.csv_path,
                engine="c",
                nrows=0,
                on_bad_lines="skip"
            )
        except Exception:
            df_header = pd.read_csv(
                self.csv_path,
                engine="python",
                nrows=0,
                on_bad_lines="skip"
            )
        
        all_channel_cols = [col for col in df_header.columns 
                           if col and not str(col).startswith("#") 
                           and not str(col).strip() == ""
                           and not str(col).startswith("Unnamed")
                           and not str(col).startswith("_duplicated_")]
        
        if not all_channel_cols:
            raise ValueError("No channel columns found in CSV.")
        
        cols_to_read = [col for col in self.needed_channels if col in all_channel_cols]
        if not cols_to_read:
            raise ValueError(f"None of the selected channels found in CSV. Available: {all_channel_cols[:10]}...")
        
        # Read CSV with downsampling using chunked reading for memory efficiency
        read_params = {
            "usecols": cols_to_read,
            "dtype": np.float32,
            "on_bad_lines": "skip",
            "low_memory": False
        }
        
        if self.downsample_factor > 1:
            # Use chunked reading with proper downsampling
            # Key: track the absolute row number across chunks to maintain correct pattern
            chunk_size = 10000  # Read 10k rows at a time
            chunks = []
            absolute_row = 0  # Track absolute row number in original file (0-indexed, excluding header)
            
            try:
                chunk_iter = pd.read_csv(
                    self.csv_path,
                    engine="c",
                    chunksize=chunk_size,
                    **read_params
                )
                for chunk in chunk_iter:
                    chunk_len = len(chunk)
                    if chunk_len == 0:
                        continue
                    
                    # Create boolean mask using numpy for efficiency
                    # Calculate which rows to keep: rows where (absolute_row + i) % downsample_factor == 0
                    # This ensures we keep every Nth row consistently across all chunks
                    chunk_start_row = absolute_row
                    chunk_end_row = absolute_row + chunk_len
                    
                    # Create array of absolute row numbers for this chunk
                    absolute_rows = np.arange(chunk_start_row, chunk_end_row)
                    
                    # Create boolean mask: keep rows where absolute_row % downsample_factor == 0
                    # CRITICAL: Use self.downsample_factor directly, don't modify it
                    downsample = int(self.downsample_factor)  # Ensure it's an integer
                    
                    # Debug: verify downsample_factor is correct (only for first chunk)
                    if chunk_start_row == 0:  # First chunk
                        print(f"DEBUG: CSVLoaderThread.downsample_factor = {self.downsample_factor}, using downsample = {downsample}")
                        expected_pattern = list(range(0, min(20, chunk_len), downsample))
                        print(f"DEBUG: Expected pattern for x{downsample}: {expected_pattern}")
                    
                    # Create mask: keep rows where absolute_row % downsample == 0
                    keep_mask = (absolute_rows % downsample == 0)
                    
                    # Debug: verify first few rows for first chunk only
                    if chunk_start_row == 0:  # First chunk
                        first_kept_rows = absolute_rows[keep_mask][:10]
                        print(f"DEBUG: Actual kept rows: {first_kept_rows[:10]}")
                    
                    # Apply mask to keep only selected rows
                    # Use .iloc with boolean mask to ensure correct indexing
                    if np.any(keep_mask):
                        # Convert numpy boolean array to list for pandas compatibility
                        keep_mask_list = keep_mask.tolist()
                        downsampled_chunk = chunk.iloc[keep_mask_list].copy()
                        chunks.append(downsampled_chunk)
                    
                    # Update absolute_row for next chunk
                    absolute_row = chunk_end_row
                    
                    # Free memory periodically
                    if len(chunks) > 100:
                        df_temp = pd.concat(chunks, ignore_index=True)
                        chunks = [df_temp]
            except Exception:
                # Fallback: read all and downsample
                read_params_python = read_params.copy()
                read_params_python.pop("low_memory", None)
                df = pd.read_csv(
                    self.csv_path,
                    engine="python",
                    **read_params_python
                )
                df = df.iloc[::self.downsample_factor].copy()
                chunks = [df]
            
            # Concatenate all chunks
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.DataFrame(columns=cols_to_read)
        else:
            # No downsampling - read normally
            try:
                df = pd.read_csv(
                    self.csv_path,
                    engine="c",
                    **read_params
                )
            except Exception:
                # Python engine doesn't support low_memory parameter
                read_params_python = read_params.copy()
                read_params_python.pop("low_memory", None)
                df = pd.read_csv(
                    self.csv_path,
                    engine="python",
                    **read_params_python
                )
        
        # Convert to numeric in-place to save memory
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast='float')
        df = df.dropna(how="all")
        
        # Ensure all columns are float32 (downcast might not work for all)
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32', copy=False)
        
        return df

# ----------------------- Plot widgets (pyqtgraph) -----------------------
class TwoWindowPlot(QtWidgets.QWidget):
    # Emit x1, y1, x2, y2, dx, dy whenever either cursor moves
    cursorMoved = QtCore.Signal(float, float, float, float, float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sec_per_sample = 1.0  # Default: x-axis in samples
        self._cached_csv_data = None  # Cache CSV data to avoid re-reading
        self._cached_csv_path = None
        self._cached_downsample_factor = 1  # Track downsample factor used for cached data
        self._progress_bar = None  # Reference to progress bar (set by MainWindow)
        self._csv_loader_thread = None  # Background thread for CSV loading
        self._pending_plot_params = None  # Store plot parameters while loading
        self._current_csv_path = None  # Track current CSV path
        self._current_groups1 = None  # Track current plot groups
        self._current_groups2 = None
        self._current_link_x = True
        self._current_downsample_factor = 1  # Track current downsample factor

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

        # Add horizontal zero reference lines so users can compare signals to 0
        try:
            zero_pen = pg.mkPen(color=(180, 180, 180), width=1)
            zero_pen.setStyle(QtCore.Qt.DashLine)
        except Exception:
            zero_pen = pg.mkPen((180, 180, 180))
        self._zero_line1 = pg.InfiniteLine(pos=0.0, angle=0, pen=zero_pen, movable=False)
        self._zero_line2 = pg.InfiniteLine(pos=0.0, angle=0, pen=zero_pen, movable=False)
        # Ensure the zero lines are above plotted curves
        try:
            self._zero_line1.setZValue(1_000_000)
            self._zero_line2.setZValue(1_000_000)
        except Exception:
            pass
        self.plot1.addItem(self._zero_line1)
        self.plot2.addItem(self._zero_line2)

        # state
        self._curves = {}            # keys: (win_id, name) -> PlotDataItem
        # Vibrant color palette (no grey/dull tones)
        self._palette = [
            "#e41a1c",  # red
            "#377eb8",  # blue
            "#4daf4a",  # green
            "#984ea3",  # purple
            "#ff7f00",  # orange
            "#ffff33",  # yellow
            "#f781bf",  # pink
            "#00ffff",  # cyan
            "#1e90ff",  # dodger blue
            "#ff1493",  # deep pink
            "#00ff7f",  # spring green
            "#ffd700",  # gold
            "#ff4500",  # orange red
            "#7fff00",  # chartreuse
            "#00ced1",  # dark turquoise
            "#ff00ff"   # magenta
        ]
        self._ch_pen = {}            # name -> QPen

        # --- cursors on plot1 (two crosshairs) ---
        self._cursors_enabled = False
        cursor_pen_1 = pg.mkPen((255, 200, 0), width=1)
        cursor_pen_2 = pg.mkPen((0, 200, 255), width=1)
        self._cursor1_v = pg.InfiniteLine(angle=90, movable=True, pen=cursor_pen_1)
        self._cursor1_h = pg.InfiniteLine(angle=0,  movable=True, pen=cursor_pen_1)
        self._cursor2_v = pg.InfiniteLine(angle=90, movable=True, pen=cursor_pen_2)
        self._cursor2_h = pg.InfiniteLine(angle=0,  movable=True, pen=cursor_pen_2)
        # keep cursors above data
        try:
            for ln in (self._cursor1_v, self._cursor1_h, self._cursor2_v, self._cursor2_h):
                ln.setZValue(1_000_001)
        except Exception:
            pass
        # Add to plot1 only; X may be linked to plot2 visually
        self.plot1.addItem(self._cursor1_v)
        self.plot1.addItem(self._cursor1_h)
        self.plot1.addItem(self._cursor2_v)
        self.plot1.addItem(self._cursor2_h)
        # Start hidden until enabled via UI
        for ln in (self._cursor1_v, self._cursor1_h, self._cursor2_v, self._cursor2_h):
            ln.hide()
        # Initialize positions roughly at 1/3 and 2/3 of current view
        try:
            xr, yr = self.plot1.getViewBox().viewRange()
            xmid1 = xr[0] + (xr[1] - xr[0]) * 0.33
            xmid2 = xr[0] + (xr[1] - xr[0]) * 0.66
            ymid1 = yr[0] + (yr[1] - yr[0]) * 0.5
            ymid2 = ymid1
        except Exception:
            xmid1 = 0.0; xmid2 = 1.0; ymid1 = 0.0; ymid2 = 0.0
        self._cursor1_v.setPos(xmid1); self._cursor1_h.setPos(ymid1)
        self._cursor2_v.setPos(xmid2); self._cursor2_h.setPos(ymid2)
        # Wire move events
        try:
            self._cursor1_v.sigPositionChanged.connect(self._emit_cursor_update)
            self._cursor1_h.sigPositionChanged.connect(self._emit_cursor_update)
            self._cursor2_v.sigPositionChanged.connect(self._emit_cursor_update)
            self._cursor2_h.sigPositionChanged.connect(self._emit_cursor_update)
        except Exception:
            pass

        # --- cursors on plot2 (two crosshairs) ---
        self._cursors2_enabled = False
        self._p2_cursor1_v = pg.InfiniteLine(angle=90, movable=True, pen=cursor_pen_1)
        self._p2_cursor1_h = pg.InfiniteLine(angle=0,  movable=True, pen=cursor_pen_1)
        self._p2_cursor2_v = pg.InfiniteLine(angle=90, movable=True, pen=cursor_pen_2)
        self._p2_cursor2_h = pg.InfiniteLine(angle=0,  movable=True, pen=cursor_pen_2)
        try:
            for ln in (self._p2_cursor1_v, self._p2_cursor1_h, self._p2_cursor2_v, self._p2_cursor2_h):
                ln.setZValue(1_000_001)
        except Exception:
            pass
        self.plot2.addItem(self._p2_cursor1_v)
        self.plot2.addItem(self._p2_cursor1_h)
        self.plot2.addItem(self._p2_cursor2_v)
        self.plot2.addItem(self._p2_cursor2_h)
        for ln in (self._p2_cursor1_v, self._p2_cursor1_h, self._p2_cursor2_v, self._p2_cursor2_h):
            ln.hide()
        # Initialize
        try:
            xr2, yr2 = self.plot2.getViewBox().viewRange()
            x2_1 = xr2[0] + (xr2[1] - xr2[0]) * 0.33
            x2_2 = xr2[0] + (xr2[1] - xr2[0]) * 0.66
            y2_1 = yr2[0] + (yr2[1] - yr2[0]) * 0.5
            y2_2 = y2_1
        except Exception:
            x2_1 = 0.0; x2_2 = 1.0; y2_1 = 0.0; y2_2 = 0.0
        self._p2_cursor1_v.setPos(x2_1); self._p2_cursor1_h.setPos(y2_1)
        self._p2_cursor2_v.setPos(x2_2); self._p2_cursor2_h.setPos(y2_2)

        # Link X positions between plots (avoid recursion)
        self._updating_cursors = False
        def _on_p1_vcursor_moved():
            if self._updating_cursors:
                return
            try:
                self._updating_cursors = True
                self._p2_cursor1_v.setPos(self._cursor1_v.value())
                self._p2_cursor2_v.setPos(self._cursor2_v.value())
            finally:
                self._updating_cursors = False
            self._emit_cursor_update()
        def _on_p2_vcursor_moved():
            if self._updating_cursors:
                return
            try:
                self._updating_cursors = True
                self._cursor1_v.setPos(self._p2_cursor1_v.value())
                self._cursor2_v.setPos(self._p2_cursor2_v.value())
            finally:
                self._updating_cursors = False
            self._emit_cursor_update()
        try:
            self._cursor1_v.sigPositionChanged.connect(_on_p1_vcursor_moved)
            self._cursor2_v.sigPositionChanged.connect(_on_p1_vcursor_moved)
            self._p2_cursor1_v.sigPositionChanged.connect(_on_p2_vcursor_moved)
            self._p2_cursor2_v.sigPositionChanged.connect(_on_p2_vcursor_moved)
            # Also emit on plot2 horizontal moves to refresh any UI if needed
            self._p2_cursor1_h.sigPositionChanged.connect(self._emit_cursor_update)
            self._p2_cursor2_h.sigPositionChanged.connect(self._emit_cursor_update)
        except Exception:
            pass

        # safe defaults
        self.set_groups(
            win1_groups=[("CH2","CH3"), ("CH4","CH5"), ("CH6","CH7")],
            win2_groups=[("CH8","CH9"), ("CH10","CH11","CH12")],
            win1_titles=["Flux","Torque","Sector/Vector"],
            win2_titles=["Ia/Ib","CH10-12"],
            link_x=True
        )

    # ---------- public API ----------
    def set_cursors_enabled(self, enabled: bool):
        self._cursors_enabled = bool(enabled)
        for ln in (self._cursor1_v, self._cursor1_h, self._cursor2_v, self._cursor2_h):
            ln.setVisible(self._cursors_enabled)
        if self._cursors_enabled:
            # On enable, place cursors at 1/3 and 2/3 of current x-range
            try:
                xr, yr = self.plot1.getViewBox().viewRange()
                x1 = xr[0] + (xr[1] - xr[0]) * (1.0 / 3.0)
                x2 = xr[0] + (xr[1] - xr[0]) * (2.0 / 3.0)
                self._cursor1_v.setPos(x1)
                self._cursor2_v.setPos(x2)
            except Exception:
                pass
            self._emit_cursor_update()

    def set_time_scale(self, sec_per_sample: float | None):
        """Set base time scale (seconds per original sample). None resets to 1.0 (samples)."""
        try:
            self._sec_per_sample = float(sec_per_sample) if sec_per_sample and sec_per_sample > 0 else 1.0
        except Exception:
            self._sec_per_sample = 1.0

    def set_cursors2_enabled(self, enabled: bool):
        self._cursors2_enabled = bool(enabled)
        for ln in (self._p2_cursor1_v, self._p2_cursor1_h, self._p2_cursor2_v, self._p2_cursor2_h):
            ln.setVisible(self._cursors2_enabled)
        if self._cursors2_enabled:
            # Prefer syncing to plot1 current cursor x positions if available
            try:
                x1 = float(self._cursor1_v.value())
                x2 = float(self._cursor2_v.value())
                self._p2_cursor1_v.setPos(x1)
                self._p2_cursor2_v.setPos(x2)
            except Exception:
                try:
                    xr, yr = self.plot2.getViewBox().viewRange()
                    x1 = xr[0] + (xr[1] - xr[0]) * (1.0 / 3.0)
                    x2 = xr[0] + (xr[1] - xr[0]) * (2.0 / 3.0)
                    self._p2_cursor1_v.setPos(x1)
                    self._p2_cursor2_v.setPos(x2)
                except Exception:
                    pass
            self._emit_cursor_update()

    def get_cursor_values(self):
        """Return (x1, y1, x2, y2, dx, dy) for current cursor positions on plot1."""
        try:
            x1 = float(self._cursor1_v.value())
            y1 = float(self._cursor1_h.value())
            x2 = float(self._cursor2_v.value())
            y2 = float(self._cursor2_h.value())
        except Exception:
            x1 = y1 = x2 = y2 = 0.0
        dx = x2 - x1
        dy = y2 - y1
        return x1, y1, x2, y2, dx, dy
    
    # ----- curve access helpers for cursor tracking -----
    def get_plot_curve_names(self, win_id: int) -> list[str]:
        names = []
        for (wid, name), _curve in self._curves.items():
            if wid == win_id:
                names.append(name)
        return names

    def _get_curve_xy(self, win_id: int, name: str):
        key = (win_id, name)
        curve = self._curves.get(key)
        if curve is None:
            return None, None
        try:
            return curve.xData, curve.yData
        except Exception:
            return None, None

    def eval_curve_at_x(self, win_id: int, name: str, x_value: float):
        """Linearly interpolate y for given curve at x_value. Returns float or None."""
        try:
            x, y = self._get_curve_xy(win_id, name)
            if x is None or y is None or len(x) == 0 or len(y) == 0:
                return None
            # Ensure monotonic x (it should be)
            import numpy as _np
            xv = float(x_value)
            # Clamp to range to avoid NaN
            x0 = float(x[0]); xN = float(x[-1])
            if xv <= x0:
                return float(y[0])
            if xv >= xN:
                return float(y[-1])
            return float(_np.interp(xv, _np.asarray(x, dtype=float), _np.asarray(y, dtype=float)))
        except Exception:
            return None

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
        # Only remove curves that are explicitly not in the new groups
        # This preserves curves that might be temporarily missing from data
        for (win_id, name), curve in list(self._curves.items()):
            should_remove = False
            if win_id == 1 and name not in names1:
                should_remove = True
            elif win_id == 2 and name not in names2:
                should_remove = True
            
            if should_remove:
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
        # Capture current ranges so switching modes preserves manual zoom
        vrs = []
        for vb in (vb1, vb2):
            xr, yr = vb.viewRange()
            vrs.append((tuple(xr), tuple(yr)))
        if mode == "x":
            for idx, vb in enumerate((vb1, vb2)):
                vb.setMouseMode(pg.ViewBox.RectMode)
                # Freeze both axes to prevent autorange from resetting zoom
                vb.disableAutoRange()  # both axes
                xr, yr = vrs[idx]
                # Restore current ranges
                vb.setXRange(xr[0], xr[1], padding=0)
                vb.setYRange(yr[0], yr[1], padding=0)
                # Restrict mouse wheel zoom to x-axis only
                vb.setMouseEnabled(x=True, y=False)
        elif mode == "y":
            for idx, vb in enumerate((vb1, vb2)):
                vb.setMouseMode(pg.ViewBox.RectMode)
                # Freeze both axes to prevent autorange from resetting zoom
                vb.disableAutoRange()  # both axes
                xr, yr = vrs[idx]
                # Restore current ranges
                vb.setXRange(xr[0], xr[1], padding=0)
                vb.setYRange(yr[0], yr[1], padding=0)
                # Restrict mouse wheel zoom to y-axis only
                vb.setMouseEnabled(x=False, y=True)
        else:
            for vb in (vb1, vb2):
                vb.enableAutoRange(axis=pg.ViewBox.XYAxes)
                vb.autoRange()
                # Re-enable both axes for normal zoom
                vb.setMouseEnabled(x=True, y=True)

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
                # Always include 0 on Y axis in view
                if ymin > 0.0:
                    ymin = 0.0
                if ymax < 0.0:
                    ymax = 0.0
                if xmax == xmin:
                    xmax += 1.0
                # If all Y values are identical, expand toward 0 (not symmetric)
                if ymax == ymin:
                    k = ymax
                    if k >= 0.0:
                        ymin = 0.0
                        ymax = 1.0 if k == 0.0 else k
                    else:
                        ymin = k
                        ymax = 0.0
                vb.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0.05)
            else:
                # No data: default to a symmetric range around 0 to keep 0 visible
                vb.setRange(yRange=(-1.0, 1.0), padding=0.05)
            # Re-enable both axes for normal zoom
            vb.setMouseEnabled(x=True, y=True)
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

    def _get_visible_x_range(self, link_x=True):
        """Get the visible x-axis range from the plots (for selective high-res loading)."""
        try:
            vb1 = self.plot1.getViewBox()
            vb2 = self.plot2.getViewBox()
            # Get visible range (returns (xmin, xmax), (ymin, ymax))
            range1 = vb1.viewRange()
            range2 = vb2.viewRange()
            # Use the intersection of both ranges (or plot1 if not linked)
            xmin1, xmax1 = range1[0]
            xmin2, xmax2 = range2[0]
            # Return the intersection (or plot1 range if not linked)
            if link_x:
                xmin = max(xmin1, xmin2)
                xmax = min(xmax1, xmax2)
            else:
                xmin = xmin1
                xmax = xmax1
            return (xmin, xmax) if xmax > xmin else None
        except Exception:
            return None
    
    # ---------- internals ----------
    def _emit_cursor_update(self, *args, **kwargs):
        if not self._cursors_enabled:
            return
        x1, y1, x2, y2, dx, dy = self.get_cursor_values()
        try:
            self.cursorMoved.emit(x1, y1, x2, y2, dx, dy)
        except Exception:
            pass
    
    def get_plot2_y_values(self):
        """Return y3, y4 for plot2 cursor horizontal lines."""
        try:
            y3 = float(self._p2_cursor1_h.value())
            y4 = float(self._p2_cursor2_h.value())
        except Exception:
            y3 = y4 = 0.0
        return y3, y4
    
    def plot_csv(self, csv_path: str, groups1, groups2, titles1=None, titles2=None, link_x=True, downsample_factor=1):
        # Get list of channels we need to plot
        needed_channels = set()
        for group in groups1:
            needed_channels.update(group)
        for group in groups2:
            needed_channels.update(group)
        
        if not needed_channels:
            raise ValueError("No channels selected for plotting.")
        
        # Store current plot params for zoom reload
        self._current_csv_path = csv_path
        self._current_groups1 = groups1
        self._current_groups2 = groups2
        self._current_link_x = link_x
        self._current_downsample_factor = downsample_factor  # Store current downsample factor
        
        df = None
        
        # Check if we have cached data for this file (same downsample factor)
        # IMPORTANT: Only use cache if downsample_factor matches exactly
        # Using wrong downsample_factor cache would give incorrect data
        if (self._cached_csv_path == csv_path and 
            self._cached_csv_data is not None and 
            self._cached_downsample_factor == downsample_factor):
            cached_df = self._cached_csv_data
            cached_cols = set(cached_df.columns)
            
            # Check if we have all needed channels in cache
            if needed_channels.issubset(cached_cols):
                # Use cached data - extract only needed channels
                df = cached_df[list(needed_channels)]
            else:
                # Some channels missing - read only the missing ones and merge
                missing_channels = needed_channels - cached_cols
                if missing_channels:
                    # Read only missing channels and merge with cache
                    try:
                        # Read header to verify channels exist
                        if POLARS_AVAILABLE:
                            try:
                                #print("DEBUG: Using Polars for CSV header reading (cache merge)")
                                df_header_polars = pl.read_csv(csv_path, n_rows=0, ignore_errors=True, truncate_ragged_lines=True)
                                df_header = df_header_polars.to_pandas()  # Convert to pandas for compatibility
                            except Exception as e:
                                print(f"DEBUG: Polars header read failed ({e}), falling back to Pandas")
                                try:
                                    df_header = pd.read_csv(csv_path, engine="c", nrows=0, on_bad_lines="skip")
                                except Exception:
                                    df_header = pd.read_csv(csv_path, engine="python", nrows=0, on_bad_lines="skip")
                        else:
                            print("DEBUG: Using Pandas for CSV header reading (cache merge, Polars not available)")
                            try:
                                df_header = pd.read_csv(csv_path, engine="c", nrows=0, on_bad_lines="skip")
                            except Exception:
                                df_header = pd.read_csv(csv_path, engine="python", nrows=0, on_bad_lines="skip")
                        
                        # Verify missing channels exist in CSV
                        available_cols = set(df_header.columns)
                        cols_to_read = [col for col in missing_channels if col in available_cols]
                        
                        if cols_to_read:
                            # Load missing channels in background thread to keep GUI responsive
                            # Stop any existing loader thread
                            if self._csv_loader_thread and self._csv_loader_thread.isRunning():
                                self._csv_loader_thread.terminate()
                                self._csv_loader_thread.wait()
                                self._csv_loader_thread = None
                            
                            # Show loading progress bar
                            if self._progress_bar:
                                self._progress_bar.setMinimum(0)
                                self._progress_bar.setMaximum(0)
                                self._progress_bar.setTextVisible(False)
                                self._progress_bar.setFormat("")
                                self._progress_bar.show()
                            
                            # Store parameters for background loading
                            self._pending_plot_params = {
                                'csv_path': csv_path,
                                'groups1': groups1,
                                'groups2': groups2,
                                'titles1': titles1,
                                'titles2': titles2,
                                'link_x': link_x,
                                'downsample_factor': downsample_factor,
                                'merge_with_cache': cached_df,  # Pass cached data for merging
                                'missing_channels': cols_to_read
                            }
                            
                            # Start background thread to load missing channels
                            self._csv_loader_thread = CSVLoaderThread(
                                csv_path,
                                set(cols_to_read),  # Only load missing channels
                                downsample_factor,
                                cached_df,  # Pass cached data for merging
                                cached_downsample,
                                None,
                                self
                            )
                            self._csv_loader_thread.sig_data_loaded.connect(self._on_cache_merge_data_loaded)
                            self._csv_loader_thread.sig_error.connect(self._on_csv_load_error)
                            self._csv_loader_thread.start()
                            return  # Exit early - plotting will happen when data is loaded
                        else:
                            # Channels don't exist, use what we have
                            df = cached_df[list(needed_channels & cached_cols)]
                    except Exception:
                        # If merge fails, fall back to full reload
                        self._cached_csv_data = None
                        self._cached_csv_path = None
        
        # Load with downsampling if needed - use background thread to keep GUI responsive
        if df is None and downsample_factor > 1:
            # Use background thread for downsampled loading too
            pass  # Fall through to background thread loading below
        
        # Load CSV if not cached - use background thread
        if df is None:
            # Stop any existing loader thread and clear its data to free memory
            if self._csv_loader_thread and self._csv_loader_thread.isRunning():
                self._csv_loader_thread.terminate()
                self._csv_loader_thread.wait()
                # Clear thread's data to free memory
                if hasattr(self._csv_loader_thread, 'df'):
                    del self._csv_loader_thread.df
                self._csv_loader_thread = None
            
            # Show loading progress bar
            if self._progress_bar:
                self._progress_bar.setMinimum(0)
                self._progress_bar.setMaximum(0)
                self._progress_bar.setTextVisible(False)
                self._progress_bar.setFormat("")
                self._progress_bar.show()
            
            # Store plot parameters for when data is loaded
            self._pending_plot_params = {
                'csv_path': csv_path,
                'groups1': groups1,
                'groups2': groups2,
                'titles1': titles1,
                'titles2': titles2,
                'link_x': link_x,
                'downsample_factor': downsample_factor
            }
            
            # Start background thread to load CSV (full file with downsampling)
            # Debug: verify downsample_factor is correct
            #print(f"DEBUG: plot_csv called with downsample_factor = {downsample_factor}")
            self._csv_loader_thread = CSVLoaderThread(
                csv_path,
                needed_channels,
                downsample_factor,  # Pass downsample_factor directly
                None,  # No cached data for background thread
                1,     # No cached downsample
                None,  # No selective range - always load full file
                self
            )
            #print(f"DEBUG: CSVLoaderThread created with downsample_factor = {self._csv_loader_thread.downsample_factor}")
            self._csv_loader_thread.sig_data_loaded.connect(self._on_csv_data_loaded)
            self._csv_loader_thread.sig_error.connect(self._on_csv_load_error)
            self._csv_loader_thread.start()
            return  # Exit early - plotting will happen when data is loaded
        
        # If we have data, plot it immediately
        self._do_plot(df, groups1, groups2, titles1, titles2, link_x, downsample_factor)
    
    def _on_csv_data_loaded(self, df):
        """Called when CSV data is loaded in background thread."""
        if self._pending_plot_params is None:
            return
        
        # Check if this is a cache merge operation
        import pandas as pd
        if 'merge_with_cache' in self._pending_plot_params:
            # This is a cache merge - merge the new data with cached data
            cached_df = self._pending_plot_params['merge_with_cache']
            df = pd.concat([cached_df, df], axis=1)
        
        # Update cache
        csv_path = self._pending_plot_params['csv_path']
        # Don't copy - use DataFrame directly to save memory
        self._cached_csv_data = df
        self._cached_csv_path = csv_path
        self._cached_downsample_factor = self._csv_loader_thread.downsample_factor
        
        # Clear thread references immediately to free memory
        self._csv_loader_thread.cached_data = None
        thread_ref = self._csv_loader_thread
        self._csv_loader_thread = None
        del thread_ref
        
        # Hide progress bar
        if self._progress_bar:
            self._progress_bar.setVisible(False)
        
        # Extract needed channels (use view, not copy)
        needed_channels = set()
        for group in self._pending_plot_params['groups1']:
            needed_channels.update(group)
        for group in self._pending_plot_params['groups2']:
            needed_channels.update(group)
        
        if needed_channels:
            available_channels = [col for col in needed_channels if col in df.columns]
            if available_channels:
                # Filter columns - this creates a view, not a copy (pandas is smart about this)
                df = df[available_channels].copy()  # Actually need copy for thread safety, but minimize columns first
        
        # Update visible range for zoom detection
        visible_range = self._get_visible_x_range(self._pending_plot_params['link_x'])
        self._current_visible_range = visible_range
        
        # Plot the data
        self._do_plot(
            df,
            self._pending_plot_params['groups1'],
            self._pending_plot_params['groups2'],
            self._pending_plot_params['titles1'],
            self._pending_plot_params['titles2'],
            self._pending_plot_params['link_x'],
            self._pending_plot_params['downsample_factor']
        )
        
        self._pending_plot_params = None
    
    def _on_cache_merge_data_loaded(self, df):
        """Called when missing channels are loaded for cache merge - same as _on_csv_data_loaded."""
        self._on_csv_data_loaded(df)
    
    def _on_csv_load_error(self, error_msg):
        """Called when CSV loading fails in background thread."""
        print(f"CSV load error: {error_msg}")
        if self._progress_bar:
            self._progress_bar.setVisible(False)
        # Fall back to x4
        if self._pending_plot_params:
            self.plot_csv(
                self._pending_plot_params['csv_path'],
                self._pending_plot_params['groups1'],
                self._pending_plot_params['groups2'],
                link_x=self._pending_plot_params['link_x'],
                downsample_factor=4  # Fall back to x4
            )
        self._pending_plot_params = None
    
    def _do_plot(self, df, groups1, groups2, titles1, titles2, link_x, downsample_factor):
        """Actually perform the plotting (called after data is loaded)."""
        if df.empty or len(df.columns) == 0:
            raise ValueError("No valid channel data found in CSV.")

        try:
            # Process events before starting to keep GUI responsive
            QtWidgets.QApplication.processEvents()

            # (A) Configure groups/curves first
            self.set_groups(groups1, groups2, titles1, titles2, link_x)

            # Process events after set_groups to keep GUI responsive
            QtWidgets.QApplication.processEvents()

            # Build x-axis array
            n = len(df.index)
            # Scale x-axis by downsample factor and seconds per original sample (if set)
            x_stride = float(downsample_factor) * float(getattr(self, "_sec_per_sample", 1.0))
            x_full = np.arange(n, dtype=np.float64) * x_stride
            
            # Set dynamic x-limits to current data span to avoid overflow in ViewBox
            try:
                x_min = 0.0
                # IMPORTANT: x axis is in seconds when time scaling is active.
                # Use x_stride to compute limits in the same units as x_full.
                x_max = float((n - 1) * x_stride) if n > 0 else 1.0
                max_range = max(1.0, x_max - x_min)
                # Allow zooming down to one-sample width (in seconds if time-scaled)
                min_x_range = max(1e-9, float(downsample_factor) * float(getattr(self, "_sec_per_sample", 1.0)))
                # Compute y-limits from the columns we are about to plot
                used_names = {name for (_wid, name) in self._curves.keys()} & set(df.columns)
                if used_names:
                    # Stack selected columns to compute finite min/max efficiently
                    y_data = df[list(used_names)].to_numpy(dtype=np.float32, copy=False)
                    # Convert non-finite to NaN, then compute nanmin/nanmax safely
                    y_min = float(np.nanmin(y_data)) if np.isfinite(y_data).any() else 0.0
                    y_max = float(np.nanmax(y_data)) if np.isfinite(y_data).any() else 1.0
                    if not np.isfinite(y_min):
                        y_min = 0.0
                    if not np.isfinite(y_max):
                        y_max = 1.0
                    if y_max == y_min:
                        y_max = y_min + 1.0
                    y_range = max(1e-6, y_max - y_min)
                    min_y_range = max(1e-6, y_range * 1e-6)
                    max_y_range = max(y_range, 1.0) * 1000.0
                else:
                    y_min, y_max = 0.0, 1.0
                    min_y_range, max_y_range = 1e-6, 1.0
                for vb in (self.plot1.getViewBox(), self.plot2.getViewBox()):
                    vb.setLimits(
                        xMin=x_min, xMax=x_max, minXRange=min_x_range, maxXRange=max_range,
                        yMin=y_min, yMax=y_max, minYRange=min_y_range, maxYRange=max_y_range
                    )
            except Exception:
                pass

            # IMPORTANT: When data is already downsampled (downsample_factor > 1), 
            # we should NOT apply additional adaptive decimation, as it would further reduce resolution.
            # The data is already at the desired resolution from CSV loading.
            # For x1, show ALL data points - no decimation at all.
            if downsample_factor == 1:
                # At x1, show all data points - no decimation
                ds = 1
                x = x_full
            else:
                # For downsampled data (x2, x4, x8, etc.), use all points - no additional decimation
                # The data is already at the correct resolution from CSV loading
                ds = 1
                x = x_full

            # (C) Push data - this can take time with large datasets
            num_curves = len(self._curves)
            curve_idx = 0
            for (win_id, name), curve in self._curves.items():
                if name in df.columns:
                    # Use .values directly and convert to float32 to avoid unnecessary copies
                    y_col = df[name].values.astype(np.float32, copy=False)
                    # Guard against non-finite values triggering autorange overflow
                    if not np.isfinite(y_col).any():
                        curve.setData(x=[], y=[])
                        continue
                    # x is already decimated, y needs to match
                    curve.setData(x=x, y=y_col[::ds], connect='finite', skipFiniteCheck=True)
                    curve_idx += 1
                    # Update UI more frequently to keep GUI responsive, especially for second plot
                    if curve_idx % 3 == 0:  # Update every 3 curves (more frequent)
                        QtWidgets.QApplication.processEvents()
                else:
                    # If a curve doesn't have data, clear it
                    # (but don't remove it - it might be added in next update)
                    curve.setData(x=[], y=[])
            
            # Final processEvents to ensure all updates are rendered
            QtWidgets.QApplication.processEvents()
        finally:
            # Hide loading progress bar after plotting completes
            if self._progress_bar:
                self._progress_bar.setVisible(False)



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
        # Throttle UI updates to prevent lag when MCU sends DIAG_STATUS frequently
        self._diag_update_timer = QtCore.QTimer(self)
        self._diag_update_timer.setSingleShot(True)
        self._diag_update_timer.timeout.connect(self._apply_pending_diag_update)
        self._pending_diag_values = None  # Store latest values for deferred UI update
        self._diag_update_interval_ms = 50  # Update UI at most every 50ms (20 Hz)
        # Throttle live plot updates separately (plots can handle higher rates)
        self._live_plot_timer = QtCore.QTimer(self)
        self._live_plot_timer.setSingleShot(True)
        self._live_plot_timer.timeout.connect(self._apply_pending_live_plot_update)
        self._pending_live_plot_values = None
        self._live_plot_update_interval_ms = 33  # Update plots at most every 33ms (~30 Hz)
        self._last_status_text = ("", "")  # Cache status text to avoid unnecessary updates
        self._fft_windows = []  # Keep references to FFT dialogs to prevent GC
        
        # ============ CSV request profile ============
        self.csv_request_data = None  # List of (time, request) tuples
        self.csv_request_start_time = None  # Time when CSV playback started
        self.csv_request_path = None  # Path to loaded CSV file
        self.csv_sequence_finished = False  # Flag to track if sequence has finished
        self.csv_sequence_started = False  # Flag to track if sequence was explicitly started
        self.csv_logging_started_auto = False  # Flag to track if logging was started automatically
        self.csv_periodic_started_auto = False  # Flag to track if periodic send was started automatically

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

        self.options_btn = QtWidgets.QToolButton()
        self.options_btn.setToolTip("Options")
        self.options_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.options_btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.options_btn.setIconSize(QtCore.QSize(16, 16))

        # Try to load a proper "settings/gear" icon; fall back to ⚙ if not available
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
                # Connect Fs_trq changes to log baudrate update
                if name == "Fs_trq":
                    e.textChanged.connect(self._update_log_baudrate)
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
            if name in ("log_scales[100]", "log_slot[LOG_N_CHANNELS]", "log_channels", "params_init_key"):
                continue  # Handle arrays, log_channels, and params_init_key separately
            
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            e = QtWidgets.QLineEdit()
            e.setFixedWidth(110)
            self.log_send_vars[name] = e
            # Connect data_logger_ts_div changes to baudrate update
            if name == "data_logger_ts_div":
                e.textChanged.connect(self._update_log_baudrate)
            grid.addWidget(e, row, 1)
            rv = QtWidgets.QLabel("—")
            self.log_recv_vars[name] = rv
            grid.addWidget(rv, row, 2)
            grid.addWidget(QtWidgets.QLabel(unit), row, 3)
            row += 1
        
        inner_layout.addLayout(grid)
        
        # Baudrate calculation display
        baudrate_layout = QtWidgets.QHBoxLayout()
        baudrate_layout.addWidget(QtWidgets.QLabel("Required Baudrate:"))
        self.log_baudrate_label = QtWidgets.QLabel("—")
        self.log_baudrate_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        baudrate_layout.addWidget(self.log_baudrate_label)
        baudrate_layout.addWidget(QtWidgets.QLabel("Mbps"))
        # Warning label for high baudrate
        self.log_baudrate_warning = QtWidgets.QLabel("")
        self.log_baudrate_warning.setStyleSheet("color: red; font-weight: bold;")
        baudrate_layout.addWidget(self.log_baudrate_warning)
        baudrate_layout.addStretch()
        inner_layout.addLayout(baudrate_layout)
        
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
            # Update baudrate calculation
            self._update_log_baudrate()
        
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
        # Save current selections before clearing widgets
        saved_selections = {}
        saved_scales = {}
        if hasattr(self, 'log_send_vars'):
            for i in range(len(self.log_slot_widgets)):
                slot_name = f"log_slot[{i}]"
                scale_key = f"scale_{i}"
                slot_widget = self.log_send_vars.get(slot_name)
                scale_widget = self.log_scales_widgets.get(scale_key) if hasattr(self, 'log_scales_widgets') else None
                if isinstance(slot_widget, QtWidgets.QComboBox):
                    saved_selections[i] = slot_widget.currentData()
                if isinstance(scale_widget, QtWidgets.QLineEdit):
                    saved_scales[i] = scale_widget.text()
        
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
            for idx, (var_name, var_type) in enumerate(available_vars):
                send_combo.addItem(f"{idx} - {var_name} ({var_type})", idx)
            
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
            # Restore saved selection if available, otherwise use default
            if i in saved_selections and saved_selections[i] is not None:
                # Find the index in the combo that matches the saved data value
                # Block signals to prevent dropdown from popping up
                send_combo.blockSignals(True)
                saved_data = saved_selections[i]
                for idx in range(send_combo.count()):
                    if send_combo.itemData(idx) == saved_data:
                        send_combo.setCurrentIndex(idx)
                        initial_slot_idx = saved_data
                        break
                else:
                    # If saved value not found, use default
                    initial_slot_idx = send_combo.currentData() if send_combo.count() > 0 else 0
                send_combo.blockSignals(False)
            else:
                # Set initial scale label based on default selection (index 0 = TIME_16US)
                initial_slot_idx = send_combo.currentData() if send_combo.count() > 0 else 0
            
            if initial_slot_idx is not None:
                scale_label.setText(f"log_scale[{initial_slot_idx}]:")
                self._update_mcu_scale_display(i, initial_slot_idx)
            
            # Restore saved scale value if available
            if i in saved_scales:
                scale_edit.setText(saved_scales[i])
            
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
                    # Special handling for params_init_key - always set to 1122867
                    if name == "params_init_key":
                        p += struct.pack(LE + "I", 1122867 & 0xFFFFFFFF)
                    else:
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
                if is_array_field(name) or name == "log_channels" or name == "params_init_key":
                    continue  # Arrays, log_channels, and params_init_key handled separately
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
                            # Block signals to prevent dropdown from popping up
                            w.blockSignals(True)
                            for j in range(w.count()):
                                if w.itemData(j) == slot_val:
                                    w.setCurrentIndex(j)
                                    break
                            w.blockSignals(False)
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
                if is_array_field(name) or name == "log_channels" or name == "params_init_key":
                    continue  # Arrays, log_channels, and params_init_key handled separately
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
                # Update baudrate calculation
                self._update_log_baudrate()
            elif name in [f[0] for f in LOG_FIELDS if not is_array_field(f[0])]:
                # Skip params_init_key - it's not displayed
                if name == "params_init_key":
                    continue
                # Other log fields
                rv = getattr(self, 'log_recv_vars', {}).get(name)
                if rv:
                    if isinstance(val, float):
                        rv.setText(f"{val:.6f}".rstrip("0").rstrip("."))
                    else:
                        rv.setText(str(val))
                # Update baudrate if data_logger_ts_div changed
                if name == "data_logger_ts_div":
                    self._update_log_baudrate()
            else:
                rv = self.recv_vars.get(name)
                if rv:
                    if isinstance(val, float):
                        rv.setText(f"{val:.6f}".rstrip("0").rstrip("."))
                    else:
                        rv.setText(str(val))
                # Update baudrate if Fs_trq changed
                if name == "Fs_trq":
                    self._update_log_baudrate()
        
        # Update log slot displays after all fields are processed
        self._update_log_slots_from_recv()
        # Update log scales after slots are set
        self._update_log_scales_from_recv()
        # Update baudrate calculation after all settings are loaded
        self._update_log_baudrate()
    
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
    
    def _update_log_baudrate(self):
        """Calculate and update the required baudrate for data logging."""
        try:
            # Get number of channels
            num_channels = 1  # default
            if hasattr(self, 'log_channel_combo'):
                num_ch = self.log_channel_combo.currentData()
                if num_ch is not None:
                    num_channels = num_ch
            
            # Get data_logger_ts_div (prefer send field, fallback to received value)
            ts_div = None
            if "data_logger_ts_div" in self.log_send_vars:
                ts_div_str = self.log_send_vars["data_logger_ts_div"].text().strip()
                if ts_div_str:
                    try:
                        ts_div = float(ts_div_str)
                    except ValueError:
                        pass
            # Fallback to received value if send field is empty
            if ts_div is None and hasattr(self, '_recv_values'):
                ts_div_val = self._recv_values.get("data_logger_ts_div")
                if ts_div_val is not None:
                    try:
                        ts_div = float(ts_div_val)
                    except (ValueError, TypeError):
                        pass
            
            # Get Fs_trq (prefer send field, fallback to received value)
            fs_trq = None
            if "Fs_trq" in self.send_vars:
                fs_trq_str = self.send_vars["Fs_trq"].text().strip()
                if fs_trq_str:
                    try:
                        fs_trq = float(fs_trq_str)
                    except ValueError:
                        pass
            # Fallback to received value if send field is empty
            if fs_trq is None and hasattr(self, '_recv_values'):
                fs_trq_val = self._recv_values.get("Fs_trq")
                if fs_trq_val is not None:
                    try:
                        fs_trq = float(fs_trq_val)
                    except (ValueError, TypeError):
                        pass
            
            # Calculate baudrate: num_channels * sizeof(uint16_t) * Fs_trq / data_logger_ts_div
            # sizeof(uint16_t) = 2 bytes
            # Fs_trq is in KHz, convert to Hz: Fs_trq * 1000
            # Result in bytes per second, then convert to bits per second: * 8
            if ts_div is not None and ts_div > 0 and fs_trq is not None and fs_trq > 0:
                # Formula: num_channels * 2 bytes * Fs_trq(KHz) * 1000 / ts_div = bytes/sec
                bytes_per_sec = num_channels * 2 * fs_trq * 1000 / ts_div
                # Convert to bits per second, then to Mbps
                baudrate_bps = bytes_per_sec * 8
                baudrate_mbps = baudrate_bps / 1_000_000
                self.log_baudrate_label.setText(f"{baudrate_mbps:.3f}")
                
                # Check if baudrate is too high (> 10 Mbps)
                if baudrate_mbps > 10.0:
                    # Change color to red and show warning
                    self.log_baudrate_label.setStyleSheet("font-weight: bold; color: red;")
                    self.log_baudrate_warning.setText("⚠ Baudrate too high!")
                else:
                    # Normal color (blue)
                    self.log_baudrate_label.setStyleSheet("font-weight: bold; color: #0066cc;")
                    self.log_baudrate_warning.setText("")
            else:
                self.log_baudrate_label.setText("—")
                self.log_baudrate_label.setStyleSheet("font-weight: bold; color: #0066cc;")
                self.log_baudrate_warning.setText("")
        except Exception:
            self.log_baudrate_label.setText("—")
            self.log_baudrate_label.setStyleSheet("font-weight: bold; color: #0066cc;")
            self.log_baudrate_warning.setText("")
    
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
            # Block signals to prevent dropdown from popping up
            self.log_channel_combo.blockSignals(True)
            for i in range(self.log_channel_combo.count()):
                if self.log_channel_combo.itemData(i) == num_channels:
                    self.log_channel_combo.setCurrentIndex(i)
                    break
            self.log_channel_combo.blockSignals(False)
            
            # Trigger slot widget update manually (since we blocked signals)
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
                # Block signals to prevent dropdown from popping up
                send_w.blockSignals(True)
                for j in range(send_w.count()):
                    if send_w.itemData(j) == slot_val:
                        send_w.setCurrentIndex(j)
                        break
                send_w.blockSignals(False)
                # Update scale label when slot is set
                scale_label = scales_widgets.get(f"label_{i}")
                if scale_label:
                    scale_label.setText(f"log_scale[{slot_val}]:")
                # Update MCU scale display
                self._update_mcu_scale_display(i, slot_val)
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
            if is_array_field(name) or name == "log_channels" or name == "params_init_key":
                continue  # Arrays, log_channels, and params_init_key handled separately
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
                        # Block signals to prevent dropdown from popping up
                        w.blockSignals(True)
                        for j in range(w.count()):
                            if w.itemData(j) == slot_val:
                                w.setCurrentIndex(j)
                                break
                        w.blockSignals(False)
                    elif isinstance(w, QtWidgets.QLineEdit):
                        w.setText(str(slot_val))
        # Copy log scales (update per-channel scale inputs)
        if "log_scales[100]" in self._recv_values:
            self._update_log_scales_from_recv()
        self._set_status("MCU → Current settings copied ✅")

    # ---------------- CSV Request Profile ----------------
    def _load_request_csv(self):
        """Load CSV file with time,request columns for request profile."""
        # Default to input_reference folder (relative to script directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_ref_dir = os.path.join(script_dir, "input_reference")
        # Use input_reference if it exists, otherwise use current directory
        default_dir = input_ref_dir if os.path.isdir(input_ref_dir) else ""
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Request CSV", default_dir, "CSV (*.csv)"
        )
        if not path:
            return
        
        try:
            # Read CSV file
            if POLARS_AVAILABLE:
                try:
                    df = pl.read_csv(path, ignore_errors=True, truncate_ragged_lines=True)
                    df = df.to_pandas()
                except Exception:
                    df = pd.read_csv(path, engine="c", on_bad_lines="skip")
            else:
                df = pd.read_csv(path, engine="c", on_bad_lines="skip")
            
            # Verify required columns exist
            if "time" not in df.columns or "request" not in df.columns:
                QtWidgets.QMessageBox.warning(
                    self, "CSV Error", 
                    "CSV must contain 'time' and 'request' columns."
                )
                return
            
            # Extract time and request columns, convert to numeric
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df["request"] = pd.to_numeric(df["request"], errors="coerce")
            
            # Remove NaN rows
            df = df.dropna(subset=["time", "request"])
            
            if len(df) == 0:
                QtWidgets.QMessageBox.warning(
                    self, "CSV Error", 
                    "No valid data found in CSV."
                )
                return
            
            # Sort by time
            df = df.sort_values("time")
            
            # Store as list of tuples (time, request)
            self.csv_request_data = list(zip(df["time"].values, df["request"].values))
            self.csv_request_path = path
            
            # Update label
            filename = os.path.basename(path)
            self.csv_request_label.setText(f"Loaded: {filename} ({len(self.csv_request_data)} points)")
            self.csv_request_label.setStyleSheet("color: #0066cc; font-weight: bold;")
            
            # Update value display to show first value or "Ready"
            if hasattr(self, 'csv_request_value_label'):
                if len(self.csv_request_data) > 0:
                    first_value = int(self.csv_request_data[0][1])
                    self.csv_request_value_label.setText(str(first_value))
                else:
                    self.csv_request_value_label.setText("Ready")
                self.csv_request_value_label.setStyleSheet("color: #0066cc; font-weight: bold; min-width: 80px;")
            
            # Reset start time, finished flag, and started flag
            self.csv_request_start_time = None
            self.csv_sequence_finished = False
            self.csv_sequence_started = False
            
            # Enable start button
            if hasattr(self, 'csv_start_btn'):
                self.csv_start_btn.setEnabled(True)
            
            self._set_status(f"Request CSV loaded: {filename} ✅")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "CSV Error", 
                f"Failed to load CSV:\n{e}"
            )
    
    def _start_csv_sequence(self):
        """Start/reset or stop the CSV sequence playback."""
        if self.csv_request_data is None or len(self.csv_request_data) == 0:
            QtWidgets.QMessageBox.warning(
                self, "CSV Error",
                "No CSV loaded. Please load a CSV file first."
            )
            return
        
        # If already started, stop it and set request to 0
        if self.csv_sequence_started:
            self.csv_sequence_started = False
            self.csv_request_start_time = None
            self.csv_sequence_finished = False
            
            # Stop logging if it was started automatically
            if self.csv_logging_started_auto and self.logging:
                self._toggle_logging()
                self.csv_logging_started_auto = False
            
            # Set request value to 0
            mode = int(self.ctrl_mode.currentData())
            if mode == 1:  # SPEED CONTROL
                self.ctrl_velocity_req.blockSignals(True)
                self.ctrl_velocity_req.setValue(0)
                self.ctrl_velocity_req.blockSignals(False)
                self._ctrl_vals["velocity_req"] = 0
            elif mode == 2:  # TORQUE CONTROL
                self.ctrl_torque_req.blockSignals(True)
                self.ctrl_torque_req.setValue(0)
                self.ctrl_torque_req.blockSignals(False)
                self._ctrl_vals["torque_req"] = 0
            
            # Update label
            filename = os.path.basename(self.csv_request_path) if self.csv_request_path else "CSV"
            self.csv_request_label.setText(f"Stopped: {filename}")
            self.csv_request_label.setStyleSheet("color: #666; font-weight: normal;")
            
            # Update value display to show 0
            if hasattr(self, 'csv_request_value_label'):
                self.csv_request_value_label.setText("0")
                self.csv_request_value_label.setStyleSheet("color: #666; font-weight: normal; min-width: 80px;")
            
            # If we auto-started periodic send for the CSV sequence, stop it now
            if getattr(self, "csv_periodic_started_auto", False):
                if hasattr(self, "periodic_chk") and self.periodic_chk.isChecked():
                    self.periodic_chk.setChecked(False)
                self.csv_periodic_started_auto = False
            
            self._set_status("CSV sequence stopped ✅")
            return
        
        # Start/reset the sequence
        # Reset start time and finished flag, mark as started
        self.csv_request_start_time = None
        self.csv_sequence_finished = False
        self.csv_sequence_started = True
        
        # Auto-start periodic control send if not already active (so CSV values get sent)
        if hasattr(self, "periodic_timer") and not self.periodic_timer.isActive():
            if hasattr(self, "periodic_chk"):
                # Use the checkbox to go through the normal path (opens port, sets interval)
                self.periodic_chk.setChecked(True)
                self.csv_periodic_started_auto = True
            else:
                # Fallback: start timer directly at 50ms
                self.periodic_timer.setInterval(50)
                self.periodic_timer.start()
                self.csv_periodic_started_auto = True
        else:
            self.csv_periodic_started_auto = False
        
        # Start logging if checkbox is checked and logging is not already active
        if hasattr(self, 'csv_log_with_sequence_chk') and self.csv_log_with_sequence_chk.isChecked():
            if not self.logging:
                self._toggle_logging()
                self.csv_logging_started_auto = True
            else:
                # Logging already active, don't mark as auto-started
                self.csv_logging_started_auto = False
        else:
            self.csv_logging_started_auto = False
        
        # Update label
        filename = os.path.basename(self.csv_request_path) if self.csv_request_path else "CSV"
        self.csv_request_label.setText(f"Playing: {filename} ({len(self.csv_request_data)} points)")
        self.csv_request_label.setStyleSheet("color: #00aa00; font-weight: bold;")
        
        # Update value display to show first value
        if hasattr(self, 'csv_request_value_label') and len(self.csv_request_data) > 0:
            first_value = int(self.csv_request_data[0][1])
            self.csv_request_value_label.setText(str(first_value))
            self.csv_request_value_label.setStyleSheet("color: #00aa00; font-weight: bold; min-width: 80px;")
        
        self._set_status("CSV sequence started ✅")
    
    def _get_csv_request_value(self):
        """Get current request value from CSV based on elapsed time.
        Returns the exact CSV value for the current time point (no interpolation).
        Returns None if CSV not loaded, periodic send not active, or sequence not started.
        """
        if self.csv_request_data is None or len(self.csv_request_data) == 0:
            return None
        
        if not self.periodic_timer.isActive():
            return None
        
        # Only return values if sequence was explicitly started
        if not self.csv_sequence_started:
            return None
        
        # If sequence already finished, don't return values
        if self.csv_sequence_finished:
            return None
        
        # Initialize start time on first call
        if self.csv_request_start_time is None:
            self.csv_request_start_time = time.monotonic()
        
        # Calculate elapsed time
        elapsed_time = time.monotonic() - self.csv_request_start_time
        
        # Find the exact value - use the CSV value for the time point we're at
        times = [t for t, _ in self.csv_request_data]
        requests = [r for _, r in self.csv_request_data]
        
        # If before first time point, use first value
        if elapsed_time <= times[0]:
            return requests[0]
        
        # If after last time point, sequence finished
        if elapsed_time >= times[-1]:
            if not self.csv_sequence_finished:
                self.csv_sequence_finished = True
                filename = os.path.basename(self.csv_request_path) if self.csv_request_path else "CSV"
                self.csv_request_label.setText(f"Sequence finished: {filename}")
                self.csv_request_label.setStyleSheet("color: #ff6600; font-weight: bold;")
                # Update value display
                if hasattr(self, 'csv_request_value_label'):
                    self.csv_request_value_label.setText("Finished")
                    self.csv_request_value_label.setStyleSheet("color: #ff6600; font-weight: bold; min-width: 80px;")
                
                # Stop logging if it was started automatically
                if self.csv_logging_started_auto and self.logging:
                    self._toggle_logging()
                    self.csv_logging_started_auto = False
                
                # If we auto-started periodic send for the CSV sequence, stop it now
                if getattr(self, "csv_periodic_started_auto", False):
                    if hasattr(self, "periodic_chk") and self.periodic_chk.isChecked():
                        self.periodic_chk.setChecked(False)
                    self.csv_periodic_started_auto = False
                
                self._set_status("CSV sequence finished ✅")
            return requests[-1]  # Return last value
        
        # Find the closest time point (use the previous point's value, not interpolate)
        # This gives step-like behavior matching the CSV exactly
        for i in range(len(times) - 1):
            if times[i] <= elapsed_time < times[i + 1]:
                # Use the value from the current time point (no interpolation)
                return requests[i]
        
        # If exactly at a time point
        for i in range(len(times)):
            if abs(elapsed_time - times[i]) < 0.001:  # Small tolerance for floating point
                return requests[i]
        
        return requests[-1]  # Fallback

    # ---------------- Periodic control ----------------
    def _on_periodic_toggled(self, checked: bool):
        if checked:
            # fixed 50 ms period
            fixed_period = 50
            self.periodic_timer.setInterval(fixed_period)
            
            # Reset CSV start time when starting periodic send
            if self.csv_request_data is not None:
                self.csv_request_start_time = None

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
        
        # Apply CSV request if available and periodic send is active
        csv_request_value = self._get_csv_request_value()
        if csv_request_value is not None:
            # Update display label
            if hasattr(self, 'csv_request_value_label'):
                self.csv_request_value_label.setText(str(int(csv_request_value)))
                self.csv_request_value_label.setStyleSheet("color: #00aa00; font-weight: bold; min-width: 80px;")
            
            mode = int(self.ctrl_mode.currentData())
            if mode == 1:  # SPEED CONTROL
                self._ctrl_vals["velocity_req"] = int(csv_request_value)
                # Update widget to reflect CSV value
                self.ctrl_velocity_req.blockSignals(True)
                self.ctrl_velocity_req.setValue(int(csv_request_value))
                self.ctrl_velocity_req.blockSignals(False)
            elif mode == 2:  # TORQUE CONTROL
                self._ctrl_vals["torque_req"] = int(csv_request_value)
                # Update widget to reflect CSV value
                self.ctrl_torque_req.blockSignals(True)
                self.ctrl_torque_req.setValue(int(csv_request_value))
                self.ctrl_torque_req.blockSignals(False)
        else:
            # Update display label to show N/A or widget value
            if hasattr(self, 'csv_request_value_label'):
                if self.csv_sequence_started and self.csv_sequence_finished:
                    self.csv_request_value_label.setText("Finished")
                    self.csv_request_value_label.setStyleSheet("color: #ff6600; font-weight: bold; min-width: 80px;")
                elif self.csv_request_data is not None and not self.csv_sequence_started:
                    # CSV loaded but not started - show first value or N/A
                    self.csv_request_value_label.setText("Ready")
                    self.csv_request_value_label.setStyleSheet("color: #0066cc; font-weight: bold; min-width: 80px;")
                else:
                    # No CSV or not active - show widget value
                    mode = int(self.ctrl_mode.currentData())
                    if mode == 1:  # SPEED CONTROL
                        val = self.ctrl_velocity_req.value()
                    elif mode == 2:  # TORQUE CONTROL
                        val = self.ctrl_torque_req.value()
                    else:
                        val = "N/A"
                    self.csv_request_value_label.setText(str(val) if val != "N/A" else "N/A")
                    self.csv_request_value_label.setStyleSheet("color: #666; font-weight: normal; min-width: 80px;")
            
            # Use widget values
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
        
        # Apply CSV request if available and periodic send is active
        csv_request_value = self._get_csv_request_value()
        if csv_request_value is not None:
            # Update display label
            if hasattr(self, 'csv_request_value_label'):
                self.csv_request_value_label.setText(str(int(csv_request_value)))
                self.csv_request_value_label.setStyleSheet("color: #00aa00; font-weight: bold; min-width: 80px;")
            
            mode = int(self.ctrl_mode.currentData())
            if mode == 1:  # SPEED CONTROL
                self._ctrl_vals["velocity_req"] = int(csv_request_value)
            elif mode == 2:  # TORQUE CONTROL
                self._ctrl_vals["torque_req"] = int(csv_request_value)
        else:
            # Update display label to show N/A or widget value
            if hasattr(self, 'csv_request_value_label'):
                if self.csv_sequence_started and self.csv_sequence_finished:
                    self.csv_request_value_label.setText("Finished")
                    self.csv_request_value_label.setStyleSheet("color: #ff6600; font-weight: bold; min-width: 80px;")
                elif self.csv_request_data is not None and not self.csv_sequence_started:
                    # CSV loaded but not started - show first value or N/A
                    self.csv_request_value_label.setText("Ready")
                    self.csv_request_value_label.setStyleSheet("color: #0066cc; font-weight: bold; min-width: 80px;")
                else:
                    # No CSV or not active - show widget value
                    mode = int(self.ctrl_mode.currentData())
                    if mode == 1:  # SPEED CONTROL
                        val = self.ctrl_velocity_req.value()
                    elif mode == 2:  # TORQUE CONTROL
                        val = self.ctrl_torque_req.value()
                    else:
                        val = "N/A"
                    self.csv_request_value_label.setText(str(val) if val != "N/A" else "N/A")
                    self.csv_request_value_label.setStyleSheet("color: #666; font-weight: normal; min-width: 80px;")
            
            # Use widget values
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
        
        # ---- CSV request profile ----
        csv_layout = QtWidgets.QHBoxLayout()
        btn_load_csv = QtWidgets.QPushButton("Load Request CSV...")
        btn_load_csv.clicked.connect(self._load_request_csv)
        csv_layout.addWidget(btn_load_csv)
        btn_start_csv = QtWidgets.QPushButton("Start")
        btn_start_csv.clicked.connect(self._start_csv_sequence)
        btn_start_csv.setEnabled(False)  # Disabled until CSV is loaded
        self.csv_start_btn = btn_start_csv  # Store reference
        csv_layout.addWidget(btn_start_csv)
        # Checkbox to start logging with sequence
        self.csv_log_with_sequence_chk = QtWidgets.QCheckBox("Start log with sequence")
        csv_layout.addWidget(self.csv_log_with_sequence_chk)
        # Display for current request value
        csv_layout.addWidget(QtWidgets.QLabel("Request:"))
        self.csv_request_value_label = QtWidgets.QLabel("N/A")
        self.csv_request_value_label.setStyleSheet("color: #0066cc; font-weight: bold; min-width: 80px;")
        csv_layout.addWidget(self.csv_request_value_label)
        self.csv_request_label = QtWidgets.QLabel("No CSV loaded")
        self.csv_request_label.setStyleSheet("color: #666; font-style: italic;")
        csv_layout.addWidget(self.csv_request_label)
        csv_layout.addStretch()
        lv.addLayout(csv_layout)

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
        # Disable x-axis auto-range for plot2 since it's linked to plot1
        # Only y-axis auto-ranges independently
        self.vb2.enableAutoRange(x=False, y=True)
        
        # Link x-axis of plot2 to plot1 (so zooming/panning x-axis affects both)
        self.live_plot2.setXLink(self.live_plot1)

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
            # Only enable y-axis auto-range for plot2 (x-axis is linked to plot1)
            self.vb2.enableAutoRange(x=False, y=True)

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
            # Restrict mouse wheel zoom to x-axis only
            vb1.setMouseEnabled(x=True, y=False)
            vb2.setMouseEnabled(x=True, y=False)
        elif mode == "y":
            vb1.setMouseMode(pg.ViewBox.RectMode)
            vb2.setMouseMode(pg.ViewBox.RectMode)
            vb1.setAspectLocked(False)
            vb2.setAspectLocked(False)
            vb1.disableAutoRange(axis=pg.ViewBox.XAxis)
            vb2.disableAutoRange(axis=pg.ViewBox.XAxis)
            # Restrict mouse wheel zoom to y-axis only
            vb1.setMouseEnabled(x=False, y=True)
            vb2.setMouseEnabled(x=False, y=True)
        else:
            vb1.enableAutoRange(axis=pg.ViewBox.XYAxes)
            # Only enable y-axis auto-range for plot2 (x-axis is linked to plot1)
            vb2.enableAutoRange(axis=pg.ViewBox.YAxis)
            # Re-enable both axes for normal zoom
            vb1.setMouseEnabled(x=True, y=True)
            vb2.setMouseEnabled(x=True, y=True)

    def _zoom_fit(self):
        self.zoomx_btn.setChecked(False)
        self.zoomy_btn.setChecked(False)
        vb1 = self.live_plot1.getViewBox()
        vb2 = self.live_plot2.getViewBox()
        # Auto-range both axes for plot1
        vb1.enableAutoRange(axis=pg.ViewBox.XYAxes)
        vb1.autoRange()
        # Only auto-range y-axis for plot2 (x-axis is linked to plot1)
        vb2.enableAutoRange(axis=pg.ViewBox.YAxis)
        vb2.autoRange()
        # Re-enable both axes for normal zoom
        vb1.setMouseEnabled(x=True, y=True)
        vb2.setMouseEnabled(x=True, y=True)

    # ---------------- Graph tab ----------------
    def _build_graph_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # Top control bar
        ctrl = QtWidgets.QHBoxLayout()
        btn_load_csv = QtWidgets.QPushButton("Load CSV…")
        btn_load_csv.clicked.connect(self._pick_csv_into_graph)
        ctrl.addWidget(btn_load_csv)
        
        # CSV info display (filename and settings)
        self.csv_info_label = QtWidgets.QLabel("No CSV loaded")
        self.csv_info_label.setStyleSheet("color: #666; font-style: italic;")
        ctrl.addWidget(self.csv_info_label)
        
        self.linkx_chk = QtWidgets.QCheckBox("Link X"); self.linkx_chk.setChecked(True)
        ctrl.addWidget(self.linkx_chk)
        # Cursors enable + readout
        self.graph_cursors_chk = QtWidgets.QCheckBox("Cursors")
        self.graph_cursors_chk.setChecked(False)
        self.graph_cursors_chk.toggled.connect(self._on_graph_cursors_toggled)
        ctrl.addWidget(self.graph_cursors_chk)
        self.graph_cursors2_chk = QtWidgets.QCheckBox("Cursors P2")
        self.graph_cursors2_chk.setChecked(False)
        self.graph_cursors2_chk.toggled.connect(self._on_graph_cursors2_toggled)
        ctrl.addWidget(self.graph_cursors2_chk)
        # Cursor tracking dropdowns
        ctrl.addWidget(QtWidgets.QLabel("Track P1:"))
        self.cursor_curve1_combo = QtWidgets.QComboBox()
        self.cursor_curve1_combo.setMinimumWidth(140)
        self.cursor_curve1_combo.currentIndexChanged.connect(lambda _ix: self._refresh_cursor_readout())
        ctrl.addWidget(self.cursor_curve1_combo)
        ctrl.addWidget(QtWidgets.QLabel("Track P2:"))
        self.cursor_curve2_combo = QtWidgets.QComboBox()
        self.cursor_curve2_combo.setMinimumWidth(140)
        self.cursor_curve2_combo.currentIndexChanged.connect(lambda _ix: self._refresh_cursor_readout())
        ctrl.addWidget(self.cursor_curve2_combo)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        # Combined channel selection and zoom controls (will be populated when CSV is loaded)
        channel_selection_container = QtWidgets.QWidget()
        channel_selection_layout = QtWidgets.QHBoxLayout(channel_selection_container)
        
        # Window 1 channel selection with MultiSelectCombo
        win1_label = QtWidgets.QLabel("Plot1 vars:")
        channel_selection_layout.addWidget(win1_label)
        self.combo_csv_plot1 = MultiSelectCombo()
        self.combo_csv_plot1.checkedChanged.connect(self._on_csv_channels_changed)
        channel_selection_layout.addWidget(self.combo_csv_plot1)
        
        # Window 2 channel selection with MultiSelectCombo
        win2_label = QtWidgets.QLabel("Plot2 vars:")
        channel_selection_layout.addWidget(win2_label)
        self.combo_csv_plot2 = MultiSelectCombo()
        self.combo_csv_plot2.checkedChanged.connect(self._on_csv_channels_changed)
        channel_selection_layout.addWidget(self.combo_csv_plot2)
        
        channel_selection_layout.addSpacing(20)  # Add some spacing before controls
        
        # Downsampling dropdown
        downsample_label = QtWidgets.QLabel("Downsample:")
        channel_selection_layout.addWidget(downsample_label)
        self.csv_downsample_combo = QtWidgets.QComboBox()
        self.csv_downsample_combo.addItem("x1", 1)
        self.csv_downsample_combo.addItem("x2", 2)
        self.csv_downsample_combo.addItem("x4", 4)
        self.csv_downsample_combo.addItem("x8", 8)
        self.csv_downsample_combo.addItem("x16", 16)
        self.csv_downsample_combo.addItem("x32", 32)
        self.csv_downsample_combo.addItem("x64", 64)
        self.csv_downsample_combo.setCurrentIndex(0)  # Default to x1 (no downsampling)
        self.csv_downsample_combo.currentIndexChanged.connect(self._on_downsample_changed)
        channel_selection_layout.addWidget(self.csv_downsample_combo)
        
        channel_selection_layout.addSpacing(10)
        
        # Zoom buttons for GRAPH (CSV) plots - now in same row
        self.graph_zoomx_btn = QtWidgets.QPushButton("Zoom X")
        self.graph_zoomy_btn = QtWidgets.QPushButton("Zoom Y")
        self.graph_fit_btn   = QtWidgets.QPushButton("Fit View")
        for b in (self.graph_zoomx_btn, self.graph_zoomy_btn, self.graph_fit_btn):
            b.setCheckable(True)
            channel_selection_layout.addWidget(b)
        # FFT buttons for Plot1 and Plot2
        self.graph_fft_p1_btn = QtWidgets.QPushButton("FFT P1")
        self.graph_fft_p2_btn = QtWidgets.QPushButton("FFT P2")
        self.graph_fft_p1_btn.clicked.connect(self._on_fft_plot1)
        self.graph_fft_p2_btn.clicked.connect(self._on_fft_plot2)
        channel_selection_layout.addWidget(self.graph_fft_p1_btn)
        channel_selection_layout.addWidget(self.graph_fft_p2_btn)
        # Cursors readout next to Fit View
        self.graph_cursor_info = QtWidgets.QLabel("C1: x=–, y=–    C2: x=–, y=–    Δx=–, Δy=–")
        self.graph_cursor_info.setVisible(False)
        channel_selection_layout.addWidget(self.graph_cursor_info)
        
        # Loading progress bar (initially hidden)
        self.csv_loading_progress = QtWidgets.QProgressBar()
        self.csv_loading_progress.setFixedWidth(150)
        self.csv_loading_progress.setTextVisible(False)  # Hide percentage text
        self.csv_loading_progress.setFormat("")  # Clear format to avoid any text
        self.csv_loading_progress.setVisible(False)
        channel_selection_layout.addWidget(self.csv_loading_progress)
        
        channel_selection_layout.addStretch()
        
        # Initially hide channel selection (shown when CSV is loaded)
        channel_selection_container.setVisible(False)
        v.addWidget(channel_selection_container)
        
        # Store references
        self.channel_selection_container = channel_selection_container
        self.available_channels = []  # Will be populated when CSV is loaded
        self.current_csv_path = None
        self._cached_csv_data = None  # Cache CSV data to avoid re-reading
        self._cached_csv_path = None

        self.two_plot = TwoWindowPlot()
        # Pass progress bar reference to plot widget
        self.two_plot._progress_bar = self.csv_loading_progress
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
        # Wire cursor updates to label
        try:
            self.two_plot.cursorMoved.connect(self._update_cursor_info_label)
        except Exception:
            pass
        v.addWidget(self.two_plot)
        self.tabs.addTab(w, "Graph")

    def _on_graph_cursors_toggled(self, checked: bool):
        try:
            self.two_plot.set_cursors_enabled(bool(checked))
        except Exception:
            pass
        # Show/hide readout
        self.graph_cursor_info.setVisible(bool(checked))
        self.cursor_curve1_combo.setVisible(bool(checked))
        self.cursor_curve2_combo.setVisible(bool(checked))
        if checked:
            try:
                x1, y1, x2, y2, dx, dy = self.two_plot.get_cursor_values()
                self._update_cursor_info_label(x1, y1, x2, y2, dx, dy)
            except Exception:
                pass

    def _on_graph_cursors2_toggled(self, checked: bool):
        try:
            self.two_plot.set_cursors2_enabled(bool(checked))
        except Exception:
            pass

    def _update_cursor_info_label(self, x1: float, y1: float, x2: float, y2: float, dx: float, dy: float):
        # Use compact formatting
        def fmt(v):
            try:
                if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.001):
                    return f"{v:.3e}"
                return f"{v:.6f}".rstrip('0').rstrip('.') if '.' in f"{v:.6f}" else f"{v:.6f}"
            except Exception:
                return str(v)
        # Evaluate y from selected curves at cursor x positions
        y1_eval = y2_eval = None
        y3 = y4 = None
        try:
            name1 = self.cursor_curve1_combo.currentText().strip()
            if name1:
                y1_eval = self.two_plot.eval_curve_at_x(1, name1, x1)
                y2_eval = self.two_plot.eval_curve_at_x(1, name1, x2)
        except Exception:
            pass
        try:
            name2 = self.cursor_curve2_combo.currentText().strip()
            if name2:
                y3 = self.two_plot.eval_curve_at_x(2, name2, x1)
                y4 = self.two_plot.eval_curve_at_x(2, name2, x2)
        except Exception:
            pass
        # Fallbacks to horizontal values only if interpolation failed
        if y1_eval is None:
            y1_eval = y1
        if y2_eval is None:
            y2_eval = y2
        dy_eval = (y2_eval - y1_eval) if (y1_eval is not None and y2_eval is not None) else dy
        # Compose label
        if y3 is not None and y4 is not None:
            dy2 = y4 - y3
            text = f"C1: x={fmt(x1)}, y={fmt(y1_eval)}    C2: x={fmt(x2)}, y={fmt(y2_eval)}    P2: y3={fmt(y3)}, y4={fmt(y4)}, Δy2={fmt(dy2)}    Δx={fmt(dx)}, Δy={fmt(dy_eval)}"
        else:
            text = f"C1: x={fmt(x1)}, y={fmt(y1_eval)}    C2: x={fmt(x2)}, y={fmt(y2_eval)}    Δx={fmt(dx)}, Δy={fmt(dy_eval)}"
        self.graph_cursor_info.setText(text)

    def _refresh_cursor_readout(self):
        try:
            x1, y1, x2, y2, dx, dy = self.two_plot.get_cursor_values()
            self._update_cursor_info_label(x1, y1, x2, y2, dx, dy)
        except Exception:
            pass

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
            if not self.dlogger:
                QtCore.QTimer.singleShot(0, lambda: self._set_status("No logger instance"))
                return
            
            # Check if paths are set
            if not self.dlogger.bin_path:
                QtCore.QTimer.singleShot(0, lambda: self._set_status("BIN path not set"))
                return
            if not self.dlogger.csv_path:
                QtCore.QTimer.singleShot(0, lambda: self._set_status("CSV path not set"))
                return
            
            # Build CSV
            self.dlogger.build_csv()
            
            # Verify CSV was created
            if os.path.isfile(self.dlogger.csv_path):
                csv_size = os.path.getsize(self.dlogger.csv_path)
                QtCore.QTimer.singleShot(0, lambda: self._set_status(f"CSV built ✅: {os.path.basename(self.dlogger.csv_path)} ({csv_size} bytes)"))
            else:
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.critical(self, "Logger", f"CSV build completed but file not found:\n{self.dlogger.csv_path}"))
        except Exception as e:
            import traceback
            error_msg = f"Failed to build CSV:\n{e}\n\n{traceback.format_exc()}"
            print(f"CSV build error: {error_msg}")
            QtCore.QTimer.singleShot(
                0,
                lambda: QtWidgets.QMessageBox.critical(self, "Logger", error_msg)
            )

    def _pick_csv_into_graph(self):
        """Open file dialog with preview on single-click, load on double-click."""
        dialog = QtWidgets.QFileDialog(self, "Pick CSV", DATA_DIR, "CSV (*.csv)")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        
        # Track selected file for preview
        selected_file = [None]
        original_label_text = self.csv_info_label.text() if hasattr(self, 'csv_info_label') else "No CSV loaded"
        
        def on_file_selected(file_path):
            """Show settings preview when file is selected (single-click)."""
            if file_path and os.path.isfile(file_path):
                selected_file[0] = file_path
                # Parse settings
                settings_dict, monitor_extras = self._parse_csv_settings(file_path)
                # Update the CSV info label next to the load button
                self._update_csv_info_display(file_path, settings_dict, monitor_extras)
            else:
                # Reset to original if no file selected
                if hasattr(self, 'csv_info_label'):
                    self.csv_info_label.setText(original_label_text)
                    self.csv_info_label.setStyleSheet("color: #666; font-style: italic;")
        
        def on_dialog_finished(result):
            """Reset preview if dialog was cancelled."""
            if result != QtWidgets.QDialog.Accepted:
                if hasattr(self, 'csv_info_label'):
                    # Only reset if no CSV is currently loaded
                    if not hasattr(self, 'current_csv_path') or self.current_csv_path is None:
                        self.csv_info_label.setText(original_label_text)
                        self.csv_info_label.setStyleSheet("color: #666; font-style: italic;")
        
        # Connect file selection signal
        dialog.currentChanged.connect(on_file_selected)
        dialog.finished.connect(on_dialog_finished)
        
        # Show dialog
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                path = selected[0]
                self._load_csv_and_update_channels(path)
    
    def _format_settings_preview(self, csv_path: str, settings_dict: dict) -> str:
        """Format settings for preview display."""
        filename = os.path.basename(csv_path)
        
        # Extract important settings
        settings_to_show = [
            ("motor_type", "motor_type"),
            ("motor_connection", "motor_conn"),
            ("mode", "mode"),
            ("sensored_control", "sensored"),
            ("Fs_trq", "Fs_trq"),
            ("data_logger_ts_div", "ts_div"),
            ("cc_trise", "cc_trise"),
            ("spdc_trise", "spdc_trise")
        ]
        
        settings_lines = [f"<b>{filename}</b>", ""]
        for setting_name, display_name in settings_to_show:
            value = settings_dict.get(setting_name, "N/A")
            # Format enum values nicely
            if setting_name == "sensored_control":
                value = "ON" if value == "ENABLED" else "OFF" if value == "DISABLED" else value
            elif setting_name in ("motor_type", "motor_connection", "mode"):
                # These are enum values, show as-is
                pass
            elif value != "N/A":
                # Try to format decimal numbers to 4 digits
                try:
                    float_val = float(value)
                    # Check if it's a decimal number (has fractional part)
                    if float_val != int(float_val):
                        # Format to 4 significant digits
                        value = f"{float_val:.4g}"
                    else:
                        # Integer value, show as-is
                        value = str(int(float_val))
                except (ValueError, TypeError):
                    # Not a number, keep as-is
                    pass
            settings_lines.append(f"<b>{display_name}:</b> {value}")
        
        return "<br>".join(settings_lines)

    def _load_csv_and_update_channels(self, path: str):
        """Load CSV, extract available channels, and update UI with MultiSelectCombo."""
        try:
            # Only read header row to get column names (much faster for large files)
            if POLARS_AVAILABLE:
                try:
                    # Try Polars first (fastest)
                    print("DEBUG: Using Polars for CSV header reading")
                    df_header_polars = pl.read_csv(path, n_rows=0, ignore_errors=True, truncate_ragged_lines=True)
                    df_header = df_header_polars.to_pandas()  # Convert to pandas for compatibility
                except Exception as e:
                    # Fallback to pandas
                    print(f"DEBUG: Polars header read failed ({e}), falling back to Pandas")
                    try:
                        df_header = pd.read_csv(path, engine="c", nrows=0, on_bad_lines="skip")
                    except Exception:
                        df_header = pd.read_csv(path, engine="python", nrows=0, on_bad_lines="skip")
            else:
                print("DEBUG: Using Pandas for CSV header reading (Polars not available)")
                try:
                    # Try fast 'c' engine first (only reads header)
                    df_header = pd.read_csv(path, engine="c", nrows=0, on_bad_lines="skip")
                except Exception:
                    # Fallback to 'python' engine if 'c' fails
                    df_header = pd.read_csv(path, engine="python", nrows=0, on_bad_lines="skip")
            
            # Get channel columns (exclude metadata columns starting with "#" and unnamed columns)
            channel_cols = [col for col in df_header.columns 
                           if col and not str(col).startswith("#") 
                           and not str(col).strip() == ""
                           and not str(col).startswith("Unnamed")
                           and not str(col).startswith("_duplicated_")]
            
            if not channel_cols:
                QtWidgets.QMessageBox.warning(self, "CSV", "No channel columns found in CSV.")
                return
            
            # Parse settings from CSV
            settings_dict, monitor_extras = self._parse_csv_settings(path)
            
            # Update CSV info display
            self._update_csv_info_display(path, settings_dict, monitor_extras)
            # Store for later (e.g., time scaling)
            self.current_csv_settings = dict(settings_dict)
            
            self.available_channels = channel_cols
            self.current_csv_path = path
            # Clear cache when loading new file
            if hasattr(self, 'two_plot') and self.two_plot:
                self.two_plot._cached_csv_data = None
                self.two_plot._cached_csv_path = None
            
            # Update MultiSelectCombo widgets with available channels
            self.combo_csv_plot1.set_items(channel_cols)
            self.combo_csv_plot2.set_items(channel_cols)
            
            # Show the channel selection container
            self.channel_selection_container.setVisible(True)
            
            self._set_status(f"CSV loaded: {len(channel_cols)} channels available. Select channels to plot.")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV", f"Failed to load CSV:\n{e}")
    
    def _parse_csv_settings(self, csv_path: str):
        """Parse settings and monitor metadata from the CSV file.
        Returns (settings_dict, monitor_extras_dict).
        """
        settings_dict: dict[str, str] = {}
        monitor_extras: dict[str, str] = {}

        try:
            import pandas as pd

            # Read CSV to find the settings column (column P contains "#SETTINGS" in first row)
            df_header = pd.read_csv(csv_path, nrows=1, engine="python", on_bad_lines="skip", header=None)

            settings_name_col_idx = None
            for col_idx in range(len(df_header.columns)):
                first_val = str(df_header.iloc[0, col_idx]) if len(df_header) > 0 else ""
                if "#SETTINGS" in first_val:
                    settings_name_col_idx = col_idx
                    break

            if settings_name_col_idx is not None:
                settings_value_col_idx = settings_name_col_idx + 1
                #print(f"DEBUG: Found #SETTINGS at column index {settings_name_col_idx}, values at {settings_value_col_idx}")

                df_full = pd.read_csv(
                    csv_path,
                    engine="python",
                    on_bad_lines="skip",
                    header=None
                )
                #print(f"DEBUG: CSV has {len(df_full)} rows, {len(df_full.columns)} columns")

                if len(df_full) > 1 and settings_value_col_idx < len(df_full.columns):
                    for idx in range(1, len(df_full)):
                        if settings_name_col_idx >= len(df_full.columns):
                            break
                        name = str(df_full.iloc[idx, settings_name_col_idx]).strip()
                        value = str(df_full.iloc[idx, settings_value_col_idx]).strip()

                        if "#SETTINGS" in name:
                            continue
                        if name == "" or name.lower() == "nan":
                            #print(f"DEBUG: Stopping at row {idx} - empty name")
                            break
                        if name.startswith("#"):
                            #print(f"DEBUG: Stopping at row {idx} - hit section marker: {name}")
                            break
                        if value == "" or value.lower() == "nan":
                            continue

                        settings_dict[name] = value
                        #print(f"DEBUG: Parsed setting: {name} = {value}")
                else:
                    print("DEBUG: Settings columns out of range while parsing")
            else:
                print(f"DEBUG: Could not find #SETTINGS column. Columns: {len(df_header.columns)}")
        except Exception as e:
            print(f"Warning: Failed to parse settings from CSV: {e}")
            import traceback
            traceback.print_exc()

        # Extract monitor metadata using csv.reader (handles ragged rows)
        try:
            import csv
            with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    if any(cell.strip().upper() == "#MONITOR" for cell in row if cell):
                        #print(f"DEBUG: Raw #MONITOR row -> {row}")

                        header_row = next(reader, [])
                        while header_row and not any(cell.strip() for cell in header_row):
                            header_row = next(reader, [])

                        value_row = next(reader, [])
                        while value_row and not any(cell.strip() for cell in value_row):
                            value_row = next(reader, [])

                        # print(f"DEBUG: Monitor header row -> {header_row[:10]}")
                        # print(f"DEBUG: Monitor value row  -> {value_row[:10]}")

                        start_idx = 0
                        for idx, cell in enumerate(header_row):
                            if cell.strip():
                                start_idx = idx
                                break

                        for name, value in zip(header_row[start_idx:], value_row[start_idx:]):
                            name = name.strip()
                            if not name or name.startswith("#"):
                                continue
                            value = value.strip()
                            if not value:
                                continue
                            monitor_extras[name] = value
                        break
        except StopIteration:
            pass
        except Exception as monitor_err:
            print(f"DEBUG: Failed to parse #MONITOR block ({monitor_err})")

        # Extract LOG parameters (e.g., data_logger_ts_div) assuming same columns as #SETTINGS
        # and that #LOG appears after the end of the #SETTINGS section.
        try:
            found_log = False
            found_ts = False
            if settings_name_col_idx is not None:
                # name column and optional value column
                name_series = df_full.iloc[:, settings_name_col_idx].astype(str).str.strip()
                value_col_exists = settings_value_col_idx < df_full.shape[1]
                log_idx_arr = np.where(name_series.str.upper() == "#LOG")[0]
                found_log = len(log_idx_arr) > 0
                print(f"DEBUG: #LOG marker found (pandas search): {found_log}")
                if found_log:
                    r = int(log_idx_arr[0])
                    rr = r + 1
                    if rr < len(df_full):
                        name2 = str(df_full.iloc[rr, settings_name_col_idx]).strip()
                        val2 = str(df_full.iloc[rr, settings_value_col_idx]).strip() if value_col_exists else ""
                        if name2.lower() == "data_logger_ts_div" and val2:
                            settings_dict["data_logger_ts_div"] = val2
                            found_ts = True
                print(f"DEBUG: data_logger_ts_div under #LOG found: {found_ts}")
        except Exception as e:
            print(f"DEBUG: Failed to parse #LOG via pandas-column search ({e})")

        mode_val = monitor_extras.get("mode") or monitor_extras.get("Mode")
        vdc_val = monitor_extras.get("Vdc") or monitor_extras.get("VDC")
        #print(f"DEBUG: monitor mode extracted -> {mode_val}")
        #print(f"DEBUG: monitor Vdc extracted -> {vdc_val}")

        return settings_dict, monitor_extras
    
    def _update_csv_info_display(self, csv_path: str, settings_dict: dict, monitor_extras: dict | None = None):
        """Update the CSV info label with filename and important settings."""
        filename = os.path.basename(csv_path)
        
        # Extract important settings
        settings_to_show = [
            ("motor_type", "motor_type"),
            ("motor_connection", "motor_conn"),
            ("mode", "mode"),
            ("sensored_control", "sensored"),
            ("Fs_trq", "Fs_trq"),
            ("cc_trise", "cc_trise"),
            ("spdc_trise", "spdc_trise")
        ]
        
        settings_parts = []
        for setting_name, display_name in settings_to_show:
            value = settings_dict.get(setting_name, "N/A")
            # Format enum values nicely
            if setting_name == "sensored_control":
                value = "ON" if value == "ENABLED" else "OFF" if value == "DISABLED" else value
            elif setting_name in ("motor_type", "motor_connection", "mode"):
                # These are enum values, show as-is
                pass
            elif value != "N/A":
                # Try to format decimal numbers to 4 digits
                try:
                    float_val = float(value)
                    # Check if it's a decimal number (has fractional part)
                    if float_val != int(float_val):
                        # Format to 4 significant digits
                        value = f"{float_val:.4g}"
                    else:
                        # Integer value, show as-is
                        value = str(int(float_val))
                except (ValueError, TypeError):
                    # Not a number, keep as-is
                    pass
            settings_parts.append(f"{display_name}: {value}")
        
        info_parts = [filename]
        if settings_parts:
            info_parts.append(" | ".join(settings_parts))

        # Append monitor-derived extras (mode, Vdc) if available
        if monitor_extras:
            mode_value = monitor_extras.get("mode") or monitor_extras.get("Mode")
            vdc_value = monitor_extras.get("Vdc") or monitor_extras.get("VDC")

            if mode_value is not None:
                mode_str = str(mode_value)
                info_parts.append(mode_str)

            if vdc_value is not None:
                vdc_str = str(vdc_value)
                try:
                    vdc_float = float(vdc_str)
                    if vdc_float != int(vdc_float):
                        vdc_str = f"{vdc_float:.4g}"
                    else:
                        vdc_str = str(int(vdc_float))
                except Exception:
                    pass
                info_parts.append(f"Vdc: {vdc_str}")

        info_text = " | ".join(info_parts)
        #print(f"DEBUG: CSV info display -> {info_text}")
 
        if hasattr(self, 'csv_info_label'):
            self.csv_info_label.setText(info_text)
            self.csv_info_label.setStyleSheet("")  # Use default/natural text color

    def _on_csv_channels_changed(self):
        """Called when channel selection changes in CSV plot combos."""
        if self.current_csv_path:
            self._plot_csv_path(self.current_csv_path)
        # Update cursor tracking combos after channels change
        try:
            self._populate_cursor_track_combos()
        except Exception:
            pass


    def _plot_csv_path(self, path: str):
        """Plot CSV with current channel selections."""
        if not path:
            return
        
        # Get selected channels from MultiSelectCombo widgets
        channels1 = self.combo_csv_plot1.checked_items()
        channels2 = self.combo_csv_plot2.checked_items()
        
        # Check if at least one channel is selected
        if not channels1 and not channels2:
            self._set_status("Please select at least one channel to plot.")
            return
        
        # Convert selected channels to groups format (each channel is its own group)
        # This allows multiple channels to be plotted together
        g1 = [(ch,) for ch in channels1] if channels1 else []
        g2 = [(ch,) for ch in channels2] if channels2 else []
        
        # Generate titles automatically from selected channels
        t1 = channels1 if channels1 else None
        t2 = channels2 if channels2 else None

        # Get downsampling factor from dropdown
        downsample_factor = self.csv_downsample_combo.currentData()
        if downsample_factor is None:
            downsample_factor = 1
        
        # Debug: verify downsample_factor from dropdown
        #print(f"DEBUG: _plot_csv_path: dropdown currentData = {self.csv_downsample_combo.currentData()}, using downsample_factor = {downsample_factor}")

        linkx = self.linkx_chk.isChecked()
        # Apply time scaling if csv contains data_logger_ts_div and Fs_trq
        settings_dict = getattr(self, "current_csv_settings", {}) or {}
        def _safe_float(val):
            try:
                s = str(val).strip()
                if s == "" or s.lower() == "nan":
                    return None
                return float(s)
            except Exception:
                return None
        ts_div = _safe_float(settings_dict.get("data_logger_ts_div"))
        fs_trq = _safe_float(settings_dict.get("Fs_trq"))
        # Debug info
        print(f"DEBUG: ts_div(raw)={settings_dict.get('data_logger_ts_div')}, fs_trq(raw)={settings_dict.get('Fs_trq')} -> ts_div={ts_div}, fs_trq={fs_trq}")
        # Fallback: if missing, try parsing CSV settings again now
        if (ts_div is None or fs_trq is None) and path:
            try:
                re_settings, _ = self._parse_csv_settings(path)
                if re_settings:
                    self.current_csv_settings = dict(re_settings)
                    ts_div = _safe_float(re_settings.get("data_logger_ts_div")) if ts_div is None else ts_div
                    fs_trq = _safe_float(re_settings.get("Fs_trq")) if fs_trq is None else fs_trq
                    print(f"DEBUG: reparsed settings -> ts_div={ts_div}, fs_trq={fs_trq}")
            except Exception as _reparse_err:
                print(f"DEBUG: reparse failed: {_reparse_err}")
        if ts_div is not None and ts_div > 0 and fs_trq is not None and fs_trq > 0:
            sec_per_sample = ts_div / (fs_trq * 1000.0)
            self.two_plot.set_time_scale(sec_per_sample)
        else:
            # Fallback to samples
            self.two_plot.set_time_scale(1.0)
        try:
            self.two_plot.plot_csv(path, g1, g2, t1, t2, linkx, downsample_factor)
            self._set_status(f"Opened: {os.path.basename(path)} ✅")
            # Refresh cursor tracking dropdowns with plotted curve names
            self._populate_cursor_track_combos()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV", f"Failed to plot:\n{e}")

    def _on_downsample_changed(self):
        """Called when downsampling factor changes - reload CSV with new factor."""
        if self.current_csv_path:
            # Don't clear cache when switching to x1 - we need it for selective loading
            # Only clear if switching away from x1 to higher downsample
            downsample_factor = self.csv_downsample_combo.currentData()
            if downsample_factor and downsample_factor > 1:
                # Switching to higher downsample - clear cache
                self.two_plot._cached_csv_data = None
                self.two_plot._cached_csv_path = None
            # Reload and replot
            self._plot_csv_path(self.current_csv_path)

    def _populate_cursor_track_combos(self):
        try:
            names1 = self.two_plot.get_plot_curve_names(1)
            names2 = self.two_plot.get_plot_curve_names(2)
        except Exception:
            names1, names2 = [], []
        # Preserve selections if possible
        sel1 = self.cursor_curve1_combo.currentText() if hasattr(self, "cursor_curve1_combo") else ""
        sel2 = self.cursor_curve2_combo.currentText() if hasattr(self, "cursor_curve2_combo") else ""
        # Update items
        if hasattr(self, "cursor_curve1_combo"):
            self.cursor_curve1_combo.blockSignals(True)
            self.cursor_curve1_combo.clear()
            self.cursor_curve1_combo.addItems(names1)
            # Restore selection or default to first
            if sel1 and sel1 in names1:
                self.cursor_curve1_combo.setCurrentText(sel1)
            elif names1:
                self.cursor_curve1_combo.setCurrentIndex(0)
            self.cursor_curve1_combo.blockSignals(False)
        if hasattr(self, "cursor_curve2_combo"):
            self.cursor_curve2_combo.blockSignals(True)
            self.cursor_curve2_combo.clear()
            self.cursor_curve2_combo.addItems(names2)
            if sel2 and sel2 in names2:
                self.cursor_curve2_combo.setCurrentText(sel2)
            elif names2:
                self.cursor_curve2_combo.setCurrentIndex(0)
            self.cursor_curve2_combo.blockSignals(False)
        # Refresh label with new selection
        self._refresh_cursor_readout()

    def _on_fft_plot1(self):
        try:
            self._do_fft_for_plot(1)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "FFT", f"Failed to compute FFT for Plot1:\n{e}")

    def _on_fft_plot2(self):
        try:
            self._do_fft_for_plot(2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "FFT", f"Failed to compute FFT for Plot2:\n{e}")

    def _do_fft_for_plot(self, win_id: int):
        # Gather all currently displayed curves for this plot window
        shown = []
        try:
            for (wid, name), curve in list(self.two_plot._curves.items()):
                if wid != win_id:
                    continue
                try:
                    x = curve.xData
                    y = curve.yData
                except Exception:
                    x = None; y = None
                if x is None or y is None:
                    continue
                if len(x) < 2 or len(y) < 2:
                    continue
                shown.append((name, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
        except Exception:
            shown = []
        if not shown:
            QtWidgets.QMessageBox.information(self, "FFT", f"No displayed curves found in Plot{win_id}.")
            return
        # Compute common dt from the first curve's x axis
        x0 = shown[0][1]
        dx = np.diff(x0)
        dt_est = float(np.median(dx[np.isfinite(dx)])) if dx.size else None
        fs_fallback = (1.0 / dt_est) if (dt_est and dt_est > 0) else None
        # Clean and determine a common length across all signals
        cleaned = []
        min_len = 10**9
        for name, x, y in shown:
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]
            min_len = min(min_len, int(y.size))
            cleaned.append((name, x, y))
        if min_len < 4:
            QtWidgets.QMessageBox.information(self, "FFT", f"Not enough samples for FFT (need >= 4).")
            return
        # Prepare window and freq axis using the common length
        n = int(min_len)
        window = np.hanning(n)
        name_to_mag = {}
        name_to_xy = {}
        for name, _x, y in cleaned:
            yy = y[:n]
            yy = yy - np.nanmean(yy)
            yy *= window
            spec = np.fft.rfft(yy)
            cg = float(np.sum(window)) / float(n) if n > 0 else 1.0
            if not np.isfinite(cg) or cg <= 0:
                cg = 1.0
            mag = (2.0 / (n * cg)) * np.abs(spec)
            name_to_mag[name] = mag
        for name, x, y in cleaned:
            name_to_xy[name] = (x, y)
        title = f"FFT — P{win_id}"
        # Snapshot current visible x-range from the source plot
        try:
            vb = (self.two_plot.plot1 if win_id == 1 else self.two_plot.plot2).getViewBox()
            xr = vb.viewRange()[0]
            visible_range = (float(xr[0]), float(xr[1]))
        except Exception:
            visible_range = None
        self._show_multi_fft_window_dynamic(n, name_to_mag, fs_fallback, title, name_to_xy, visible_range)

    def _show_fft_window(self, freqs, mag, title: str):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        lay = QtWidgets.QVBoxLayout(dlg)
        pw = pg.PlotWidget()
        pw.showGrid(x=True, y=True)
        pw.setLabel('bottom', 'Frequency', units='Hz')
        pw.setLabel('left', 'Amplitude')
        pw.plot(freqs, mag, pen=pg.mkPen('#1e90ff', width=1.5))
        lay.addWidget(pw)
        dlg.resize(900, 450)
        dlg.show()
        # Keep a reference to prevent garbage collection
        self._fft_windows.append(dlg)

    def _show_multi_fft_window_dynamic(self, n: int, name_to_mag: dict, fs_fallback: float | None, title: str, name_to_xy: dict | None = None, visible_x_range: tuple | None = None):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        vbox = QtWidgets.QVBoxLayout(dlg)
        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        auto_chk = QtWidgets.QCheckBox("Auto")
        fs_label = QtWidgets.QLabel("Fs (Hz):")
        fs_spin = QtWidgets.QDoubleSpinBox()
        fs_spin.setDecimals(6)
        fs_spin.setRange(1e-9, 1e12)
        fs_spin.setSingleStep(1.0)
        fs_spin.setValue(1000.0)
        ctrl.addWidget(auto_chk)
        ctrl.addSpacing(10)
        ctrl.addWidget(fs_label)
        ctrl.addWidget(fs_spin)
        ctrl.addSpacing(10)
        use_vis_chk = QtWidgets.QCheckBox("Use visible window")
        use_vis_chk.setChecked(False)
        ctrl.addWidget(use_vis_chk)
        ctrl.addStretch(1)
        vbox.addLayout(ctrl)
        # Plot
        pw = pg.PlotWidget()
        pw.showGrid(x=True, y=True)
        pw.setLabel('bottom', 'Frequency', units='Hz')
        pw.setLabel('left', 'Amplitude')
        if pw.plotItem.legend is None:
            pw.addLegend()
        vbox.addWidget(pw)
        # Zoom + cursors controls
        ctrl2 = QtWidgets.QHBoxLayout()
        zoomx_btn = QtWidgets.QPushButton("Zoom X")
        zoomy_btn = QtWidgets.QPushButton("Zoom Y")
        fit_btn  = QtWidgets.QPushButton("Fit View")
        for b in (zoomx_btn, zoomy_btn, fit_btn):
            b.setCheckable(True)
            ctrl2.addWidget(b)
        ctrl2.addSpacing(12)
        cursors_chk = QtWidgets.QCheckBox("Cursors")
        ctrl2.addWidget(cursors_chk)
        ctrl2.addSpacing(8)
        ctrl2.addWidget(QtWidgets.QLabel("Track:"))
        track_combo = QtWidgets.QComboBox()
        track_combo.addItems(list(name_to_mag.keys()))
        if track_combo.count() > 0:
            track_combo.setCurrentIndex(0)
        ctrl2.addWidget(track_combo)
        ctrl2.addSpacing(12)
        ctrl2.addWidget(QtWidgets.QLabel("Magnitude:"))
        mag_mode_combo = QtWidgets.QComboBox()
        mag_mode_combo.addItems(["Absolute", "Log (dB)"])
        mag_mode_combo.setCurrentIndex(0)
        ctrl2.addWidget(mag_mode_combo)
        ctrl2.addSpacing(12)
        cursor_info = QtWidgets.QLabel("C1: f=–, A=–    C2: f=–, A=–    Δf=–, ΔA=–")
        cursor_info.setVisible(False)
        ctrl2.addWidget(cursor_info)
        ctrl2.addStretch(1)
        vbox.addLayout(ctrl2)
        # Crosshair cursors
        cursor_pen_1 = pg.mkPen((255, 200, 0), width=1)
        cursor_pen_2 = pg.mkPen((0, 200, 255), width=1)
        c1_v = pg.InfiniteLine(angle=90, movable=True, pen=cursor_pen_1)
        c1_h = pg.InfiniteLine(angle=0,  movable=True, pen=cursor_pen_1)
        c2_v = pg.InfiniteLine(angle=90, movable=True, pen=cursor_pen_2)
        c2_h = pg.InfiniteLine(angle=0,  movable=True, pen=cursor_pen_2)
        for ln in (c1_v, c1_h, c2_v, c2_h):
            try:
                ln.setZValue(1_000_001)
            except Exception:
                pass
            ln.hide()
            pw.addItem(ln)
        # Track evaluation helper
        def _eval_curve_at_x(freqs_arr, mags_arr, x_value):
            try:
                xv = float(x_value)
                if freqs_arr is None or mags_arr is None or len(freqs_arr) == 0:
                    return None
                x0 = float(freqs_arr[0]); xN = float(freqs_arr[-1])
                if xv <= x0:
                    return float(mags_arr[0])
                if xv >= xN:
                    return float(mags_arr[-1])
                return float(np.interp(xv, np.asarray(freqs_arr, dtype=float), np.asarray(mags_arr, dtype=float)))
            except Exception:
                return None
        # Zoom helpers
        def _set_zoom_mode(mode: str):
            vb = pw.getViewBox()
            xr, yr = vb.viewRange()
            if mode == "x":
                zoomx_btn.setChecked(True); zoomy_btn.setChecked(False); fit_btn.setChecked(False)
                vb.setMouseMode(pg.ViewBox.RectMode)
                vb.disableAutoRange()
                vb.setXRange(xr[0], xr[1], padding=0)
                vb.setYRange(yr[0], yr[1], padding=0)
                vb.setMouseEnabled(x=True, y=False)
            elif mode == "y":
                zoomx_btn.setChecked(False); zoomy_btn.setChecked(True); fit_btn.setChecked(False)
                vb.setMouseMode(pg.ViewBox.RectMode)
                vb.disableAutoRange()
                vb.setXRange(xr[0], xr[1], padding=0)
                vb.setYRange(yr[0], yr[1], padding=0)
                vb.setMouseEnabled(x=False, y=True)
            else:
                zoomx_btn.setChecked(False); zoomy_btn.setChecked(False); fit_btn.setChecked(False)
                vb.enableAutoRange(axis=pg.ViewBox.XYAxes)
                vb.autoRange()
                vb.setMouseEnabled(x=True, y=True)
        zoomx_btn.clicked.connect(lambda: _set_zoom_mode("x"))
        zoomy_btn.clicked.connect(lambda: _set_zoom_mode("y"))
        fit_btn.clicked.connect(lambda: _set_zoom_mode("none"))
        # Create curve objects
        name_to_curve = {}
        for name, mag in name_to_mag.items():
            try:
                pen = self.two_plot._pen_for(name)
            except Exception:
                pen = pg.mkPen('#1e90ff', width=1.5)
            c = pw.plot([], [], name=name, pen=pen)
            name_to_curve[name] = c
        # Magnitude transform helper
        def _transform_mag(arr):
            mode = mag_mode_combo.currentIndex()
            if mode == 0:
                return arr
            # Log magnitude in dB
            return 20.0 * np.log10(np.maximum(np.asarray(arr, dtype=np.float64), 1e-15))
        # Active spectrum cache (may switch between full and visible window)
        active_mag = [dict(name_to_mag)]
        current_n = [int(n)]
        # Cursor toggle
        def _on_cursors_toggled(checked: bool):
            for ln in (c1_v, c1_h, c2_v, c2_h):
                ln.setVisible(bool(checked))
            cursor_info.setVisible(bool(checked))
            if checked:
                try:
                    vb = pw.getViewBox()
                    xr, yr = vb.viewRange()
                    x1 = xr[0] + (xr[1] - xr[0]) * (1.0 / 3.0)
                    x2 = xr[0] + (xr[1] - xr[0]) * (2.0 / 3.0)
                    y1 = yr[0] + (yr[1] - yr[0]) * 0.5
                    y2 = y1
                except Exception:
                    x1, x2, y1, y2 = 0.0, 1.0, 0.0, 0.0
                c1_v.setPos(x1); c2_v.setPos(x2); c1_h.setPos(y1); c2_h.setPos(y2)
                _refresh_cursor_readout()
        cursors_chk.toggled.connect(_on_cursors_toggled)
        # Compute auto Fs availability and value
        settings_dict = getattr(self, "current_csv_settings", {}) or {}
        def _safe_float(val):
            try:
                s = str(val).strip()
                if s == "" or s.lower() == "nan":
                    return None
                return float(s)
            except Exception:
                return None
        ts_div = _safe_float(settings_dict.get("data_logger_ts_div"))
        fs_trq = _safe_float(settings_dict.get("Fs_trq"))
        downsample_ratio = None
        try:
            downsample_ratio = self.csv_downsample_combo.currentData()
        except Exception:
            downsample_ratio = None
        if not downsample_ratio or downsample_ratio <= 0:
            downsample_ratio = 1
        auto_available = bool(ts_div and ts_div > 0 and fs_trq and fs_trq > 0)
        auto_fs = (fs_trq*1000 / (ts_div * downsample_ratio)) if auto_available else None
        # Default states
        if auto_available:
            auto_chk.setChecked(True)
            fs_spin.setValue(float(auto_fs))
        else:
            auto_chk.setChecked(False)
            if fs_fallback and fs_fallback > 0:
                fs_spin.setValue(float(fs_fallback))
        # Updater
        current_freqs = [None]  # boxed for closure
        def update_plot_from_fs():
            fs = None
            if auto_chk.isChecked():
                if not auto_available:
                    QtWidgets.QMessageBox.information(dlg, "FFT", "Auto Fs is not available from CSV; please enter Fs manually.")
                    auto_chk.blockSignals(True)
                    auto_chk.setChecked(False)
                    auto_chk.blockSignals(False)
                else:
                    fs_spin.blockSignals(True)
                    fs_spin.setValue(float(auto_fs))
                    fs_spin.blockSignals(False)
                    fs = float(auto_fs)
            if fs is None:
                fs = float(fs_spin.value())
            if fs <= 0:
                return
            freqs = np.fft.rfftfreq(current_n[0], d=1.0/float(fs))
            current_freqs[0] = freqs
            for name, curve in name_to_curve.items():
                mag = active_mag[0].get(name)
                if mag is None or len(mag) == 0:
                    curve.setData([], [])
                else:
                    curve.setData(freqs, _transform_mag(mag))
            _refresh_cursor_readout()
        auto_chk.toggled.connect(update_plot_from_fs)
        fs_spin.valueChanged.connect(update_plot_from_fs)
        # Update magnitude mode
        def update_mag_mode():
            # Update y-axis label
            if mag_mode_combo.currentIndex() == 0:
                pw.setLabel('left', 'Amplitude')
            else:
                pw.setLabel('left', 'Magnitude (dB)')
            # Re-apply current data with transform
            freqs_arr = current_freqs[0]
            if freqs_arr is None:
                return
            for name, curve in name_to_curve.items():
                mag = active_mag[0].get(name)
                if mag is None or len(mag) == 0:
                    curve.setData([], [])
                else:
                    curve.setData(freqs_arr, _transform_mag(mag))
            _refresh_cursor_readout()
        mag_mode_combo.currentIndexChanged.connect(lambda _ix: update_mag_mode())
        # Visible window recompute
        def recompute_visible_if_needed():
            if not use_vis_chk.isChecked() or not name_to_xy or not visible_x_range:
                active_mag[0] = dict(name_to_mag)
                current_n[0] = int(n)
                update_plot_from_fs()
                return
            xmin, xmax = visible_x_range
            # Build per-signal segment arrays within [xmin, xmax]
            segs = {}
            min_len_local = 10**9
            for name, (x, y) in name_to_xy.items():
                try:
                    mask = (x >= xmin) & (x <= xmax) & np.isfinite(x) & np.isfinite(y)
                    yy = y[mask]
                except Exception:
                    yy = np.array([], dtype=float)
                min_len_local = min(min_len_local, int(yy.size))
                segs[name] = yy
            if min_len_local < 4:
                # Not enough points; clear spectra
                active_mag[0] = {k: np.array([], dtype=float) for k in name_to_curve.keys()}
                current_n[0] = 4
                update_plot_from_fs()
                return
            nn = int(min_len_local)
            current_n[0] = nn
            win = np.hanning(nn)
            new_mag = {}
            for name, yy in segs.items():
                ycut = yy[:nn]
                ycut = ycut - np.nanmean(ycut)
                ycut *= win
                spec = np.fft.rfft(ycut)
                cg = float(np.sum(win)) / float(nn) if nn > 0 else 1.0
                if not np.isfinite(cg) or cg <= 0:
                    cg = 1.0
                new_mag[name] = (2.0 / (nn * cg)) * np.abs(spec)
            active_mag[0] = new_mag
            update_plot_from_fs()
        use_vis_chk.toggled.connect(recompute_visible_if_needed)
        # Cursor readout
        def _refresh_cursor_readout():
            if not cursor_info.isVisible():
                return
            freqs_arr = current_freqs[0]
            if freqs_arr is None or freqs_arr.size == 0:
                cursor_info.setText("C1: f=–, A=–    C2: f=–, A=–    Δf=–, ΔA=–")
                return
            try:
                x1 = float(c1_v.value()); x2 = float(c2_v.value())
                y1 = float(c1_h.value()); y2 = float(c2_h.value())
            except Exception:
                x1 = x2 = y1 = y2 = 0.0
            name = track_combo.currentText().strip()
            mag_arr = active_mag[0].get(name)
            if mag_arr is None:
                y1_eval = y2_eval = float('nan')
            else:
                transformed = _transform_mag(mag_arr)
                y1_eval = _eval_curve_at_x(freqs_arr, transformed, x1)
                y2_eval = _eval_curve_at_x(freqs_arr, transformed, x2)
            def fmt(v):
                try:
                    if v is None or not np.isfinite(v):
                        return "–"
                    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.001):
                        return f"{v:.3e}"
                    return f"{v:.6g}"
                except Exception:
                    return "–"
            df = x2 - x1
            dy = (y2_eval - y1_eval) if (y1_eval is not None and y2_eval is not None and np.isfinite(y1_eval) and np.isfinite(y2_eval)) else float('nan')
            cursor_info.setText(f"C1: f={fmt(x1)}, A={fmt(y1_eval)}    C2: f={fmt(x2)}, A={fmt(y2_eval)}    Δf={fmt(df)}, ΔA={fmt(dy)}")
        try:
            c1_v.sigPositionChanged.connect(_refresh_cursor_readout)
            c1_h.sigPositionChanged.connect(_refresh_cursor_readout)
            c2_v.sigPositionChanged.connect(_refresh_cursor_readout)
            c2_h.sigPositionChanged.connect(_refresh_cursor_readout)
        except Exception:
            pass
        track_combo.currentIndexChanged.connect(lambda _ix: _refresh_cursor_readout())
        # Initial render
        # Initial render
        recompute_visible_if_needed()
        update_mag_mode()
        dlg.resize(1000, 520)
        dlg.show()
        self._fft_windows.append(dlg)


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

        # Store latest values for deferred updates
        self._pending_diag_values = dict(values)
        self._last_diag_values = dict(values)
        
        # Throttle live plot updates separately (they can handle higher rates)
        self._pending_live_plot_values = dict(values)
        if not self._live_plot_timer.isActive():
            # Update plots immediately and start timer
            self._apply_pending_live_plot_update()
            self._live_plot_timer.start(self._live_plot_update_interval_ms)
        
        # Throttle UI updates: if timer not running, start it; otherwise just update pending values
        if not self._diag_update_timer.isActive():
            # Apply immediately and start timer for next update
            self._apply_pending_diag_update()
            self._diag_update_timer.start(self._diag_update_interval_ms)

    def _apply_pending_diag_update(self):
        """Apply pending DIAG status updates to UI (called by throttled timer)."""
        if self._pending_diag_values is None:
            return
        
        values = self._pending_diag_values
        
        # Disable widget updates during bulk changes for better performance
        widgets_to_update = []
        if hasattr(self, 'hl_vals'):
            widgets_to_update.extend(self.hl_vals.values())
        if hasattr(self, 'mon_num_vars'):
            widgets_to_update.extend(self.mon_num_vars.values())
        if hasattr(self, 'status_text1'):
            widgets_to_update.append(self.status_text1)
        if hasattr(self, 'status_text2'):
            widgets_to_update.append(self.status_text2)
        
        for w in widgets_to_update:
            w.setUpdatesEnabled(False)
        
        try:
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

            # Decoded text split into two columns (only update if changed)
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
            text1 = "\n".join(lines[:half]) if lines else ""
            text2 = "\n".join(lines[half:]) if lines else ""
            
            # Only update status text if it changed
            if (text1, text2) != self._last_status_text:
                self.status_text1.setPlainText(text1)
                self.status_text2.setPlainText(text2)
                self._last_status_text = (text1, text2)

            # --- Update limiter flag indicators ---
            lf_val = int(values.get("limiter_flags", 0))
            for bit, name in LIMITER_FLAG_BITS:
                active = bool(lf_val & (1 << bit))
                if name in self.limiter_bits:
                    self.limiter_bits[name].setActive(active)
        finally:
            # Re-enable updates and trigger repaint
            for w in widgets_to_update:
                w.setUpdatesEnabled(True)

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

    def _apply_pending_live_plot_update(self):
        """Apply pending live plot updates (called by throttled timer)."""
        if self._pending_live_plot_values is None:
            return
        self._update_live_plots(self._pending_live_plot_values)

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

        # Only update curve management if selection changed (expensive operation)
        all_names = set(names1 + names2)
        existing_curves = set(list(self.live_curves1.keys()) + list(self.live_curves2.keys()))
        if all_names != existing_curves:
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

        # Append new values to buffers
        for n in names1 + names2:
            if n in values:
                self.live_buf[n].append(values[n])

        # Update plot data (only if we have data)
        if len(self.live_time) > 2:
            x = np.fromiter(self.live_time, dtype=float)
            for n, curve in self.live_curves1.items():
                if n in self.live_buf and len(self.live_buf[n]) > 2:
                    y = np.fromiter(self.live_buf[n], dtype=float)
                    curve.setData(x[-len(y):], y)
            for n, curve in self.live_curves2.items():
                if n in self.live_buf and len(self.live_buf[n]) > 2:
                    y = np.fromiter(self.live_buf[n], dtype=float)
                    curve.setData(x[-len(y):], y)

            # Auto-range (only if we have curves)
            if self.live_curves1 or self.live_curves2:
                self.vb1.enableAutoRange(x=True, y=True)
                # Only enable y-axis auto-range for plot2 (x-axis is linked to plot1)
                self.vb2.enableAutoRange(x=False, y=True)

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