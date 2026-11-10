"""
อ่านสัญญาณ EOG จาก OpenBCI Ganglion แล้วแสดงค่าใน Terminal
ดัดแปลงจาก ganglion_sender(3).py:
- ใช้ BrainFlow และ GANGLION_NATIVE_BOARD เหมือนโค้ดต้นฉบับ
- เลือก Channel 1 (+1) สำหรับ Active electrode ที่หน้าผาก
- แสดงค่าเป็น microvolts (uV) โดยไม่คูณ SCALE_FACTOR
- ไม่ส่ง UDP และไม่บันทึกไฟล์
หยุดโปรแกรมด้วย Ctrl+C
"""

import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ---------------- CONFIGURATION ----------------
SERIAL_PORT = "COM3"       # คงค่าไว้ตามไฟล์ต้นฉบับ
BOARD_NAME = ""            # ปล่อยว่าง หากไม่จำเป็นต้องระบุชื่อบอร์ด
PRINT_INTERVAL = 0.10      # แสดงผลทุก 0.1 วินาที (ประมาณ 10 ครั้ง/วินาที)
DURATION_SECONDS = 0       # 0 = ทำงานต่อเนื่องจนกด Ctrl+C
# ------------------------------------------------


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    if BOARD_NAME:
        params.other_info = BOARD_NAME

    # ใช้ board ID แบบเดียวกับสคริปต์ที่แนบมา
    board_id = BoardIds.GANGLION_NATIVE_BOARD
    board = BoardShim(board_id, params)

    is_streaming = False
    start_time = None

    try:
        # BrainFlow row index ของ EEG channel แรกบน Ganglion คือ 1
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eog_channel = eeg_channels[0]  # Channel 1 / input +1

        sampling_rate = BoardShim.get_sampling_rate(board_id)

        print("=" * 58)
        print(" OpenBCI Ganglion - EOG Terminal Monitor")
        print("=" * 58)
        print(f"Board ID       : {board_id}")
        print(f"Serial setting : {SERIAL_PORT}")
        print(f"EOG channel    : Channel 1 (BrainFlow row {eog_channel})")
        print(f"Sampling rate  : {sampling_rate} Hz")
        print("Unit           : microvolts (uV)")
        print("-" * 58)
        print("กำลังเตรียมการเชื่อมต่อบอร์ด...")
        print("หากเชื่อมต่อสำเร็จ ค่าจะแสดงด้านล่าง")
        print("กด Ctrl+C เพื่อหยุดโปรแกรม\n")

        board.prepare_session()
        board.start_stream()
        is_streaming = True
        start_time = time.time()

        print("เชื่อมต่อและเริ่มรับข้อมูลแล้ว")
        print("เวลา (s)       EOG (uV)       จำนวน samples")
        print("-" * 58)

        last_print = 0.0
        total_samples = 0

        while True:
            # get_board_data() ดึงข้อมูลที่สะสมอยู่และนำออกจาก buffer
            data = board.get_board_data()

            if data.shape[1] > 0:
                total_samples += data.shape[1]
                latest_uv = float(data[eog_channel, -1])
                elapsed = time.time() - start_time

                now = time.time()
                if now - last_print >= PRINT_INTERVAL:
                    print(
                        f"{elapsed:8.2f}       "
                        f"{latest_uv:10.3f}       "
                        f"{data.shape[1]:5d}",
                        flush=True
                    )
                    last_print = now

            if DURATION_SECONDS > 0 and (time.time() - start_time) >= DURATION_SECONDS:
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nได้รับคำสั่งหยุดจาก Ctrl+C")

    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}")
        print("\nตรวจสอบว่า:")
        print("1. Ganglion เปิดอยู่และแบตเตอรี่เพียงพอ")
        print("2. ไม่มีโปรแกรมอื่นกำลังเชื่อมต่อบอร์ดอยู่")
        print("3. การตั้งค่า Bluetooth / Dongle ตรงกับโหมดเชื่อมต่อของสคริปต์")

    finally:
        print("\nกำลังปิดการเชื่อมต่อ...")

        try:
            if is_streaming:
                board.stop_stream()
        except Exception:
            pass

        try:
            if board.is_prepared():
                board.release_session()
        except Exception:
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

        print("ปิดการเชื่อมต่อเรียบร้อย")


if __name__ == "__main__":
    main()
