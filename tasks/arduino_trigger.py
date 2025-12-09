import time
import pyfirmata2
from serial.tools import list_ports


def find_arduino_port():
    """
    Scans all serial ports and returns the port of a SparkFun/Arduino device.
    Returns None if nothing is found.
    """
    ports = list_ports.comports()

    for port in ports:
        desc = port.description.lower()

        # You can extend this keyword list anytime
        if ("SparkFun Pro Micro" in desc or
            "sparkfun" in desc or
            "arduino" in desc or
            "usb serial" in desc):

            print(f"[INFO] Found Arduino/SparkFun device on: {port.device}")
            return port.device

    print("[ERROR] No Arduino/SparkFun device found.")

    return None



def init_board(
    # PORT = 'COM6',          # Deinen Port hier setzen  --> TODO should be automated
    PIN = 9,                # D9 als TTL-Ausgang!  is default in triggerbox hardware
):
    """
    COM-PORT can vary per device and sessions,
    PIN: is default hardware config in board

    returns initiated arduino pin and board
    - pin: to send triggers
    - board: to close board at end
    """

    # try:
    #     board = pyfirmata2.Arduino(PORT)
    # except:
    #     raise ValueError('COM PORT incorrect? --> check "DeviceManger" or "GeräteManager"')
    
    port = find_arduino_port()

    if port is None:
        print("[FATAL ARDUINO] Cannot continue without a valid port.")

    # Connect using pyfirmata2
    print("[INFO] Connecting to arduino...")
    board = pyfirmata2.Arduino(port)
    print("[INFO] Arduino-connection established.")

    print("Board connected:", board)

    # initialise PWM-Pin
    pin = board.get_pin(f'd:{PIN}:p')

    # set pin on low
    pin.write(0) #--- output signal on digital IO Pin9 is low/"0" now


    return pin, board


def send_trigger(pin, TRIG_type, TRIG_version='v1',):
    """
    Signal output levels for the BNC-TriggerBox
    is the positiv logic:
    - "0" stands for 0.01 V or low voltage,
    - "1"  stand for 4.64 V or high voltage.

    overview of various triggers, every value is the duration
    of a pulse that will be send in seconds, the amount of
    durations represent the amount of triggers.
    for example: [.1, .3, .1] is a trigger-train of 3 triggers
    """

    # TODO: convert into JSON
    TRIGGER_SCHEME = {
        'v1': {
            'go': [.1, .05],
            'nogo': [.1, .15],
            'abort': [.1, .3],
            'LOW_pause': .05,
            'rest': [0.5,]
        }
    }

    TRIGGERS = TRIGGER_SCHEME[TRIG_version]

    print(f"\n### send trigger '{TRIG_type}")

    for TRIG_dur in TRIGGERS[TRIG_type]:
        # set output signal on digital IO Pin9 high/"1"
        pin.write(1.0)
        # keep signal high for n-sec duration
        time.sleep(TRIG_dur) # delay between rising and falling edge, make the time for sleep "(0.5)" a variable to change the pulse width then

        # after waiting: set signal to low again
        pin.write(0) #--- output signal on digital IO Pin9 is low/"0" now
        # keep signal low for default pause
        time.sleep(TRIGGERS['LOW_pause']) # delay between two consecutive triggers


def close_board(pin, board):

    # close board at end of task
    pin.write(0)      # Sicherheit: LOW
    board.exit()      # Verbindung sauber schließen

    print('\n\n#### Arduino Board formally closed ####\n')