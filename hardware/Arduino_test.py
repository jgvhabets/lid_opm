import pyfirmata
from pyfirmata import util
import time

#i = 0
board = pyfirmata.Arduino('COM5')

#it = pyfirmata.util.Iterator(board)
#it.start()
#led = board.get_pin("d:13:p")

while True:
    board.digital[13].write(1)
    time.sleep(1)
    board.digital[13].write(0)
    time.sleep(1)
# mal checken von raiks testprogramm ob ich die lampe (pin 13?) zum leuchten kriege.
# muss schauen, wie das dann läuft, dass sich die befehle und der task nicht in die quere kommen

# kann mit time auch schonmal einfach schauen ob das funktioniert indem ich schaue, dass die lampe leuchtet

