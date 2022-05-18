import sys
from PyQt5 import QtWidgets

def arayuz():
    nesne=QtWidgets.QApplication(sys.argv)
    pencere= QtWidgets.QWidget()
    pencere.setWindowTitle("Deneme")
    pencere.setGeometry(250,100,600,300)

    """etiket=QtWidgets.QLabel(pencere)
    etiket.setText("Kamera Kontrol Paneli")
    etiket.move(300,150)"""

    """satir= QtWidgets.QLineEdit(pencere)
    satir.setText("2.4")
    satir.setReadOnly(True)"""

    buton=QtWidgets.QPushButton(pencere)
    buton.setText("Kaydet")

    pencere.show()
    nesne.exec_()

arayuz()