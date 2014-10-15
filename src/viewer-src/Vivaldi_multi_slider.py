from PyQt4 import QtGui, QtCore


class multi_slider(QtGui.QWidget):
	slider_dict = {}
	slider_opacity_dict = {}
	num_of_slider = 5
	def __init__(self, parent):
		super(multi_slider, self).__init__(parent)

		MainBox = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
		for elem in range(self.num_of_slider):
			tmpBox = QtGui.QBoxLayout(QtGui.QBoxLayout.LeftToRight)
			sld1 = QtGui.QSlider(QtCore.Qt.Horizontal,self)
			sld1.setFixedSize(150,25)
			tmpBox.addWidget(sld1)
			sld2 = QtGui.QSlider(QtCore.Qt.Horizontal,self)
			sld2.setFixedSize(150,25)
			tmpBox.addWidget(sld2)

			MainBox.addLayout(tmpBox)
			self.slider_dict[elem] = sld1 
			self.slider_opacity_dict[elem] = sld2

		self.setLayout(MainBox)
	
		#self.setGeometry(300,300,400,100)
		#self.setWindowTitle('Cotrol Panel')
		#self.show()

#if __name__ == '__main__' :
	#import sys
	#app = QtGui.QApplication(sys.argv)
	#slider = multi_slider()
	#sys.exit(app.exec_())		
