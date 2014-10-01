from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
from OpenGL.GL import *
import numpy 
import Image


class TFN_window(QtGui.QMainWindow):
	mouse_state = 0
	mouse_flag = 0
	prev_x, prev_y = 0, 0
	TF_bandwidth = 1

	data_cover = None

	current_color = (0,0,0)
	def __init__(self, parent=None, cnt=0):
		super(TFN_window, self).__init__()
	
		
		if parent.TFF==None:
			self.widget = TFN_widget(cnt, parent)
			self.setGeometry(parent.window.widget.width,0, 600, 200)

		else:
			self.widget = TFN_widget2(cnt,parent)
			self.setGeometry(parent.window.widget.width,300, 600, 200)
		self.setFixedSize(600,200)
		self.setCentralWidget(self.widget)
		self.setWindowTitle("TransferFunction")
	

		self.parent = parent
		self.widget.pixmap = QtGui.QPixmap(10,10)

	def set_app(self, app):
		self.app = app

	def set_cover(self):
		self.TF_bandwidth = 3220
		pass
		

	def keyPressEvent(self, event):
		if type(event) == QtGui.QKeyEvent:
			if event.key() == QtCore.Qt.Key_Escape:
				self.parent.TFF = None
				self.close()
			elif event.key() == QtCore.Qt.Key_S:
				self.parent.window.save_mvmtx_tf()
			elif event.key() == QtCore.Qt.Key_L:
				file_name = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
				self.parent.window.load_mvmtx_tf(file_name)

	def mousePressEvent(self, event):
		self.mouse_state = event.button()
		tmp_x = int(event.x() * 255 / 600)
		tmp_y = int((200-event.y()) * 255 / 200)

		if event.button() == 1:
			 #set alpha value	
			self.widget.transfer_alpha[tmp_x] = tmp_y;
			self.prev_x, self.prev_y = tmp_x, tmp_y
		
		elif event.button() == 2:
			self.prev_x = tmp_x
			self.widget.setColor(tmp_x, self.current_color)
			

		elif event.button() == 4:
			self.col = QtGui.QColorDialog.getColor()
			self.current_color = (self.col.red(),self.col.green(), self.col.blue())
			self.widget.pixmap.fill(self.col)
			
		
		self.widget.updateGL()
	
	def mouseMoveEvent(self, event):
		if self.mouse_state == 1:
			tmp_x = int(event.x() * 255 / 600)
			tmp_y = int((200-event.y()) * 255 / 200)
	
			if tmp_x <= 0: tmp_x = 0
			elif tmp_x >= 255: tmp_x = 255
			if tmp_y <= 0: 
				tmp_y = 0
				self.prev_y = 0
			elif tmp_y >= 255: 
				tmp_y = 255
				self.prev_y = 255
			diff = 1
			if self.prev_x > tmp_x:
				diff = -1

			if self.prev_x != tmp_x:
				slope = (tmp_y - self.prev_y) / (tmp_x - self.prev_x) 

				for elem in range(self.prev_x, tmp_x, diff):
					self.widget.transfer_alpha[elem] = self.prev_y + slope * (elem - self.prev_x)
					#self.widget.updateGL()

			self.widget.transfer_alpha[tmp_x] = tmp_y
			self.widget.updateGL()
			self.prev_x, self.prev_y = tmp_x, tmp_y

		elif self.mouse_state == 2:
			pass
					

	def mouseReleaseEvent(self, event):
		self.widget.updateGL()
		self.mouse_state = 0
		self.mouse_flag = 0
		self.prev_x = 0
		self.prev_y = 0


class TFN_widget(QGLWidget):
	color_texId = 1
	transfer_function = numpy.zeros(256*4, dtype=numpy.uint8)
	for elem in range(256):
		transfer_function[4*elem + 3] = elem
	color_list = {0: (0,0,0), 255: (0,0,0)}
	transfer_alpha = numpy.zeros(256, dtype = numpy.uint8)

	for elem in range(256):
		transfer_alpha[elem] = elem

	def __init__(self, cnt, parent):
		super(TFN_widget, self).__init__()
		self.color_texId += cnt
		self.updated = 0
		self.parent= parent
		self.pixmap = None
	
	def enterEvent(self, e):
		self.parent.app.setOverrideCursor(QtGui.QCursor(self.pixmap))

	def leaveEvent(self, e):
		self.parent.app.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

	def initTexture(self):

		glGenTextures(1, self.color_texId)
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
	
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.transfer_function)
	
		glBindTexture(GL_TEXTURE_2D, 0)

	def setColor(self, x, col):
		self.color_list[x] = col


		print list(self.color_list).sort()
		elem_prev = (0,self.color_list[0])
		clist = list(self.color_list)
		clist.sort()
		for elem1 in clist:
			elem = (elem1, self.color_list[elem1])
			diff = float(elem[0] - elem_prev[0])
			for i in range(elem_prev[0], elem[0]):
				for cnt in range(3):
					self.transfer_function[i*4 + cnt] = int(elem_prev[1][cnt] * ( elem[0] - i) / diff + elem[1][cnt] * (i - elem_prev[0]) / diff)

			elem_prev = elem
		print self.color_list
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.transfer_function)
		glBindTexture(GL_TEXTURE_2D, 0)


	def setTexture(self, x, col):
		self.transfer_function[x*4 + 0] = int(col[0]*255)
		self.transfer_function[x*4 + 1] = int(col[1]*255)
		self.transfer_function[x*4 + 2] = int(col[2]*255)
		
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.transfer_function)
		glBindTexture(GL_TEXTURE_2D, 0)

	def updateTexture2(self, trf):
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, trf)
		glBindTexture(GL_TEXTURE_2D, 0)


	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT)


		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.color_texId)

		glBegin(GL_QUADS)
		glVertex3f(-3.0, 3.0, -2) 
		glTexCoord2f(1, 1)
		glVertex3f( 3.0, 3.0, -2) 	
		glTexCoord2f(1, 0)
		glVertex3f( 3.0, 0.0, -2) 
		glTexCoord2f(0, 0)
		glVertex3f(-3.0, 0.0, -2) 
		glTexCoord2f(0, 1)
		glEnd()

		glBindTexture(GL_TEXTURE_2D, 0)
		glDisable(GL_TEXTURE_2D)



		# Draw Lines
		glColor3f(1, 1, 1)
		glLineWidth(3.0)
		glBegin(GL_LINE_STRIP)
		glVertex2f(-3.0, 0)

		for elem in range(256):
			glVertex2f(elem*6.0/255.0-3.0, self.transfer_alpha[elem] * 3.0/255.0 )
		glVertex2f(3.0, 0)
		glEnd()

		self.updated = 1

	def paintOverlayGL(self):
		self.paintGL()


	def resizeGL(self, width, height):
		self.width, self.height = width, height

		glViewport(0, 0, width, height)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-3, 3, -0.1, 3, -3, 3)
		
		glMatrixMode(GL_MODELVIEW)

		self.initTexture()

	def getTFF(self):
		for elem in range(256):
			self.transfer_function[4*elem + 3] = numpy.uint8(self.transfer_alpha[elem] * float(self.transfer_alpha[elem] / 255.0) * float(self.transfer_alpha[elem] /255.0 ))
		return self.transfer_function

	def setLoadedTFF(self):
		import math
		tmp_tf = numpy.asarray(self.transfer_function, dtype=numpy.float32)*255.0*255.0
		#for elem in range(256):
		#	tmp = tmp_tf[elem*4+3]
			#self.transfer_alpha[elem] = numpy.uint8(math.pow(tmp,1.0/3.0))


class TFN_widget2(QGLWidget):
	color_texId = 1
	transfer_function = numpy.zeros(256*4, dtype=numpy.uint8)
	transfer_alpha = numpy.zeros(256, dtype = numpy.uint8)

	def __init__(self, cnt, parent):
		super(TFN_widget2, self).__init__()
		self.color_texId += cnt
		self.updated = 0
		self.parent= parent

		self.pixmap = None
	
	def enterEvent(self, e):
		self.parent.app.setOverrideCursor(QtGui.QCursor(self.pixmap))

	def leaveEvent(self, e):
		self.parent.app.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

	def initTexture(self):

		glGenTextures(1, self.color_texId)
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
	
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.transfer_function)
	
		glBindTexture(GL_TEXTURE_2D, 0)


	def setTexture(self, x, col):
		self.transfer_function[x*4 + 0] = int(col[0]*255)
		self.transfer_function[x*4 + 1] = int(col[1]*255)
		self.transfer_function[x*4 + 2] = int(col[2]*255)
		
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.transfer_function)
		glBindTexture(GL_TEXTURE_2D, 0)

	def updateTexture2(self, trf):
		glBindTexture(GL_TEXTURE_2D, self.color_texId)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, trf)
		glBindTexture(GL_TEXTURE_2D, 0)


	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT)


		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.color_texId)

		glBegin(GL_QUADS)
		glVertex3f(-3.0, 3.0, -2) 
		glTexCoord2f(1, 1)
		glVertex3f( 3.0, 3.0, -2) 	
		glTexCoord2f(1, 0)
		glVertex3f( 3.0, 0.0, -2) 
		glTexCoord2f(0, 0)
		glVertex3f(-3.0, 0.0, -2) 
		glTexCoord2f(0, 1)
		glEnd()

		glBindTexture(GL_TEXTURE_2D, 0)
		glDisable(GL_TEXTURE_2D)



		# Draw Lines
		glColor3f(1, 1, 1)
		glLineWidth(3.0)
		glBegin(GL_LINE_STRIP)
		glVertex2f(-3.0, 0)

		for elem in range(256):
			glVertex2f(elem*6.0/255.0-3.0, self.transfer_alpha[elem] * 3.0/255.0 )

		glVertex2f(3.0, 0)
		glEnd()


		self.updated = 1

	def paintOverlayGL(self):
		self.paintGL()


	def resizeGL(self, width, height):
		self.width, self.height = width, height

		glViewport(0, 0, width, height)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-3, 3, -0.1, 3, -3, 3)
		
		glMatrixMode(GL_MODELVIEW)

		self.initTexture()

	def getTFF(self):
		for elem in range(256):
			self.transfer_function[4*elem + 3] = numpy.uint8(self.transfer_alpha[elem] * float(self.transfer_alpha[elem] / 255.0) * float(self.transfer_alpha[elem] /255.0 ))
		return self.transfer_function

	def setLoadedTFF(self):
		import math
		tmp_tf = numpy.asarray(self.transfer_function, dtype=numpy.float32)*255.0*255.0
		for elem in range(256):
			tmp = tmp_tf[elem*4+3]
			self.transfer_alpha[elem] = numpy.uint8(math.pow(tmp,1.0/3.0))

