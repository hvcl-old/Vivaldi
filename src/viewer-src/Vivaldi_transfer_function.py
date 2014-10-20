from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
from OpenGL.GL import *
import numpy 
import Image

global_texCnt = 1
class TFN_widget(QGLWidget):
	
	def __init__(self, parent):
		super(TFN_widget, self).__init__(parent)
		global global_texCnt
		self.color_texId = global_texCnt
		global_texCnt = global_texCnt + 1

		self.updated = 0
		self.parent= parent
		self.pixmap = QtGui.QPixmap(10,10)
		self.transfer_function = numpy.zeros(256*4, dtype=numpy.uint8)
		for elem in range(256):
			self.transfer_function[4*elem + 3] = elem
		self.color_list = {0: (0,0,0), 255: (0,0,0)}
		self.transfer_alpha = numpy.zeros(256, dtype = numpy.uint8)

		for elem in range(256):
			self.transfer_alpha[elem] = elem

	
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

		glPushMatrix()
		glLoadIdentity()

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
	
		glPopMatrix()

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
