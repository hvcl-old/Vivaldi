from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
from OpenGL.GL import *
import numpy
import Image
import time
import os


import Vivaldi_transfer_function as VTF
import Vivaldi_multi_slider as VMS

window = None
app = None
viewer_data = None
xb, yb, zb = 0, 0 ,0
transX, transY, transZ = 0, 0, 0
viewer_on = False
trans_on = False
transN = 0
slider_on = False


# TEMP
img_cnt = 0
sc_img = False

FPS_prev = time.time()
FPS_cnt = 0
diff = 0.00000001

v = None

def enable_viewer(dummy, trans=None, dimension='3D', TF_bandwidth=1, sld=None):
	global v, viewer_on, trans_on, transN , slider_on
	if trans == 'TFF':
		trans_on = True
		transN = 1
	elif trans == 'TFF2':
		trans_on = True
		transN = 2
	if sld == 'SLIDER':
		slider_on = True
		
	viewer_on = True
	v = Vivaldi_viewer(dummy, dimension, TF_bandwidth)

	print "Viewer is Enabled"

	v.show()

def VIVALDI_GATHER(dp):
	pass

def collect_result(data_pkg):
	tmp = VIVALDI_GATHER(data_pkg)
	return tmp, data_pkg.data_contents_dtype

class Vivaldi_viewer():
	slider = None
	def __init__(self, FNandARG, dim, TF_bandwidth):
		self.app = QtGui.QApplication(['Vivaldi viewer'])
		self.window = Vivaldi_window(self, FNandARG)
		self.window.set_control_type(dim)
		self.window.set_app(self.app)
		self.TF_bandwidth = TF_bandwidth 
		
		self.slider = self.window.slider
		
		

	def show(self):
		self.window.show() 
		self.window.update_widget()

		#global trans_on, transN
		#if trans_on == True:
			#self.window.TFF.show()
			#if transN==2:
				#self.window.TFF2.show()

		#if slider_on == True:
			#self.slider.show()
				
		self.app.exec_()

	def getTFF(self):
		return numpy.array(self.window.TFF.getTFF())

	def getTFF2(self):
		return numpy.array(self.window.TFF2.getTFF())

	def get_sliders(self):
		return numpy.array([self.slider.slider_dict[0].value(), self.slider.slider_dict[1].value(), self.slider.slider_dict[2].value(), self.slider.slider_dict[3].value()],dtype=numpy.int32)
	def get_slider_opacity(self):
		return numpy.array([self.slider.slider_opacity_dict[0].value(), self.slider.slider_opacity_dict[1].value(), self.slider.slider_opacity_dict[2].value(), self.slider.slider_opacity_dict[3].value()], dtype=numpy.int32)

	def getTFBW(self):
		return self.TF_bandwidth

	def getIsTFupdated(self):
		if self.window.TFF == None:
			return 0 
		return self.window.TFF.updated

	def getIsTFupdated2(self):
		if self.window.TFF2 == None:
			return 0 

		return self.window.TFF2.updated

	def getFB(self):
		return self.window.FB

	def enable_TFF(self):
		pass
		
pressedButton = 0
class Vivaldi_window(QtGui.QMainWindow):
	TFF = None
	TFF2 = None
	slider = None
	def __init__(self, parent, FNandARG):
		super(Vivaldi_window, self).__init__()

		_, self.func_name, self.args = FNandARG
		#print "\n\n\n\n\n\n\n", self.args[4][0], "\n\n\n\n\n\n\n"
		self.inBox_val = 0
		self.widget = Vivaldi_widget(self, self.args[2])

		global trans_on, transN, slider_on
		if trans_on == False:
			self.setGeometry(200, 100, self.widget.width, self.widget.height+30)
		else:
			self.setGeometry(200, 100, self.widget.width+420, self.widget.height+30)
		if trans_on == True:
			self.TFF = VTF.TFN_widget(self)
			self.TFF.setGeometry(self.widget.width+20, 30, 400, 200)
			if transN == 2:
				self.TFF2 = VTF.TFN_widget(self)
				self.TFF2.setGeometry(self.widget.width+20, 260, 400, 200)

		if slider_on == True:
			self.slider = VMS.multi_slider(self)
			self.slider.setGeometry(self.widget.width+20, 30+200, 400,100)
	
		#self.setCentralWidget(self.widget)
		self.setWindowTitle("Vivaldi")

		self.parent = parent
		self.CRG = 0
		self.viewer_image_cnt = 0

		#for video
		self.ret_image_cnt=0

		self.isnotExist = True
		self.folder_name = ''

		# Elongate Z direction 
		#LoadIdentity()
		#Translate(0, 0, self.args[1][0].full_data_shape[0] * 2 * 3.0 /2.0 )
		#Scaled(1,1,3)
	
		#Translate(-self.args[1][0].full_data_shape[2]/2.0, -self.args[1][0].full_data_shape[1]/2.0,  -self.args[1][0].full_data_shape[0]/2.0)
		#self.z_ = self.args[1][0].full_data_shape[0] * 3.0 / 2.0 * 2

		# Origin
		LoadIdentity()
		Translate(-self.args[1][0].full_data_shape[2]/2.0, -self.args[1][0].full_data_shape[1]/2.0,  self.args[1][0].full_data_shape[0]*2.0)
		self.z_ = self.args[1][0].full_data_shape[0]*5.0/2.0


	def set_control_type(self,dim):
		self.dimension = dim

	def loadmmtx(self, filename):
		global mmtx;
		mmtx = numpy.fromstring(open(filename,'r').read(), dtype=numpy.float32).reshape(4,4)

	def loadinvmmtx(self, filename):
		global inv_mmtx;
		inv_mmtx = numpy.fromstring(open(filename,'r').read(), dtype=numpy.float32).reshape(4,4)

	def loadtff(self, filename):
		self.parent.TFF.widget.transfer_function = numpy.fromstring(open(filename,'r').read(), dtype=numpy.uint8)
		self.parent.TFF.widget.updateTexture2(self.parent.TFF.widget.transfer_function)
		self.parent.TFF.widget.setLoadedTFF()
		self.parent.TFF.widget.updateOverlayGL()
		self.update_widget()
	def loadalpha(self, filename):
		self.parent.TFF.widget.transfer_alpha = numpy.fromstring(open(filename,'r').read(), dtype=numpy.uint8)
		self.parent.TFF.widget.updateTexture2(self.parent.TFF.widget.transfer_function)
		self.parent.TFF.widget.setLoadedTFF()
		self.parent.TFF.widget.updateOverlayGL()
		self.update_widget()
		

	def load_mvmtx_tf(self, filename):
		print filename
		if filename != '':
			fptr = open(filename,'r')
	
			mvmtx_name = fptr.readline().strip()
			invmvmtx_name = fptr.readline().strip()
			tf_name = fptr.readline().strip()
			global mmtx, inv_mmtx, transN

			if transN==2:
				tf2_name = fptr.readline().strip()
			if tf2_name is not '':
				self.parent.TFF2.widget.transfer_function = numpy.fromstring(open(tf2_name,'r').read(), dtype=numpy.uint8)
				self.parent.TFF2.widget.updateTexture2(self.parent.TFF2.widget.transfer_function)
				self.parent.TFF2.widget.setLoadedTFF()
				self.parent.TFF2.widget.updateOverlayGL()

			mmtx = numpy.fromstring(open(mvmtx_name,'r').read(), dtype=numpy.float32).reshape(4,4)
			inv_mmtx = numpy.fromstring(open(invmvmtx_name,'r').read(), dtype=numpy.float32).reshape(4,4)
			self.parent.TFF.widget.transfer_function = numpy.fromstring(open(tf_name,'r').read(), dtype=numpy.uint8)
			fptr.close()
			self.parent.TFF.widget.updateTexture2(self.parent.TFF.widget.transfer_function)
			self.parent.TFF.widget.setLoadedTFF()
			self.parent.TFF.widget.updateOverlayGL()
			self.update_widget()

		

	def save_mvmtx_tf(self):
		if self.isnotExist:
			os.system("mkdir -p result/")
			self.folder_name += self.args[1] + str(int(time.time())%10000)
			os.system("mkdir -p result/%s"%(self.folder_name))
			print "Created folder name:", os.getcwd(), self.folder_name
			self.isnotExist = False
		
		a = Image.fromarray(self.widget.data)
		save_file_name = "./result/%s/snapshot%s"%(self.folder_name,str(self.viewer_image_cnt))
		a.save(save_file_name+".png")
		f = open(save_file_name+".mvmtx", "w")

		global mmtx, inv_mmtx, v, transN
		f.write(mmtx)
		f.close()
		f = open(save_file_name+".invmvmtx", "w")
		f.write(inv_mmtx)
		f.close()
		f = open(save_file_name+".tf", "w")
		f.write(v.getTFF())
		f.close()

		f = open(save_file_name+".alpha", "w")
		f.write(self.parent.TFF.widget.transfer_alpha)
		f.close()
	
		if transN == 2:
			f = open(save_file_name+".tf2", "w")
			f.write(v.getTFF2())
			f.close()
		
		
		f = open(save_file_name+".comb", 'w')
		text = save_file_name+".mvmtx\n"+save_file_name+".invmvmtx\n" + save_file_name+".tf\n"
		if transN == 2:
			text += save_file_name+".tf2\n"
		f.write(text)
		f.close()
	
		self.viewer_image_cnt+=1

	#for video
	def Make_cinema_source(self):
		for elem in range(130):
			viewer_trans(-transX, -transY, -self.z_-transZ)
			viewer_rotate(360/130.0, 0, 10, 0)
			viewer_trans(transX, transY, self.z_+transZ)

			data_pkg = self.func_name(*self.args)
			viewer_data, viewer_dtype = collect_result(data_pkg)
		
			self.widget.setData(viewer_data, viewer_data.shape[1], viewer_data.shape[0])
			

			a = Image.fromarray(self.widget.data)
			a.save("./resultsss/result-"+str('%03d'%(self.ret_image_cnt))+".tif")
			self.ret_image_cnt = self.ret_image_cnt + 1



		

	def keyPressEvent(self, event):
		global func_dict_
		dummy = None
		if type(event) == QtGui.QKeyEvent:
			if event.key() == QtCore.Qt.Key_Escape:
				self.app.exit()
			#elif event.key() == QtCore.Qt.Key_A:func_dict['A'](dummy)
			elif event.key() == QtCore.Qt.Key_A:
				viewer_trans(0, 0, 50)
				self.z_ = self.z_ + 50
			elif event.key() == QtCore.Qt.Key_B:func_dict['B'](dummy)
			#elif event.key() == QtCore.Qt.Key_C:func_dict['C'](dummy)
			elif event.key() == QtCore.Qt.Key_C:
				self.CRG=0
			elif event.key() == QtCore.Qt.Key_D:func_dict['D'](dummy)
			elif event.key() == QtCore.Qt.Key_E:func_dict['E'](dummy)
			elif event.key() == QtCore.Qt.Key_F:func_dict['F'](dummy)
			#elif event.key() == QtCore.Qt.Key_G:func_dict['G'](dummy)
			elif event.key() == QtCore.Qt.Key_G:
				self.CRG = 2
			elif event.key() == QtCore.Qt.Key_H:func_dict['H'](dummy)
			#elif event.key() == QtCore.Qt.Key_I:func_dict['I'](dummy)
			elif event.key() == QtCore.Qt.Key_I:
				filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
				self.loadmmtx(filename)
			#elif event.key() == QtCore.Qt.Key_J:func_dict['J'](dummy)
			elif event.key() == QtCore.Qt.Key_J:
				filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
				self.loadinvmmtx(filename)
			#elif event.key() == QtCore.Qt.Key_K:func_dict['K'](dummy)
			elif event.key() == QtCore.Qt.Key_K:
				filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
				self.loadtff(filename)
			elif event.key() == QtCore.Qt.Key_L:
				#filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
				#self.load_mvmtx_tf(filename)
				filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
				self.loadalpha(filename)
			#for video
			#elif event.key() == QtCore.Qt.Key_M:func_dict['M'](dummy)
			elif event.key() == QtCore.Qt.Key_M:
				self.Make_cinema_source()
			elif event.key() == QtCore.Qt.Key_N:func_dict['N'](dummy)
			elif event.key() == QtCore.Qt.Key_O:func_dict['O'](dummy)
			elif event.key() == QtCore.Qt.Key_P:func_dict['P'](dummy)
			elif event.key() == QtCore.Qt.Key_Q:func_dict['Q'](dummy)
			#elif event.key() == QtCore.Qt.Key_R:func_dict['R'](dummy)
			elif event.key() == QtCore.Qt.Key_R:
				self.CRG=1
			elif event.key() == QtCore.Qt.Key_S:
				self.save_mvmtx_tf()
										
				#a = Image.fromarray(self.widget.data)
				#a.save("./result/result-"+str(self.viewer_image_cnt)+".tif")
				#self.viewer_image_cnt+=1
			#elif event.key() == QtCore.Qt.Key_T:func_dict['T'](dummy)
			elif event.key() == QtCore.Qt.Key_T:
				self.parent.enable_TFF()
			#elif event.key() == QtCore.Qt.Key_U:func_dict['U'](dummy)
			elif event.key() == QtCore.Qt.Key_U:
				self.update_widget()
			elif event.key() == QtCore.Qt.Key_V:func_dict['V'](dummy)
			elif event.key() == QtCore.Qt.Key_W:func_dict['W'](dummy)
			elif event.key() == QtCore.Qt.Key_X:func_dict['X'](dummy)
			elif event.key() == QtCore.Qt.Key_Y:func_dict['Y'](dummy)
			elif event.key() == QtCore.Qt.Key_Z:func_dict['Z'](dummy)
			else:
				event.ignore()

		self.update_widget()

	def inBox(self, x, y):
		if x > 0 and x < self.widget.width and y > 30 and y < self.widget.height+30:
			return 1
		elif x > self.widget.width+20 and y > 30 and y < 230:
			return 2
		elif x > self.widget.width+20 and y > 250 and y < 450 and self.TFF2!=None:
			return 3
		else:
			return 0

	def mousePressEvent(self, event):
		global pressedButton
		pressedButton = event.button()
		
		self.inBox_val = self.inBox(event.x(), event.y())

		if self.inBox_val == 1:
			if self.dimension == '3D':
				if event.button() == 1:
					import math
					global xb, yb, zb
					x, y = event.x(), (event.y()-30)
					width, height = self.widget.width+20, self.widget.height
	
					xb = (2.0*x - width) / width
					yb = (height - 2.0*y) /height
					d = math.sqrt(xb*xb + yb*yb)
					if d > 1.0 : d = 1.0
					zb = math.cos(math.pi/2.0 * d)
			
					a = 1.0 / math.sqrt(xb*xb + yb*yb + zb*zb)
					xb *= a
					yb *= a
					zb *= a
	
				elif event.button() == 2:
					global scale_y
					scale_y = event.y()
				elif event.button() == 4:
					global trans_x, trans_y
					trans_x, trans_y = event.x(), event.y()-30
			elif self.dimension == '2D':
				if event.button() == 1:
					trans_x, trans_y = event.x(), event.y()-30
				elif event.button() == 2:
					scale_y = event.y()
				elif event.button() == 4:
					pass
		elif self.inBox_val == 2:
			tmp_x = int((event.x()-(self.widget.width+20)) * 255 / 400)
			tmp_y = int((200-(event.y()-30)) * 255 / 200)

			if event.button() == 1:
			 	#set alpha value	
				self.TFF.transfer_alpha[tmp_x] = tmp_y;
				self.prev_x, self.prev_y = tmp_x, tmp_y
			
			elif event.button() == 2:
				self.prev_x = tmp_x
				self.TFF.setColor(tmp_x, self.current_color)
				
	
			elif event.button() == 4:
				self.col = QtGui.QColorDialog.getColor()
				self.current_color = (self.col.red(),self.col.green(), self.col.blue())
				self.TFF.pixmap.fill(self.col)
			
		
			self.TFF.updateGL()

		elif self.inBox_val == 3:
			tmp_x = int((event.x()-(self.widget.width+20)) * 255 / 400)
			tmp_y = int((200-(event.y()-250)) * 255 / 200)

			if event.button() == 1:
			 	#set alpha value	
				self.TFF2.transfer_alpha[tmp_x] = tmp_y;
				self.prev_x, self.prev_y = tmp_x, tmp_y
			
			elif event.button() == 2:
				self.prev_x = tmp_x
				self.TFF2.setColor(tmp_x, self.current_color)
				
	
			elif event.button() == 4:
				self.col = QtGui.QColorDialog.getColor()
				self.current_color = (self.col.red(),self.col.green(), self.col.blue())
				self.TFF2.pixmap.fill(self.col)
			
		
			self.TFF2.updateGL()

		
			
	def mouseMoveEvent(self, event):
		global pressedButton
		if self.inBox_val == 1:
			if pressedButton == 1:
				if self.dimension == '3D':
					self.rotate_3D(event)
				elif self.dimension == '2D':
					self.trans_2D(event)
			
			elif pressedButton == 2:
				if self.dimension == '3D':
					self.scale_3D(event)
				elif self.dimension == '2D':
					self.scale_2D(event)
				
			elif pressedButton == 4:
				if self.dimension == '3D':
					self.trans_3D(event)
				elif self.dimension == '2D':
					pass

			self.update_widget()

		elif self.inBox_val == 2:
			if pressedButton == 1:
				tmp_x = int((event.x()-(self.widget.width+20)) * 255 / 400)
				tmp_y = int((200-(event.y()-30)) * 255 / 200)
		
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
						self.TFF.transfer_alpha[elem] = self.prev_y + slope * (elem - self.prev_x)
						#self.TFF.updateGL()

				self.TFF.transfer_alpha[tmp_x] = tmp_y
				self.TFF.updateGL()
				self.prev_x, self.prev_y = tmp_x, tmp_y

			elif pressedButton == 2:
				pass

			self.update_widget()

		elif self.inBox_val == 3:
			if pressedButton == 1:
				tmp_x = int((event.x()-(self.widget.width+20)) * 255 / 400)
				tmp_y = int((200-(event.y()-250)) * 255 / 200)
		
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
						self.TFF2.transfer_alpha[elem] = self.prev_y + slope * (elem - self.prev_x)
						#self.TFF.updateGL()

				self.TFF2.transfer_alpha[tmp_x] = tmp_y
				self.TFF2.updateGL()
				self.prev_x, self.prev_y = tmp_x, tmp_y

			elif pressedButton == 2:
				pass
	

			self.update_widget()

	def rotate_3D(self, event):
		import math
		x, y = event.x(), (event.y()-30)
		width, height = self.widget.width, self.widget.height

		xa = (2.0*x - width) / width
		ya = (height - 2.0*y) /height
		d = math.sqrt(xa*xa + ya*ya)
		if d > 1.0 : d = 1.0
		za = math.cos(math.pi/2.0 * d)

		a = 1.0 / math.sqrt(xa*xa + ya*ya + za*za)
		xa *= a
		ya *= a
		za *= a

		global xb, yb, zb
		if xb is not xa or yb is not ya or zb is not za:
			xaxis, yaxis, zaxis = yb*za - zb*ya, zb*xa - xb*za, xb*ya - yb*xa
			root_value = xa*xb + ya*yb + za*zb 
			if root_value < 0: root_value = 0
			
			angle = 90 * math.acos(math.sqrt(root_value))

			xb, yb, zb = xa, ya, za

		global transZ, transX, transY

		viewer_trans(-transX, -transY, -self.z_-transZ)
		viewer_rotate(angle, xaxis, -yaxis, -zaxis)
		viewer_trans(transX, transY, self.z_+transZ)

	def scale_3D(self, event):
		global scale_y, transX, transY
		scale_value = 1.0 + float(scale_y - event.y())/300

		viewer_trans(-transX, -transY, -self.z_-transZ)
		viewer_scale(scale_value ,scale_value ,scale_value)
		viewer_trans(transX, transY, self.z_+transZ)

		scale_y = event.y()
	
	def trans_3D(self, event):
		global trans_x, trans_y, transX, transY
		dx = trans_x - event.x()
		dy = trans_y - (event.y()-30)
		transX += dx
		transY += dy
	
		viewer_trans(dx, dy, 0)

		trans_x , trans_y = event.x(), event.y()

	def trans_2D(self, event):
		global trans_x, trans_y
		dx = trans_x - event.x()
		dy = trans_y - event.y()
	
		self.widget.trans(dx,dy)

		trans_x , trans_y = event.x(), event.y()

	def scale_2D(self, event):
		global scale_y
		
		self.widget.scale(float(scale_y - event.y())/300)
		
		scale_y = event.y()
	
	def mouseReleaseEvent(self, event):
		global pressedButton 
		pressedButton = 0
		self.mouse_flag = 0
		self.prev_x = 0
		self.prev_y = 0
		self.inBox_val = 0


	def wheelEvent(self, event):
		global transZ
		transZ += event.delta()/60

		viewer_trans(0, 0, event.delta()/60)

		self.update_widget()

	def set_app(self, app):
		self.app = app
	
	def update_widget(self):
		import time 
		st = time.time()
		global FPS_prev, img_cnt, sc_img
		FPS_prev = st
		
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glMultMatrixf(mmtx)
		st3 = time.time()
		data_pkg = self.func_name(*self.args)
		viewer_data, viewer_dtype = collect_result(data_pkg)
		print "DATA return", time.time() - st3
		st4 = time.time()
		self.widget.setDtype(viewer_dtype)
		self.widget.setData(viewer_data, viewer_data.shape[1], viewer_data.shape[0], data_pkg.data_contents_memory_dtype)

		self.widget.updateGL()

		print "texture mapping time: ", time.time() - st4
		self.get_FPS()
		aft = time.time()
		print "Viewer latency: ", aft-st

	def get_FPS(self):
		global FPS_prev, diff, FPS_cnt
		
		import time
				
		st = time.time()
		diff += st - FPS_prev 
		FPS_cnt += 1
	
		if diff > 1.0:	
			self.setWindowTitle("Vivaldi " + str(round(FPS_cnt / diff, 1)))
			diff = 0.00000001
			FPS_cnt = 0
			

func_dict = {}

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	
def dummy_function(dummy=None):pass

for elem in range(26):
	func_dict[alphabet[elem]] = dummy_function
	
def attach_function_to_key(key, function):
	func_dict[key] = function

	
def viewer_rotate(angle, x, y, z):
	import math
	pi = math.pi
	if x+y+z != 0:
		l = 1/math.sqrt(x*x + y*y + z*z)
	else: l = 1
	x, y, z = x*l, y*l, z*l

	#matrix
	th = math.pi/180*(angle)
	c = math.cos(th)
	s = math.sin(th)
	tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
	tm = tm.reshape((4,4))
	
	global mmtx
	mmtx = numpy.dot(tm, mmtx)


	#inverse
	th = math.pi/180*(-angle)
	c = math.cos(th)
	s = math.sin(th)
	tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
	tm = tm.reshape((4,4))
	global inv_mmtx
	inv_mmtx = numpy.dot(inv_mmtx, tm)



def viewer_trans(x, y, z):
	#matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][3] = x
	tm[1][3] = y
	tm[2][3] = z
	global mmtx
	mmtx = numpy.dot(tm, mmtx)


	#inverse matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][3] = -x
	tm[1][3] = -y
	tm[2][3] = -z
	global inv_mmtx
	inv_mmtx = numpy.dot(inv_mmtx, tm)

	
def viewer_scale(x, y, z):
	#matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][0] = x
	tm[1][1] = y
	tm[2][2] = z
	global mmtx
	mmtx = numpy.dot(tm, mmtx)


		#inverse matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][0] = 1.0/x
	tm[1][1] = 1.0/y
	tm[2][2] = 1.0/z
	global inv_mmtx
	inv_mmtx = numpy.dot(inv_mmtx, tm)


class Vivaldi_widget(QGLWidget):
	width, height = 600, 600
	data_width, data_height= 1, 1 
	texid = 0
	flag = 0
	data = None
	dtype = GL_LUMINANCE
	transx, transy = 0, 0
	scale_factor = 1.0
	
	def __init__(self, parent, work_range):
		super(Vivaldi_widget, self).__init__(parent)
		a = work_range['y']
		self.width = work_range['x'][1] - work_range['x'][0]
		self.height= work_range['y'][1] - work_range['y'][0]

		self.setGeometry(0, 30, self.width, self.height)

		

	def initializeGL(self):
		self.data = numpy.zeros((300, 300), dtype=numpy.uint8)

	def initData(self):
		glGenTextures(1, self.texid)
		glBindTexture(GL_TEXTURE_2D, self.texid)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.data_width, self.data_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

		glBindTexture(GL_TEXTURE_2D,0)

	
		

	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT)
		glEnable(GL_TEXTURE_2D)

		glPushMatrix()
		glLoadIdentity()

		glBindTexture(GL_TEXTURE_2D, self.texid)


		glLoadIdentity()

		glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
		glTranslatef(self.transx, self.transy, 0)
	
		glBegin(GL_QUADS)
		glVertex2f(-1, -1)
		glTexCoord2f(0, 0)
		glVertex2f(-1, 1)
		glTexCoord2f(1, 0)
		glVertex2f(1, 1)
		glTexCoord2f(1, 1)
		glVertex2f(1, -1)
		glTexCoord2f(0, 1)
		glEnd()
		glBindTexture(GL_TEXTURE_2D, 0)
		glPopMatrix()
	def paintOverlayGL(self):
		self.paintGL()

	def resizeGL(self, width, height):
		self.width, self.height = width, height
		self.initData()
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-1, 1, -1, 1, -1, 1)

	def update_texture(self):
		glBindTexture(GL_TEXTURE_2D, self.texid)
		glTexImage2D(GL_TEXTURE_2D, 0, self.dtype, self.data_width, self.data_height, 0, self.dtype, GL_UNSIGNED_BYTE, self.data)
		#glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.data_width, self.data_height, self.dtype, GL_UNSIGNED_BYTE, self.data)
		glBindTexture(GL_TEXTURE_2D, 0)


	def setData(self, data, width, height, memdtype):
		if memdtype == numpy.uint8:
			self.data = numpy.array(numpy.clip(data,0, 255), dtype=numpy.uint8)
		else:
			self.data = numpy.array(data/data.max()*255, dtype=numpy.uint8)
		self.data_width = width
		self.data_height = height

		self.update_texture()
	
	def setDtype(self, dtype):
		#if type(self.data.shape)
		if len(self.data.shape) == 2:
			self.dtype = GL_LUMINANCE
		elif self.data.shape[2] == 4 or dtype == 'RGBA':
			self.dtype = GL_RGBA
		elif self.data.shape[2] == 3 or dtype=='RGB':
			self.dtype = GL_RGB
		else: 
			self.dtype = GL_LUMINANCE

	def trans(self, x, y):
		self.transx += x
		self.transy += y

	def scale(self, factor):
		self.scale_factor += factor

def LoadIdentity():
	global mmtx
	global inv_mmtx
	mmtx = numpy.eye(4,dtype=numpy.float32)
	inv_mmtx = numpy.eye(4,dtype=numpy.float32)

def Rotate(angle, x, y, z):
	import math
	pi = math.pi

	l = x*x + y*y + z*z
	l = 1/math.sqrt(l)
	x = x*l
	y = y*l
	z = z*l

	#matrix
	th = math.pi/180*(angle)
	c = math.cos(th)
	s = math.sin(th)
	tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
	tm = tm.reshape((4,4))
	global mmtx
	mmtx = numpy.dot(mmtx, tm)


	#inverse
	th = math.pi/180*(-angle)
	c = math.cos(th)
	s = math.sin(th)
	tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
	tm = tm.reshape((4,4))
	global inv_mmtx
	inv_mmtx = numpy.dot(tm, inv_mmtx)

def Translate(x, y, z):
	#matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][3] = x
	tm[1][3] = y
	tm[2][3] = z
	global mmtx
	mmtx = numpy.dot(mmtx, tm)


	#inverse matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][3] = -x
	tm[1][3] = -y
	tm[2][3] = -z
	global inv_mmtx
	inv_mmtx = numpy.dot(tm, inv_mmtx)

def Scaled(x, y, z):
	#matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][0] = x
	tm[1][1] = y
	tm[2][2] = z
	global mmtx
	mmtx = numpy.dot(mmtx, tm)


	#inverse matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][0] = 1.0/x
	tm[1][1] = 1.0/y
	tm[2][2] = 1.0/z
	global inv_mmtx
	inv_mmtx = numpy.dot(tm, inv_mmtx)



