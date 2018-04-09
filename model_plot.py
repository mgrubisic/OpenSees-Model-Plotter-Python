# Plotting tool for OpenSees models using Python
# Created: November 2017
# Gerard O'Reilly
# Copyright by Gerard J. O'Reilly, 2017

print("Begin")
print("===========================")

# Import some libraries
import os
import numpy as np

# Input variables
# ------------------------------------------------------------------------------------------------
vw='3d';	#	Viewpoint for plotting ('3d', )
Npf=0; 		# 	Node label plot flag (1 for labels, 0 for none)
Epf=0; 		# 	Element label plot flag (1 for labels, 0 for none)
LPpf=1; 	# 	Load pattern plot flag (1 for labels, 0 for none)
eig=0; 		#	number of mode shapes, 0 to do nothing

# Define the file names
# ------------------------------------------------------------------------------------------------
# fname_model="model.txt";
# fname_model="/Users/Gerard/GDrive/ProgettoScuoleGerard/ModalAnalysis/Carrara_model_v5.txt"
# fname_model="/Users/Gerard/GDrive/ProgettoScuoleGerard/ModalAnalysis/Ancona_model_v5.txt"
fname_model="/Users/Gerard/GDrive/PhD/Modelling/RC_pre1970/Galli/SPO/info/Galli_model_6st_Bare.txt";
# fname_model="/Users/Gerard/GDrive/OpenSees/GLD_frames/outs_Galli_SPO/info/models/Galli_model_2st_Bare.txt";
# fname_eigen="eigenVectors_modal.txt";
# fname_periods="Periods_modal.txt";

# Define the classes
# ------------------------------------------------------------------------------------------------
class node:
	Xamp=5; # 	Amplification on Xcoord disps
	Yamp=5; # 	Amplification on Ycoord disps
	Zamp=5; # 	Amplification on Zcoord disps

	num=[];
	Xcoord=[];
	Ycoord=[];
	Zcoord=[];
	Xdisp=[];
	Ydisp=[];
	Zdisp=[];
	Xmass=[];
	Ymass=[];
	Zmass=[];

	# Create class method to append new nodes
	@classmethod
	def add_node(self,num,Xcoord,Ycoord,Zcoord,Xdisp,Ydisp,Zdisp,Xmass,Ymass,Zmass):
		self.num.append(num);
		self.Xcoord.append(Xcoord);
		self.Ycoord.append(Ycoord);
		self.Zcoord.append(Zcoord);
		self.Xdisp.append(Xdisp);
		self.Ydisp.append(Ydisp);
		self.Zdisp.append(Zdisp);
		self.Xmass.append(Xmass);
		self.Ymass.append(Ymass);
		self.Zmass.append(Zmass);

class element(node):
	# Initialise the lists within the class
	typ=[];
	num=[];
	iNode=[];
	jNode=[];
	iXcoord=[];
	iYcoord=[];
	iZcoord=[];
	jXcoord=[];
	jYcoord=[];
	jZcoord=[];

	# Create class method to append new elements
	@classmethod
	def add_element(self,typ,num,iNode,jNode):
		self.typ.append(typ);
		self.num.append(num);
		self.iNode.append(iNode);
		self.jNode.append(jNode);

	# Create class method to identify element co-ordinates
	@classmethod
	def def_ele_coord(self,node):
		for i in range(len(self.num)):
			idi=node.num.index(self.iNode[i]);
			idj=node.num.index(self.jNode[i]);

			self.iXcoord.append(node.Xcoord[idi]+node.Xamp*node.Xdisp[idi]);
			self.iYcoord.append(node.Ycoord[idi]+node.Yamp*node.Ydisp[idi]);
			self.iZcoord.append(node.Zcoord[idi]+node.Zamp*node.Zdisp[idi]);
			self.jXcoord.append(node.Xcoord[idj]+node.Xamp*node.Xdisp[idj]);
			self.jYcoord.append(node.Ycoord[idj]+node.Yamp*node.Ydisp[idj]);
			self.jZcoord.append(node.Zcoord[idj]+node.Zamp*node.Zdisp[idj]);

if LPpf==1:
	class load_pattern(node):
		num=[];
		typ=[];
		node=[];
		fX=[];
		fY=[];
		fZ=[];
		mX=[];
		mY=[];
		mZ=[];
		Xcoord=[];
		Ycoord=[];
		Zcoord=[];

		# Create class method to append new load patterns
		@classmethod
		def add_load_pattern(self,num,typ,node,fX,fY,fZ,mX,mY,mZ):
			self.num.append(num);
			self.typ.append(typ);
			self.node.append(node);
			self.fX.append(fX);
			self.fY.append(fY);
			self.fZ.append(fZ);
			self.mX.append(mX);
			self.mY.append(mY);
			self.mZ.append(mZ);

		# Create class method to identify load patterns coordinates
		@classmethod
		def def_load_pattern_coord(self,node):
			for i in range(len(self.num)):
				temp_coord_X=[];
				temp_coord_Y=[];
				temp_coord_Z=[];
				for j in range(len(self.node[i])):
					idx=node.num.index(self.node[i][j]);
					temp_coord_X.append(node.Xcoord[idx]+node.Xamp*node.Xdisp[idx]);
					temp_coord_Y.append(node.Ycoord[idx]+node.Yamp*node.Ydisp[idx]);
					temp_coord_Z.append(node.Zcoord[idx]+node.Zamp*node.Zdisp[idx]);

				self.Xcoord.append(temp_coord_X);
				self.Ycoord.append(temp_coord_Y);
				self.Zcoord.append(temp_coord_Z);

# Start reading the model file
# ------------------------------------------------------------------------------------------------
with open(fname_model,"r") as model_file:
	for x in model_file:
		# Get the nodes
		if x.find(" Node: ")==0:
			node_num=x.split()[1]; # Take second then turn remainder to integer
			x1=next(model_file); # Get the next line
			Xcoord=float(x1.split()[2]); # Get the x co-ordinate
			Ycoord=float(x1.split()[3]); # Get the y co-ordinate
			Zcoord=float(x1.split()[4]); # Get the z co-ordinate
			x1=next(model_file); # Get the next line
			Xdisp=float(x1.split()[1]); # Get the x co-ordinate
			Ydisp=float(x1.split()[2]); # Get the x co-ordinate
			Zdisp=float(x1.split()[3]); # Get the x co-ordinate
			for i in range(8):
				x1=next(model_file);
				if i==4:
					Xmass=float(x1.split()[0]); # Get the x mass
				elif i==5:
					Ymass=float(x1.split()[1]); # Get the y mass
				elif i==6:
					Zmass=float(x1.split()[2]); # Get the z mass
			node.add_node(node_num,Xcoord,Ycoord,Zcoord,Xdisp,Ydisp,Zdisp,Xmass,Ymass,Zmass);


		# Get the elements
		# Look for Elements
		if x.find("Element:")==0:
			ele_num=x.split()[1];
			ele_type=x.split()[3];
			if ele_type=="ForceBeamColumn3d":
				ele_iNode=x.split()[6];
				ele_jNode=x.split()[7];
			elif  ele_type=="ZeroLength":
				ele_iNode=x.split()[5];
				ele_jNode=x.split()[7];
			elif  ele_type=="Truss":
				ele_iNode=x.split()[5];
				ele_jNode=x.split()[7];
			element.add_element(ele_type,ele_num,ele_iNode,ele_jNode);

		# Look for ElasticBeam3d
		if x.find("ElasticBeam3d")==0:
			ele_type="ElasticBeam3d";
			ele_num=x.split()[1]; # This splits takes the second and converts it to an integer
			x1=next(model_file); # This takes the next line of the modelfile
			ele_iNode=x1.split()[2]; # Take first
			ele_jNode=x1.split()[3]; # Take second
			element.add_element(ele_type,ele_num,ele_iNode,ele_jNode);

		# Look for load patterns
		if LPpf==1:
			if x.find("Load Pattern: ")==0:
				lp_num=x.split()[2];
				# Shift down two lines to get load pattern type
				x1=next(model_file);
				x1=next(model_file);
				lp_typ=x1.split()[0];
				x1=next(model_file);
				x1=next(model_file);
				lp_node=[];
				lp_fX=[];
				lp_fY=[];
				lp_fZ=[];
				lp_mX=[];
				lp_mY=[];
				lp_mZ=[];
				while x1.find("Nodal Load: ")==0:
					lp_node.append(x1.split()[2]);
					lp_fX.append(float(x1.split()[5]));
					lp_fY.append(float(x1.split()[6]));
					lp_fZ.append(float(x1.split()[7]));
					lp_mX.append(float(x1.split()[8]));
					lp_mY.append(float(x1.split()[9]));
					lp_mZ.append(float(x1.split()[10]));
					x1=next(model_file);
				load_pattern.add_load_pattern(lp_num,lp_typ,lp_node,lp_fX,lp_fY,lp_fZ,lp_mX,lp_mY,lp_mZ);

# Compute the element and load-pattern co-ordinates
# ------------------------------------------------------------------------------------------------
element.def_ele_coord(node);
if LPpf==1:
	load_pattern.def_load_pattern_coord(node);

# Plot the model using Matplotlib
# ------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig=plt.figure();
# ax=fig.add_subplot(111, projection=vw);
# # Plot the nodes
# for i in range(len(node.num)):
# 	# Get node color depending on the presence of mass or not
# 	if node.Xmass[i]>0 or node.Ymass[i]>0 or node.Zmass[i]>0:
# 		node_sz=30;
# 		node_clr='k';
# 		node_shp='o';
# 	else:
# 		node_clr='k';
# 		node_sz=5;
# 		node_shp='s';
# 	ax.scatter(node.Xcoord[i]+node.Xamp*node.Xdisp[i],node.Ycoord[i]+node.Yamp*node.Ydisp[i],node.Zcoord[i]+node.Zamp*node.Zdisp[i],color=node_clr,s=node_sz,marker=node_shp);
# # Plot the elements
# for i in range(len(element.num)):
# 	# Set up the color of the element types
# 	if element.typ[i]=='ForceBeamColumn3d':
# 		ele_clr='b';
# 	elif element.typ[i]=='Truss':
# 		ele_clr='g';
# 	elif element.typ[i]=='ElasticBeam3d':
# 		ele_clr='k';
# 	elif element.typ[i]=='ZeroLength':
# 		ele_clr='m';
# 	ax.plot([element.iXcoord[i], element.jXcoord[i]],[element.iYcoord[i], element.jYcoord[i]],[element.iZcoord[i], element.jZcoord[i]],linestyle='-',color=ele_clr,linewidth=1);
# if LPpf==1:
# 	# From here: https://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector/11156353#11156353
# 	from matplotlib.patches import FancyArrowPatch
# 	from mpl_toolkits.mplot3d import proj3d
# 	class Arrow3D(FancyArrowPatch):
#
# 	    def __init__(self, xs, ys, zs, *args, **kwargs):
# 	        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
# 	        self._verts3d = xs, ys, zs
#
# 	    def draw(self, renderer):
# 	        xs3d, ys3d, zs3d = self._verts3d
# 	        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
# 	        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
# 	        FancyArrowPatch.draw(self, renderer)
#
# 	# Plot the load patterns
# 	for i in range(len(load_pattern.num)):
# 		for j in range(len(load_pattern.node[i])):
# 			if load_pattern.fX[i][j]!=0:
# 				a = Arrow3D([load_pattern.Xcoord[i][j]-np.sign(load_pattern.fX[i][j])*1.0, load_pattern.Xcoord[i][j]], [load_pattern.Ycoord[i][j], load_pattern.Ycoord[i][j]], [load_pattern.Zcoord[i][j], load_pattern.Zcoord[i][j]], mutation_scale=20,lw=1, arrowstyle="-|>", color="r");
# 				ax.add_artist(a);
# 			elif load_pattern.fY[i][j]!=0:
# 				a = Arrow3D([load_pattern.Xcoord[i][j], load_pattern.Xcoord[i][j]], [load_pattern.Ycoord[i][j]-np.sign(load_pattern.fY[i][j])*1.0, load_pattern.Ycoord[i][j]], [load_pattern.Zcoord[i][j], load_pattern.Zcoord[i][j]], mutation_scale=20,lw=1, arrowstyle="-|>", color="r");
# 				ax.add_artist(a);
# 			elif load_pattern.fZ[i][j]!=0:
# 				a = Arrow3D([load_pattern.Xcoord[i][j], load_pattern.Xcoord[i][j]], [load_pattern.Ycoord[i][j], load_pattern.Ycoord[i][j]], [load_pattern.Zcoord[i][j]-np.sign(load_pattern.fZ[i][j])*1.0, load_pattern.Zcoord[i][j]], mutation_scale=20,lw=1, arrowstyle="-|>", color="r");
# 				ax.add_artist(a);
# ax.set_xlabel('X');
# ax.set_ylabel('Y');
# ax.set_zlabel('Z');
# plt.show();

# Plot the model using Plotly
# ------------------------------------------------------------------------------------------------
import plotly.plotly as py
import plotly.tools as plyt
import plotly.graph_objs as go
plyt.set_credentials_file(username='gerard.oreilly', api_key='SMgMODaCoZjC7LgYJsHY')


# Plot the nodes
data1=[]
for i in range(len(node.num)):
	# Get node color depending on the presence of mass or not
	if node.Xmass[i]>0 or node.Ymass[i]>0 or node.Zmass[i]>0:
		node_sz=10;
		node_clr='rgb(0,0,0)';
		node_shp='o';
	else:
		node_clr='rgb(255,0,0)';
		node_sz=5;
		node_shp='s';
	trace=go.Scatter3d(x=node.Xcoord[i]+node.Xamp*node.Xdisp[i],y=node.Ycoord[i]+node.Yamp*node.Ydisp[i],z=node.Zcoord[i]+node.Zamp*node.Zdisp[i],mode='markers',marker=dict(color=node_clr,size=node_sz));
	data1.append(trace)

# Plot the elements
for i in range(len(element.num)):
	# Set up the color of the element types
	if element.typ[i]=='ForceBeamColumn3d':
		ele_clr='rgb(0,0,255)';
	elif element.typ[i]=='Truss':
		ele_clr='rgb(0,128,0)';
	elif element.typ[i]=='ElasticBeam3d':
		ele_clr='rgb(0,0,0)';
	elif element.typ[i]=='ZeroLength':
		ele_clr='rgb(255,0,255)';
	trace=go.Scatter3d(x=[element.iXcoord[i], element.jXcoord[i]],y=[element.iYcoord[i], element.jYcoord[i]],z=[element.iZcoord[i], element.jZcoord[i]],mode='lines',line=dict(color=ele_clr));
	data1.append(trace)

layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=10))

fig = go.Figure(data=data1, layout=layout)
py.iplot(fig, filename='OpenSees-Model-Plotter-Python-Demo2')






print("This is the end of the file")
print("===========================")
