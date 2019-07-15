in="overthrust.vites"
n1=801 n2=801  n3=187 
o1=0.0 o2=0.0  o3=0.0
d1=25.0 d2=25.0 d3=25.0
esize=4
data_format=xdr_float
2.1-git	sfdd	data/overthrust/model:	kazeiv@kw60444	Wed May 22 11:45:33 2019

	data_format="native_float"
	esize=4
	in="stdout"
	in="stdin"

2.1-git	sfscale	data/overthrust/model:	kazeiv@kw60444	Wed May 22 11:45:33 2019

	data_format="native_float"
	esize=4
	in="stdout"
	in="stdin"

2.1-git	sfput	data/overthrust/model:	kazeiv@kw60444	Wed May 22 11:45:33 2019

	data_format="native_float"
	unit=km/s
	in="/home/kazeiv/Madagascar/RSFTMP/data/overthrust/model/overthrust.rsf@"
	d1=0.025
	unit1=km
	d2=0.025
	unit2=km
	d3=0.025
	unit3=km
	label=Velocity
	label1=X
	label2=Y
	label3=Z
2.1-git	sfwindow	data/overthrust/model:	kazeiv@kw60444	Fri May 24 17:17:19 2019

	d2=0.025
	o1=0
	n2=187
	d3=0.025
	n3=1
	o2=0
	label1="X"
	data_format="native_float"
	o3=5
	label2="Z"
	f2=200
	label3="Y"
	esize=4
	in="stdout"
	unit1="km"
	unit2="km"
	unit3="km"
	d1=0.025
	n1=801
	in="stdin"

2.1-git	sftransp	data/overthrust/model:	kazeiv@kw60444	Fri May 24 17:17:19 2019

	o1=0
	d2=0.025
	n2=801
	label1="Z"
	o2=0
	data_format="native_float"
	label2="X"
	esize=4
	in="stdout"
	unit1="km"
	unit2="km"
	d1=0.025
	n1=187
	in="stdin"

2.1-git	sfwindow	data/overthrust/model:	kazeiv@kw60444	Fri May 24 17:17:19 2019

	d2=0.025
	n2=801
	o1=0
	o2=0
	label1="Z"
	data_format="native_float"
	label2="X"
	--out=stdout
	esize=4
	max1=3.5
	in="stdout"
	unit1="km"
	unit2="km"
	d1=0.025
	n1=141
	in="stdin"

2.1-git	sfput	data/overthrust/model:	kazeiv@kw60444	Fri May 24 17:55:18 2019

	in="/home/kazeiv/Madagascar/RSFTMP/overthrust2D.hh@"
	d2=25
	data_format="native_float"
	unit1=m
	unit2=m
	d1=25
2.1-git	sfmath	Manuscripts/log_estimation/data:	kazeiv@kw60444	Fri May 24 19:12:53 2019

	x2=2
	input=0
	data_format="native_float"
	esize=4
	in="/home/kazeiv/Madagascar/RSFTMP/overthrust_2D.hh@"
	x1=1
2.1-git	sfwindow	Manuscripts/log_estimation/data:	kazeiv@kw60444	Sat May 25 12:33:56 2019

	d2=25
	n2=801
	o1=25
	o2=0
	label1="Z"
	f1=1
	data_format="native_float"
	label2="X"
	esize=4
	in="/home/kazeiv/Madagascar/RSFTMP/overthrust2D.hh@"
	unit1="m"
	unit2="m"
	d1=25
	n1=120
