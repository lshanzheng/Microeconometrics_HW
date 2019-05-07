//郑丽珊&宋悦溪 HW3 
reg percentcultivated temperature
gen p=percentcultivated
gen eps=1e-4
replace p=p-eps if p==1
gen z=log(p/(1-p))
reg z temperature
