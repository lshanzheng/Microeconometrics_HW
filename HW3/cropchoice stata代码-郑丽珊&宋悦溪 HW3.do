//郑丽珊&宋悦溪 HW3 
gen eps=1e-4
gen p1=noncrop/field
drop p1
gen p1=noncrop/ fields
gen p2=corn/ fields
gen p3=wheat/ fields
gen p4=rice/ fields
replace p1=p1+eps if p1==0
replace p2=p2+eps if p2==0
replace p3=p3+eps if p3==0
replace p4=p4+eps if p4==0
gen z1=log(p2)-log(p1)
reg z1 temperature rainfall
gen z2=log(p3)-log(p1)
reg z2 temperature rainfall
gen z3=log(p4)-log(p1)
reg z3 temperature rainfall
