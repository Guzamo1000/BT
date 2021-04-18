s=input()
# d=len(s)
# ef=d

def xl1(s):
    x1=s.split()
    x2=[]

    for i in x1:
        st=''
        for j in i:
            if j.isalpha():
                st= st + j
        st=st.lower()
        st=st.capitalize()
        if st.isalpha:
            x2.append(st)
    aw=' '.join(x2)
    return aw
aw=xl1(s)
# print(aw)
def xl2(aw):
    st1=''
    st2=''
    ter=''
    d=len(aw)-1
    num=d
    while d>0 and aw[d]!=' ':
        d-=1
    st1=st1+aw[0]
    for i in range(1,d,1):
        if aw[i-1]==' ':
            st1=st1+aw[i]
    ter=ter+st1
    ter=ter+'.'
    for i in range(d+1,num+1,1):
        st2=st2+aw[i]
    ter=ter+st2
    return ter
ef=xl2(aw)
print(ef)



