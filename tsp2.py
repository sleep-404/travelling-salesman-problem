import numpy as np
import matplotlib.pyplot as plt
import matplotlib
neurons=1000
epochs=80
sigo=500
no=0.9

r1=epochs/np.log(no)
r2=epochs/np.log(sigo)

x=np.array([[20833.33],[20900],[21300],[21600],[21600],[21600],[22183.33],[22583.33],[22683.33],[23616.67],[23700],[23883.33],[24166.67],[25149.17],[26133.33],[26150],[26283.33],[26433.33],[26550],[26733.33],[27026.11],[27096.11],[27153.61],[27166.67],[27233.33],[27233.33],[27266.67],[27433.33],[27462.5]])
y=np.array([[17100],[17066.67],[13016.67],[14150],[14966.67],[16500],[13133.33],[14300],[12716.67],[15866.67],[15933.33],[14533.33],[13250],[12365.83],[14500],[10550],[12766.67],[13433.33],[13850],[11683.33],[13051.94],[13415.83],[13203.33],[9833.33],[10450],[11783.33],[10383.33],[12400],[12992.22]])


x_n=(x-np.amin(x))/(np.amax(x)-np.amin(x))
y_n=(y-np.amin(y))/(np.amax(y)-np.amin(y))

w=np.random.rand(neurons+1,2)

plt.scatter(x_n,y_n,s=12,color='red')

for i in range(29):
    plt.text(x_n[i], y_n[i], str(i+1))

for m in range(epochs):
    for i in range(29):
        x_input=np.array([x_n[i],y_n[i]]).reshape(1,2)
        xw=np.zeros((neurons,1))
        for j in range(neurons):
            xw[j]=((x_input[0,0]-w[j,0])*(x_input[0,0]-w[j,0])+(x_input[0,1]-w[j,1])*(x_input[0,1]-w[j,1]))
        i_x=np.argmin(xw)
        d=np.zeros((neurons,1))
        for k in range(neurons):
            d[k]=np.minimum(np.abs(i_x-k),neurons-np.abs(i_x-k))
        #weight update
        n=no*np.exp(-1*m/r1)
        sig=sigo*np.exp(-1*m/r2)

        for l in range(neurons):
            w[l,:]=w[l,:]+n*np.exp(-1*d[l]*d[l]/(2*sig*sig))*(x_input-w[l,:])



w[neurons,0]=w[0,0]
w[neurons,1]=w[0,1]

plt.title("Travelling_salesman_problem")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.plot(w[:,0],w[:,1])
plt.scatter(x_n,y_n,s=12,color='red')
plt.show()
