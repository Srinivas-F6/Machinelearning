import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1,2,9,4,5]
plt.plot(x,y)
plt.scatter(x,y, c='#5367')
# plt.show()
plt.pie(x,labels=y)
plt.show()