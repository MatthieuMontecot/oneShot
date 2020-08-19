import numpy as np
import pygame
import sys
deltaT=0.5
import time
from sklearn.preprocessing import normalize
from scipy.special import expit
import pickle
displayBool=True #boolean setting either we display or not the simulation. The display process roughtly slows down the simulation by 40%
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255) 
if displayBool:
    windows_w=1300
    windows_h=700
    center=np.array((windows_w/2,windows_h/2)).reshape(1,2,1)
    screen = pygame.display.set_mode((windows_w, windows_h))
    clock = pygame.time.Clock()
    FPS = 10000
    screen.fill(WHITE)
else:
    screen=2 #just so that functions with a default screen argument gets something in function declaration

deltaPhi=0.2 #this number correspond to the angular differences between 2 lines of sight for each car, setting it to 2*np.pi/5 would mean a 360 degree vision but with big holes in it
Dmax=200 #this is the distance of sight
long=20 #half length of the car
larg=5 #harl width of the car
tmax=10000 #maximum number of iteration per epoch
v=105 #v=105,vm=20 is not bad, v is the number of vehicles i.e the population size
Lambda=0.01# some friction coefficient. This prevent quadratic speeds and ensures that the speed doesn't reach more
#that a maximum speed, in a natural maner due to limitations of the vehicle engine and this friction coefficient
m=20 #the number of population selected
r=4 #number of new individuals
store=False #either we store or not the best models
load=True  #either we load it from a file
fileNameStoringName=str(v)+'Cars.pkl'#the file where we store the models
fileNameLoadingName='105Cars.pkl'#the file where we store the models
noise=0.01
epochs=50#number of generations


class model():  #a model is a very simple neural network with 2 hidden layers of size 4,3, the input size is 5 and the output size is 2
                #the input are 5 values corresponding to the distance of sight in a given direction. (in front of the car, and 4 other directions (2 at right and 2 at left)
                #the output are 2 values, one in the range (0,1) corresponding to the engine power (i don't want cars to move backward because if let's them hack the perf function
                #the other output value is a float number in [-1,1] corresponding to the derivative of the angle of the car i.e the rotaion of the car. This rotation number is
                #multiplied later by some coefficient to be in an expected natural range. (a variation of 1 gradient in 0.1 second is not natural and allows models to do loops)
    def __init__(self):
        self.W1=np.random.uniform(-1,1,[5,4])
        self.b1=np.random.uniform(-1,1,[4,1])
        self.W2=np.random.uniform(-1,1,[4,3])
        self.b2=np.random.uniform(-1,1,[3,1])
        self.W3=np.random.uniform(-1,1,[3,2])
        self.b3=np.random.uniform(-1,1,[2,1])
    def feed(self): #this function allows to compute the output of the neuron using the self.X array (input) which should be set before
        self.h1=np.tanh(np.dot(np.transpose(self.W1),np.transpose(self.X))-self.b1)
        self.h2=np.tanh(np.dot(np.transpose(self.W2),self.h1)-self.b2)
        self.h3=np.dot(np.transpose(self.W3),self.h2)-self.b3
        acc=expit(self.h3[1])
        turn=np.tanh(self.h3[0])
        self.Y=np.array([turn,acc])
#the following functions are display functions using the pygame library, the names are straightforwards.
def plotRectange(l,color=BLACK):
    A,B,C,D=l
    pygame.draw.line(screen,color,A,B)
    pygame.draw.line(screen,color,B,C)
    pygame.draw.line(screen,color,C,D)
    pygame.draw.line(screen,color,D,A)
def plotLine(l):
    A,B=l
    pygame.draw.line(screen,BLUE,A,B)
def plotCross(A):
    pygame.draw.line(screen,BLUE,[A[0]-5,A[1]],[A[0]+5,A[1]])
    pygame.draw.line(screen,BLUE,[A[0],A[1]-5],[A[0],A[1]+5])
#to gain speed efficiency the following functions doesn't return new matrix but modify the previous ones, avoiding new matrix creations
def getEtheta(theta,eTheta):
    '''given a matrix theta of (v) angles (vehivles orientation), it modifys eTheta to be a matrix (v,2) of corresponding normalized vectors (cos(theta, sin(theta))
    those vectors are bidimentional have a norm of 1 and have a direction corresponding to the angle theta. eThata hasa shape  (v,2)'''

    eTheta[:,0]=np.cos(theta)
    eTheta[:,1]=np.sin(theta)


def getEthetaVdetection(thetaV,thetaVdetection,eThetaVdetection):
    '''To detect collisions between vehicles and lines, the method is to project the points of the corners of the rectangle  and the corresponding line along differen directions
    and see if they overlap. If they never overlap, and because they are convex objects, the do not collide, otherwise, they do. The directions of projection we need to check
    or the directions of edges of both convex polygons (a line can here be seen as a polygon. But because the edges of a recangles have only 2 different directions (due to parallelism,
    we need the orientation of the car and the perpendicular orientation. That's what we are looking for here.
    given a matrix thetaV of (v) angles, it creates a matrix thetaVdetection  of 2 angles, with the vehicles orientation and another one perpendicular to it.
    After, it puts in eThetaVdetection the corresponding director vectors, which are normalized (norm of 1) and oriented with corresponding angles'''

    thetaVdetection[:,0]=thetaV
    thetaVdetection[:,1]=thetaV+(np.pi/2)*np.ones(thetaV.shape)
    eThetaVdetection[:,:,0,0]=np.cos(thetaVdetection)
    eThetaVdetection[:,:,1,0]=np.sin(thetaVdetection)
    
def update(M,Vm,Am,theta,thetaPrime,VnotColliding,perf,Lambda=Lambda,thetaPmax=0.1):
    '''this updates the physics of the gave i.e the position, speed and orientations of the vehicles given there acceleration and angular momentums,
    end previous positions speed and orientations. Also, we use the friction parameter which compensate the acceleration proportionaly to the speed and prevent
    to have a higher speed that a certain limit. Alsoe, we update the performance of all vehicles by the distance they drove at this step, minus a constant at each iteration
    to penalize slow solutions.'''
    
    Vm+=deltaT*Am*(1-Lambda)-Lambda*Vm
    theta+=VnotColliding*(deltaT*thetaPmax*thetaPrime)
    eTheta=np.transpose(np.array([np.cos(theta),np.sin(theta)]))
    M+=VnotColliding[:,np.newaxis,np.newaxis]*(deltaT*(Vm*eTheta).reshape(v,1,2))
    perf+=VnotColliding*Vm.reshape(v)-0.1

#rectangle shape
corner1=np.array((long,larg))
corner2=np.array((-long,larg))
corner3=np.array((-long,-larg))
corner4=np.array((long,-larg))


def getCorners(M,theta,S,corner1=corner1,corner2=corner2,corner3=corner3,corner4=corner4):
    '''computes the corners of the vehicles given there center M and there orientations.'''
    eTheta=np.transpose(np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])) #rotation matrix of the vehicle, given his orientation
    S[:,0,:]=M+np.dot(eTheta,corner1)
    S[:,1,:]=M+np.dot(eTheta,corner2)
    S[:,2,:]=M+np.dot(eTheta,corner3)
    S[:,3,:]=M+np.dot(eTheta,corner4)
def pathCreator(l1,goalL):
    '''computes the path we want, avoiding to create a line on the arrival'''
    l=[l1[i:i+2]for i in range(len(l1)-1) if ((goalL[0][0]!=l1[i:i+2][0][0] or goalL[0][1]!=l1[i:i+2][0][1]) or (goalL[1][0]!=l1[i:i+2][1][0] or goalL[1][1]!=l1[i:i+2][1][1]))]
    l.append([l1[0],l1[-1]])
    return(l)


def select(models,perf,m,p=2,replace=True):
    '''probabilist selection process, proportinal to performances minus the minimum performances, to the power of p, the selection pressure.
    (the higher p is, the more we select high performing solutions'''
    f=np.array(perf)-min(perf)+0.01
    f=f/f.sum()
    f=f**p
    f=f/f.sum()
    selectedPopulation=np.random.choice(models,m,replace=replace,p=f)
    return(selectedPopulation)
def select2(models,perf,m,p=2):
    '''deterministic selection process, we select the m best ones'''
    ind = np.argpartition(perf, -m)[-m:]
    selectedPopulation=np.array(models)[ind]
    return(selectedPopulation)
def mixPop(selectedPopulation,m=m,v=v,mutationP=0.7):
    ''' return a list of models corresponding to selected models + offsprings that are crossovers + some mutation with some probability
    cross over points are random, and offsprings have weights of the 1st parents at the left of the crossover point, and the weights from the
    other parent at the right.'''
    newModels=[]
    listCoupleIdx=[]
    for i in range(m):
        for j in range(m):
            listCoupleIdx.append(str(i)+'.'+str(j))
    listParentsIdx=np.random.choice(listCoupleIdx,v-m,replace=False)
    for i in range(m):
        newModels.append(selectedPopulation[i])
    for i in range(v-m):
        child=model()
        crossOverPoint=np.random.randint(1,3)
        idx_1,idx_2=[int(s) for s in listParentsIdx[i].split('.')]
        child.W1=1*selectedPopulation[idx_1].W1
        child.b1=1*selectedPopulation[idx_1].b1
        if crossOverPoint==1:
            child.W2=1*selectedPopulation[idx_2].W2
            child.b2=1*selectedPopulation[idx_2].b2
        else:
            child.W2=1*selectedPopulation[idx_1].W2
            child.b2=1*selectedPopulation[idx_1].b2
        child.W3=1*selectedPopulation[idx_2].W3
        child.b3=1*selectedPopulation[idx_2].b3
        randomNumber=np.random.uniform(0,1,1)
        if randomNumber<mutationP:
            child=mutate(child)
            #print(i)
        newModels.append(child)
    return(newModels)
def mutate(child):
    '''mutation of a child, replace a given weight per another'''
    i=np.random.randint(3)
    j=np.random.randint(2)
    if i==0 and j==0:
        layer=child.W1
        child.W1[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)              
    if i==0 and j==1:
        layer=child.b1
        child.b1[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)              
    if i==1 and j==0:
        layer=child.W2
        child.W2[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)              
    if i==1 and j==1:
        layer=child.b2
        child.b2[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)                                 
    if i==2 and j==0:
        layer=child.W3
        child.W3[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)              
    if i==2 and j==1:
        layer=child.b3
        child.b3[np.random.randint(layer.shape[0]),np.random.randint(layer.shape[1])]=np.random.uniform(-1,1,1)              
    return(child)

def f(x):
    '''return the opposit booleaan'''
    return(not(x))
f=np.vectorize(f)

def updatePhis(deltaV,phis,deltaPhi=deltaPhi):
    '''updates view angles of vehicles'''
    for k in range(5):
        phis[:,k]=thetaV+(k-2)*deltaPhi*np.ones((v))

def getRotPhi(phis):
    '''update rotation matrix of phis, the view angles of the car, those matrixes are used to change coordonate system to the one of each angle'''
    rotPhi[:,:,0,0]=np.cos(phis)
    rotPhi[:,:,0,1]=np.sin(phis)
    rotPhi[:,:,1,0]=-np.sin(phis)
    rotPhi[:,:,1,1]=np.cos(phis)

def collisionDetection(eThetaLproj,eThetaVdetection,S,lines,dif):
    '''returns a boolean array of eiter vehicles collided'''
    Pv=np.einsum('ijz,klj->iklz',eThetaLproj,S)#l,v,4,1 ligne voiture points, thetaL(1)
    Pl=np.einsum('ijky,lmkz->limj',eThetaVdetection,lines)#l,v,2,2  ligne voiture points thetaV
    #projection of lines and cars with angles with there own edges (to check overlaps with Pv and Pl in same coordonate systems)
    Cl=np.einsum('ijz,iljy->izly',eThetaLproj,lines)#l,1, points,1
    Cv=np.einsum('ijkz,imk->zimj',eThetaVdetection,S)#1,v, points, thetaV
    #then, we ll check over lap, there is no overlap if the min of polygon1 if smaller of max of polygon2 or vice versa. (Now, everything has been projected on a 1 dimentional axis
    #min and max of all points:
    mCl=np.min(Cl,axis=-2)#l,1,thetaL
    mCv=np.min(Cv,axis=-2)#1,v,thetaV
    mPl=np.min(Pl,axis=-2)#l,v,thetaV
    mPv=np.min(Pv,axis=-2)#l,v,thetaL(1)
    MCl=np.max(Cl,axis=-2)#l,1,thetaL
    MCv=np.max(Cv,axis=-2)#1,v,thetaV
    MPl=np.max(Pl,axis=-2)#l,v,thetaV
    MPv=np.max(Pv,axis=-2)#l,v,thetaL(1)
    #so we compute min-max for each couple (Pv,Cl),(Pl,Cv)
    difL1=np.max(mPv-MCl,axis=-1)#l,v, if there is one positiv number in one of those dif matrix, there is no collision for the gicen couple line,vehicle.
    difL2=np.max(mCl-MPv,axis=-1)#l,v,
    difV1=np.max(mPl-MCv,axis=-1)#l,v,
    difV2=np.max(mCv-MPl,axis=-1)#l,v,
    dif[:,:,0]=difL1
    dif[:,:,1]=difL2
    dif[:,:,2]=difV1
    dif[:,:,3]=difV2
    dualCol=np.max(dif,axis=-1)
    VnotColliding=np.sign(np.min(dualCol,axis=0))#if sign!=1, M should stop and theta too (1= doesn't collide, 0= the touch by corner to corner (should stop),-1,they overlap.
    VnotColliding[VnotColliding<0]=0
    return(VnotColliding)

def getDistanceOfSight(B,Mv,rotPhi,eThetaL,Dmax=Dmax):
    Br1=np.einsum('vpik,vlk->lvpi',rotPhi,B-Mv)#l,v,5,coordinates of B the 1st verticises of lines in the phi coordinate system (evrything rotated by the corresponding phi)
    eThetaLr1=np.einsum('vpik,lkz->lvpi',rotPhi,eThetaL)#l,v,5,coordinates of eThetaLr1 the director vectors of lines in the phi coordinate system (evrything rotated by the corresponding phi)
    works0_idx=eThetaLr1[:,:,:,1]!=0           #allows to test if eTheta L is paralel to phi, which leads to special case (either they are aligned or not
    #b is used to know how much we should retrieve eThetaLr1 from Br1 to reach horizntal axis (in phi coordinate system, to get the distance of sight
    b[works0_idx]=np.divide(Br1[:,:,:,1:2][works0_idx],eThetaLr1[:,:,:,1:2][works0_idx])
    #b l,v,5,2 pour theta!=0 (works0)
    H[works0_idx]=Br1[works0_idx]-b[works0_idx]*eThetaLr1[works0_idx]#point d intersection des 2 droites l,v,5,2
    #H=coordonnees des points d intersection pour theta!=0
    ## if theta=0 and aligned, we should set D to b (but not relevant in the paths i plan to use)
    H[f(works0_idx),0]=Dmax# if phi // thetaL, put line of sight=Dmax
    #the following lines are necessary if the intersection point betweren the sight line and the border is out of the border, put Dmax in those cases
    works8_idx=abs(b[:,:,:,0])>lengths
    H[works8_idx,0]=Dmax
    works9_idx=b[:,:,:,0]>0
    H[works9_idx,0]=Dmax
    #if the intersection point is behind the car we shouldn't consider that it limits the line of sight
    H[H<0]=Dmax
    D=H[:,:,:,0].min(axis=0)
    D[D>Dmax]=Dmax
    return(D)
 
def level1(v=v):
    A=np.array(([50,50]))
    B=np.array(([250,50]))
    C=np.array(([450,100]))
    D=np.array(([600,200]))
    E=np.array(([650,350]))
    F=np.array(([650,450]))
    G=np.array(([600,600]))
    H=np.array(([450,750]))
    I=np.array(([350,850]))
    J=np.array(([400,1000]))
    K=np.array(([600,1050]))
    L=np.array(([750,1050]))
    M=np.array(([850,950]))
    N=np.array(([900,850]))
    O=np.array(([900,500]))
    P=np.array(([1000,500]))
    Q=np.array(([1000,850]))
    R=np.array(([950,950]))
    S=np.array(([800,1150]))
    T=np.array(([600,1200]))
    U=np.array(([400,1200]))
    V=np.array(([250,1050]))
    W=np.array(([200,900]))
    X=np.array(([250,750]))
    Y=np.array(([400,650]))
    Z=np.array(([500,550]))
    A1=np.array(([550,450]))
    B1=np.array(([500,300]))
    C1=np.array(([400,200]))
    D1=np.array(([250,150]))
    E1=np.array(([50,150]))
    lines=np.array(pathCreator([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,A1,B1,C1,D1,E1],[O,P]))
    M=100*np.ones((v,2))            #initial position of the cars
    initialThetaV=np.zeros((v))
    return(lines,M,initialThetaV)
def level2(v=v):
    A=np.array(([50,50]))
    B=np.array(([250,50]))
    C=np.array(([450,100]))
    D=np.array(([600,200]))
    E=np.array(([650,350]))
    F=np.array(([650,450]))
    G=np.array(([600,600]))
    H=np.array(([450,750]))
    I=np.array(([350,850]))
    J=np.array(([400,1000]))
    K=np.array(([600,1050]))
    L=np.array(([750,1050]))
    M=np.array(([850,950]))
    N=np.array(([900,850]))
    O=np.array(([900,500]))
    P=np.array(([1000,500]))
    Q=np.array(([1000,850]))
    R=np.array(([950,950]))
    S=np.array(([800,1150]))
    T=np.array(([600,1200]))
    U=np.array(([400,1200]))
    V=np.array(([250,1050]))
    W=np.array(([200,900]))
    X=np.array(([250,750]))
    Y=np.array(([400,650]))
    Z=np.array(([500,550]))
    A1=np.array(([550,450]))
    B1=np.array(([500,300]))
    C1=np.array(([400,200]))
    D1=np.array(([250,150]))
    E1=np.array(([50,150]))
    lines=np.array(pathCreator([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,A1,B1,C1,D1,E1],[O,P]))
    lines[:,:,1]=-lines[:,:,1]
    M=100*np.ones((v,2))            #initial position of the cars
    M[:,1]=-M[:,1]
    initialThetaV=np.zeros((v))
    return(lines,M,initialThetaV)

def initializeLevel(level,v=v):
    '''initialize matrix depending on level specifics'''
    lines,M,initialThetaV=level()
    l=lines.size//4
    lines=lines.reshape(l,2,2,1)
    lengths=np.sqrt(np.square(lines[:,1,:]-lines[:,0,:]).sum(axis=-2)).reshape(l,1,1)#lentghs of lines l,1,1, used to check if the intersection beteween vision direction of the vehicles
        #and the line is on the line
    B=lines[:,0,:].reshape(1,l,2)#1,l coordinates of the 1st vertices of each line
    deltaLines=lines[:,1,:,0]-lines[:,0,:,0]    #l, coordinate
    thetaL=np.angle(deltaLines[:,0]+1j*deltaLines[:,1])#l
    eThetaL=normalize(deltaLines, axis=1, norm='l2').reshape(l,2,1)#l,coordinate
    eThetaLproj=np.transpose(np.array([np.sin(thetaL),-np.cos(thetaL)])).reshape(l,2,1) #calcul of projection matrixes from a view parallel to lines, which is used in collision detection
    H=np.zeros((l,v,5,2))   #will be used to store coordinates of intersections of eThetaVdetection and lines (and later determine either they intersect on the line.),
                            #which are used in sight Distance calculus
    b=np.zeros((l,v,5,1))   #after turning the lines in the vehicle coordinate system, used to calculate the intersection point quantifying the amount of director vector of the line necessary
                            #to retrieve to be on the horizontal axis in the vehicle coordinate system
    dif=np.zeros((l,v,4)) #used as storage matrix in collision detection
    return(lines,M,initialThetaV,lengths,B,deltaLines,thetaL,eThetaL,eThetaLproj,H,b,dif,l)

def resetEpoch(v,e,initialThetaV,noise=0):
    Vm=np.ones((v,1))   #vehicle speed
    Mv[:,0]=M           #vehicle position
    thetaV=initialThetaV
    thetaV+=e*noise*np.random.uniform(-1,1,v)
    thetaVdetection=np.zeros((v,2)) #angles of projection among vehicle edges do detect collisions
    thetaVdetection[:,0]=thetaV
    thetaVdetection[:,1]=thetaV+(np.pi/2)*np.ones(thetaV.shape)
    eThetaV=np.transpose(np.array([np.cos(thetaV),np.sin(thetaV)]))
    eThetaVdetection=np.transpose(np.array([-np.sin(thetaVdetection),np.cos(thetaVdetection)])).reshape(v,2,2,1)#v, angle , coordinate,
    #projection matrix along 1 axis, from a view parallel to vehicle edges in order to detect collisions
    perf=np.zeros((v))  #performance of each model, used in the parenting selection process
    t=0
    return(Vm,thetaV,thetaVdetection,eThetaVdetection,eThetaV,perf,t)

tTot=0 #total amount of frames*

if True:
    phis=np.zeros((v,5))            #initialisation of matrix to be modified befores
    rotPhi=np.zeros((v,5,2,2))      #matrix of projection among phi orientations (phi are angles of vision of cars
    Am=np.ones((v,1))               #acceleration matrix
    S=np.zeros((v,4,2))             # v,4,2, corners of vehicles
    #calculation of thetaL and eThetaL (angles and corresponding director vector of lines (normalized)
    projectedSight=np.zeros((v,5,2))    #stores the position of sight views of each vehicles (blue crosses displayed on the screen)
    ePhis=np.zeros((v,5,2)) #angles of vision of each vehicle
    thetaVPrime=0.01*np.ones((v))   #derivative of direction of each vehicle: the rotation of each vehicle
    Mv=np.zeros((v,1,2))    #the position of each vehicle
t0=time.time()

levels=[level1]             #we can use several levels to avoid overfitting to the first part of the course the level 2 is symnetric to lvl 1 and allows to train turning right
                            #and left simultaneously
if load:
    with open(fileNameLoadingName, 'rb') as input:
        selectedPopulation= pickle.load(input)
        models=mixPop(selectedPopulation,m=len(selectedPopulation))
else:
    models=[model() for i in range(v)]  #the list of NN models in the population
    
    
for e in range(epochs):
    print('generation',e)
    #initialize level related matrixes (lines and line projection related matrixes
    lvl=0
    for level in levels:
        lvl+=1
        print('level:',lvl)
        lines,M,initialThetaV,lengths,B,deltaLines,thetaL,eThetaL,eThetaLproj,H,b,dif,l=initializeLevel(level,v=v)
        #firtst,reset the positions, performances, speed and p 
        Vm,thetaV,thetaVdetection,eThetaVdetection,eThetaV,perf,t=resetEpoch(v,e,initialThetaV,noise=0)
        if displayBool:
            clock.tick(FPS)
        while t<tmax:
            if displayBool:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # if we close the window, exit
                        print(tTot/(time.time()-t0),'frames per second with',v,'cars and',l,'lines')
                        pygame.quit()
                        sys.exit()
            tTot+=1
            if displayBool:
                clock.tick(FPS)
                pygame.display.update()
                screen.fill(WHITE)
            #get rectangles    
            getCorners(Mv[:,0,:],thetaV,S)#v,4,2
            #get eThetaV, the director vectors of the vehicle, along their orientation
            getEtheta(thetaV,eThetaV)
            #collision detection process:   VnotColliding eThetaLproj eThetaVdetection S lines
            #first, get EthetaVdetection i.e the director vectors of projections along vehicle edges, this allows to see either there is angles where they don't overlap
            #which would mean they are not colliding, the same is done for lines, but since they don't move, we did it only once at initialisation
            getEthetaVdetection(thetaV,thetaVdetection,eThetaVdetection)#update thetaVdetection and eThetaVdetection
            #second, project vehicles with line angles and lines with vehicle edges angles
            #for convex polygon collision detection (they don't collide if and only if at least one angle of view parallel to edges of first or second polygon shows they don't overlap
            #Pv=projection of vehicles with line edges angles and Pl=projection of lines with vehicles edges angles
            VnotColliding=collisionDetection(eThetaLproj,eThetaVdetection,S,lines,dif)
            #creation of phis, the angles of view of each vehicle
            updatePhis(thetaV,phis)
            #creation of rotPhis, the rotation matrix for phis, used to move from eulerian coordinate system to phi coordinate system (rotate everything so the line of sight is horizontal
            #and the corresponding distance is the coordinates on the 1st axis
            getRotPhi(phis)
            #get the distance of sight of each car, given B the 1st point of each line, Mv the car positions, rotPhi, the rotation matrix of each view angle of vehicles and eThetaL the director
            #vectors of each lines
            D=getDistanceOfSight(B,Mv,rotPhi,eThetaL)
            #all inside this if loop are display functions
            if displayBool: #if we want to display the simulation
                ePhis[:,:,0]=np.cos(phis)
                ePhis[:,:,1]=np.sin(phis)
                projectedSight.fill(0)  #used to display the lines of sight of the vehicles (blue crosses)
                projectedSight+=Mv+D[:,:,np.newaxis]*ePhis
                bestV=Mv[np.argmax(perf)].reshape(1,2,1)-center
                bestV=bestV.reshape(1,2)
                #those are display functions, bestV is used to center the camera on the best vehicle
                for li in lines:
                    plotLine((li-bestV.reshape(1,2,1))[:,:,0])
                for i in range(len(S)):
                    voit=S[i]   
                    if VnotColliding[i]==1:
                        plotRectange(voit-bestV)
                    else:
                        plotRectange(voit-bestV,(255,0,0))
                for p in projectedSight[VnotColliding==1]:
                    for k in p:
                        plotCross((k-bestV)[0])
            if not 1 in VnotColliding or max(perf)>12000:   #if all vehicles crashed or after a max number of iterations, let's mmove to the next generation
                break
            for i in range(len(models)):        #let's feed our neural networks
                models[i].X=D[i:i+1]/Dmax       #set the entry of each model to the corresponding distances of sight
                models[i].feed()                #propagate de entry
                thetaVPrime[i]=models[i].Y[0]   #the rotation of the vehicle is the 1st number of the output
                Am[i]=models[i].Y[1]            #the acceleration of the vehicle is the 2nd number of the output
            #now, we can calculate new positions, rotations, performances and speeds
            update(Mv,Vm,Am,thetaV,thetaVPrime,VnotColliding,perf)
            t+=1 
    #selection process
    p=min(0.5+0.1*e,3) #the selection pressure increase with time from easy to hard
    perf+=VnotColliding*perf
    selectedPopulation=select(models,perf,m,p)
    #generate offsprings
    models=mixPop(selectedPopulation,m=m,v=v)
    models[-r:]=[model() for k in range(r)]
    print('epoch',e,'perf mean',np.mean(perf),'perf max',max(perf))
print(tTot/(time.time()-t0),'frames per second with',v,'cars and',l,'lines')
#store 10 best models
if store:
    with open(fileNameStoringName, 'wb') as output:
        pickle.dump(models, output, pickle.HIGHEST_PROTOCOL)
if displayBool:
    pygame.quit()
sys.exit()
