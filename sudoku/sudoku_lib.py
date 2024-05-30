import numpy as np
import exercices as ex

def load_friendly(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        quizzes = np.zeros((9,9), np.int32)
        for i, line in enumerate(lines):
            quizzes[i] = [int(c) for c in line.split(',')[:9]]
        M = ex.createM(np.transpose(quizzes))
    return M

def fill_M_with_s(M, s):
    assert(len(s) == 81)
    for i in range(9):
        M[i] = [int(c) for c in s[9 * i: 9 * (i + 1)]]

def from_M_to_s(M):
    A=np.sum(M,axis=-1)
    s = ''
    for b in range(3):
        for d in range(3):
            for a in range(3):
                for c in range(3):
                    if A[a,b,c,d]==1:
                        k=np.where(M[a,b,c,d]==1)
                        s+=str(k[0][0]+1)
                    else:
                        s+='0'
    return s
    
    
def plot(M):
    A=np.sum(M,axis=-1)
    print('_____________')
    for b in range(3):
        for d in range(3):
            s=''
            for a in range(3):
                for c in range(3):
                    if A[a,b,c,d]==1:
                        k=np.where(M[a,b,c,d]==1)
                        s+=str(k[0][0]+1)
                    else:
                        s+='X'
                if a < 2:
                    s += ','
            print(s)
        if b <2:
            print('------------')
    print('_____________')
    
def roll(M,depth=1):
    '''temporary bit'''
    loop=1
    sumM=-5
    newA=1
    newC=1
    savingM=M
    while loop:
        loop=0
        while sumM!=np.sum(M):
            sumM=np.sum(M)
            A=np.sum(M,axis=-1)
            if 0 in A:
                break
            if newA==1:
                Aprime=A
                newA=0
                a,b,c,d=np.where(A==1)
            else:
                a,b,c,d=np.where(A*(Aprime!=A)==1)
            for f,g,h,i in zip(a,b,c,d):
                k=np.where(M[f,g,h,i]==1)[0]
                mask=ex.getMask(f,g,h,i,k)
                M=M*mask
#etape B:
            detectHorizontaly=np.sum(M,axis=(0,2))
            boxesLines,lines,ks=np.where(detectHorizontaly==1)
            for yBox,k,line in zip(boxesLines.flatten(),ks.flatten(),lines.flatten()):
                abscisseBoxes,abscisses,=np.where(M[:,yBox,:,line,k]==1)
                if len(abscisseBoxes)!=0:
                    mask=ex.getMask(abscisseBoxes[0],yBox,abscisses[0],line,k)
                    M=M*mask
        
            detectVerticaly=np.sum(M,axis=(1,3))
            boxesColumns,columns,ks=np.where(detectVerticaly==1)
            for xBox,k,column in zip(boxesColumns.flatten(),ks.flatten(),columns.flatten()):    
                columnBoxes,ys,=np.where(M[xBox,:,column,:,k]==1)
                if len(columnBoxes)!=0:
                    mask=ex.getMask(xBox,columnBoxes[0],column,ys[0],k)
                    M=M*mask
#etapeC:
            if newC:
                interL=np.empty((9,9),dtype=set)
                interC=np.empty((9,9),dtype=set)
                numL=np.empty((9,9),dtype='int32')
                numC=np.empty((9,9),dtype='int32')
                newC=0
            for Bx in range(3):
                for By in range(3):
                    for Lx in range(3):
                        C=3*Bx+Lx
                        B=Bx+3*By
                        Atemp=M[Bx,By,Lx]
                        Afilter=np.sum(Atemp,axis=-1).reshape((3,1))
                        Atemp=Atemp*(Afilter!=1)
                
                        j,k=np.where(Atemp==1)
                        j=set(j)
                        possibilities=set(k)
                        numC[B,C]=len(j)
                        interC[B,C]=possibilities
                    for Ly in range(3):
                        L=3*By+Ly
                        B=Bx+3*By
                        Atemp=M[Bx,By,:,Ly]
                        Afilter=np.sum(Atemp,axis=-1).reshape((3,1))
                        Atemp=Atemp*(Afilter!=1)
                        j,k=np.where(Atemp==1)
                        j=set(j)
                        possibilities=set(k)
                        numL[B,L]=len(j)
                        interL[B,L]=possibilities

            AC=np.sum(M,axis=-1)
            if 0 in AC:
                break
            for Bx in range(3):
                for By in range(3):
                    for Lx in range(3):
                        C=3*Bx+Lx
                        B=Bx+3*By
                        if numC[B,C]==len(interC[B,C]):
                            yCoordinates,=np.where(AC[Bx,By,Lx]!=1)
                            if len(yCoordinates)==0 or len(interC[B,C])==0:
                                continue
                            M[np.ix_([Bx],[By],range(3),range(3),list(interC[B,C]))]=np.zeros((1,1,3,3,len(list(interC[B,C]))))
                            M[np.ix_([Bx],range(3),[Lx],range(3),list(interC[B,C]))]=np.zeros((1,3,1,3,len(list(interC[B,C]))))
                            M[np.ix_([Bx],[By],[Lx],yCoordinates,list(interC[B,C]))]=np.ones([len(yCoordinates),len(list(interC[B,C]))]).reshape((1,1,1,len(yCoordinates),len(list(interC[B,C]))))
                    for Ly in range(3):
                        L=3*By+Ly
                        B=Bx+3*By
                        if numL[B,L]==len(interL[B,L]):
                            xCoordinates,=np.where(AC[Bx,By,:,Ly]!=1)
                            if len(xCoordinates)==0 or len(interL[B,L])==0:
                                continue
                            M[np.ix_([Bx],[By],range(3),range(3),list(interL[B,L]))]=np.zeros((1,1,3,3,len(list(interL[B,L]))))
                            M[np.ix_(range(3),[By],range(3),[Ly],list(interL[B,L]))]=np.zeros((3,1,3,1,len(list(interL[B,L]))))
                            M[np.ix_([Bx],[By],xCoordinates,[Ly],list(interL[B,L]))]=np.ones([len(xCoordinates),len(list(interL[B,L]))]).reshape((1,1,len(xCoordinates),1,len(list(interL[B,L]))))
        A=np.sum(M,axis=-1)
        detectHorizontaly=np.sum(M,axis=(0,2))
        detectVerticaly=np.sum(M,axis=(1,3))
        for i in range(3):
            for j in range(3):
                B=np.sum(M[i,j],axis=(0,1))
                if 0 in B:
                    return(-1,savingM)
        if 0 in A or 0 in detectHorizontaly or 0 in detectVerticaly:
            return(-1,savingM)
        if depth==0:
            return(0,savingM)
            return(1,M)
        dirtyM=M
        savingSum2=np.sum(M)
        A=np.sum(M,axis=-1)
        hypothesis=np.where(A!=1)
        aL,bL,cL,dL=hypothesis
        aL=list(aL)
        bL=list(bL)
        cL=list(cL)
        dL=list(dL)
        values=A[hypothesis[0],hypothesis[1],hypothesis[2],hypothesis[3]].tolist()
        #values=A.tolist()
        while len(values)!=0:
            xMin=np.argmin(values)
            a=aL.pop(xMin)
            b=bL.pop(xMin)
            c=cL.pop(xMin)
            d=dL.pop(xMin)
            values.pop(xMin)
            #une remarque est nécessaire ici: on traite les valeurs ici sur une case (celle qui a le moins de possibilités)
            #il se peutdonc que la derniere valeur traitée soit forcée par elimination des autres et en soit déductible
            #elle sera ici malgré tout considéré comme une hypothese et ca impactera la profondeur maximale de l'arbre des recherches.
            kL=np.where(M[a,b,c,d]==1)[0]
            for key in kL:
                mask=ex.getMask(a,b,c,d,key)
                dirtyM=dirtyM*mask
                #print('hypothese (',3*a+c,',',3*b+d,')=',key)
                worked,M2=roll(dirtyM,depth-1)
                if worked==1:
                    #print('testing hypothese (',3*a+c,',',3*b+d,')=',key)
                    #print('worked',np.sum(M))  
                    return(worked,M2)
                if worked==-1:
                    #print('failed')
                    M[a,b,c,d,key]=0
                    values=[]
                    loop=1
                    break
    return(0,M)