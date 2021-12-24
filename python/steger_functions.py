# -*- coding: utf-8 -*-
"""
Steger function set

Created on Thu Oct 17 17:22:04 2019

@author: LAB
"""
import numpy as np
import skimage.feature as sf
import skimage.filters as fil
import matplotlib.pyplot as plt
import steger_functions as stfn
from PIL import Image #Чтобы в .bmp сохранять
from joblib import Parallel, delayed
import time
from scipy import interpolate

#def first_max_area(xy_l, v, Ev_max):
##    Ф-я находит некоторую область, перпендикулярную к линии, в которой стоит
##    искать первую точку линии
#    ix = np.abs(Ev_max[xy_l[:,0],xy_l[:,1]]).argmax()

def steger_line(T,sigma_line,sz,finding_area=1,col=None):
    maxrat_max = 4
    Helems = sf.hessian_matrix(T, sigma=sigma_line, 
                               mode='reflect',order='xy');
    Hxx, Hxy = Helems[0], Helems[1]
    
    Ev = sf.hessian_matrix_eigvals(Helems)
    Ev_max = Ev[1]*finding_area
    
    sz = 256;
    v  = np.zeros((2,sz,sz))
    v0 = np.zeros((2,sz,sz))
    
    #А вот другие получаются собственные вектора!
    tmpMax = (Hxx-Ev_max)**2 + Hxy**2
    res90 = tmpMax == 0.
    v[1,:,:] = (Hxx-Ev_max)/np.sqrt( tmpMax )
    v[0,:,:] = -Hxy/np.sqrt( tmpMax )
    
    tmpMin = (Hxx-Ev[0])**2 + Hxy**2
    v0[1,:,:] = (Hxx-Ev[0])/np.sqrt( tmpMin )
    v0[0,:,:] = -Hxy/np.sqrt( tmpMin )
    
    v[1,res90] = 1
    v[0,res90] = 0
    
#    atan = np.arctan(v[1,:,:]/v[0,:,:])/np.pi*180
    
    
    #добавим, чтобы максимум находился где-нибудь в центре
    inmax = np.full((sz,sz), False)
    inmax[20:236,20:200] = True
    
#    e1,e2 = np.unravel_index(np.abs(Ev_max*inmax).argmax(),Ev_max.shape )
    
#    if (col != None):        
#        e1,e2 = col,int( np.unravel_index(np.abs(Ev_max[20:236,col-20]).argmax()
#                                                  ,Ev_max[20:236,col-20].shape )[0] )
    if (col != None):        
#        e1,e2 = col,int( np.unravel_index(np.abs(Ev_max[col,20:200]).argmax()
#                                                  ,Ev_max[col,20:200].shape )[0] )
        e1,e2 = int( 20+np.abs(Ev_max[20:236,col]).argmax() ),col
    else:
        e1,e2 = np.unravel_index(np.abs(Ev_max*inmax).argmax(),Ev_max.shape )
    
    tana = v0[1,e1,e2]/v0[0,e1,e2] #tana1 = v[1,e1,e2]/v[0,e1,e2]
    
    sq2 = 1/np.sqrt(2)
        
    M = np.array([[ 1., 1/sq2, 0., -1/sq2, -1., -1/sq2,  0.,  1/sq2 ],
                  [ 0., 1/sq2, 1.,  1/sq2, -0., -1/sq2, -1., -1/sq2 ]])
                  # 0    45    90   135    180   -135   -90    -45
    
    inline = np.full((sz,sz), False) #тут будут пиксели линии
        
    inline[e1][e2] = True
    
    nx,ny = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))    
    #np.cos(np.arctan(tana1)),np.sin(np.arctan(tana1))
    
    ind = 16    
    first_branch  = np.zeros((1200,2),dtype=np.int32)
    second_branch = np.zeros((1200,2),dtype=np.int32)    
    
    #Собираем в линию    
    
    maxiter = stfn.link_line(v0,M,Ev_max,inline,first_branch,sz,ind,
                        maxrat_max,*(e1,e2,nx,ny))    
#    plt.figure(2); plt.imshow(inline,'gray');
    
    first_branch = np.delete(first_branch, range(maxiter,1200), axis=0)
        
    nx,ny = -nx,-ny

    maxiter = stfn.link_line(v0,M,Ev_max,inline,second_branch,sz,ind,
                        maxrat_max,*(e1,e2,nx,ny))
    
    second_branch = np.delete(second_branch, range(maxiter,1200), axis=0)
    
    return inline, first_branch, second_branch, v, Ev_max

def link_line(v0,M,Ev_max,inline,branch_xy,sz,ind,maxrat_max,*ep):
    mx = 0 #Максимальное количество итераций
    maxrat = 1 #Отношение максимумов
    
    e1 = ep[0]
    e2 = ep[1]
    inline[e1][e2] = True
    branch_xy[mx,:] = e1,e2
    mx += 1
    
    maxEv = np.abs(Ev_max[e1,e2])
    
    nx, ny = ep[2],ep[3]
    n = np.array([nx,ny])/np.sqrt(nx**2+ny**2)
    
    
    while ( (e1 != ind) and (e1 != sz-ind) and (e2 != ind) 
        and (e2 != sz-ind) and (maxrat<maxrat_max) and (mx<1000)):    
        
        scp = n @ M #scalar product    
        #orientations list 
        o_list = (-scp).argsort()[:3] #индексы трех максимальных значений. Т.к.argsort
                                      #сортирует от меньшего, то вставлен минус
        e1, e2 = stfn.new_dot(e1, e2, o_list, Ev_max)
        maxrat = maxEv/np.abs(Ev_max[e1,e2])
        inline[e1][e2] = True
        branch_xy[mx,:] = e1,e2
        tana = v0[1,e1,e2]/v0[0,e1,e2]
        
        nx,ny = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
#        if nx*nx2+ny*ny2 >= 0: #Это чтобы в обратную сторону не поворачивало
#            nx,ny = nx2,ny2
#        else:
#            nx,ny = -nx2,-ny2
#        n = np.array([nx,ny])/np.sqrt(nx**2+ny**2)
        
        mx += 1
    
    return mx

def intersection_point(inline,Xstrt,Xend,lnorm_float,Xcn):
    #inline - координаты линии, Xs - start, Xe - end
    #Тут находим начальную и конечную позиции линии границы, которая лежит в рам-
    #ках нашей линиии.
    if (np.abs(Xstrt-Xend) < 2):
        Xc = int( np.around( np.abs(Xstrt+Xend)/2 ) )
        Ytmp =  np.argwhere(inline[:,1] == Xc)
        if Ytmp == None:
            return None,None
        Yc = int( inline[Ytmp,0] )
        return Xc, Yc
    
    Xs = np.abs(inline[:,1] - Xstrt).argmin()
    Xe = np.abs(inline[:,1] - Xend).argmin()
    if (np.abs(Xs - Xe) < 2):
        return None, None
    Xsl = np.abs(Xcn - inline[Xs,1]).argmin()
    Xel = np.abs(Xcn - inline[Xe,1]).argmin()
    
    LlineX = Xcn[Xsl:Xel]
    LlineY = lnorm_float[Xsl:Xel]

    lng2 = LlineX.shape[0]
    
    
#    Tmp = np.full((256,256), False)
    
    #Находим значения Y и далее его интерполируем
    lng1 = Xe-Xs
    Yline = inline[Xs:Xe:np.sign(lng1),0]
    Xline = inline[Xs:Xe:np.sign(lng1),1]
    
    
    lng1 = np.abs(lng1)
        
    f = interpolate.interp1d(np.arange(0,lng1)
                                               ,Yline,fill_value="extrapolate")
    Ynew = f(np.linspace(0,lng1,lng2))
    f = interpolate.interp1d(np.arange(0,lng1)
                                               ,Xline,fill_value="extrapolate")    
    Xnew = f(np.linspace(0,lng1,lng2))
#    Tmp[np.int32(LlineY),np.int32(LlineX)] = 1
##    Tmp[np.int32(lnorm_float),np.int32(Xcn)] = 1
#    Tmp[np.int32(Ynew),np.int32(Xnew)] = 1
#    plt.figure(5); plt.imshow(Tmp,'gray'); plt.pause(5)
    #Находим разницу между игриками, находим ближайшую точку перед пересечением
    delta_line = Ynew-LlineY
    ix = np.where(delta_line[1:] * delta_line[:-1] < 0)[0]
    if (ix.size == 0):
        Xc, Yc = None, None
    elif (ix.size > 1):
        Xc, Yc = None, None
    else:
        deltaX = delta_line[ix]/(delta_line[ix]-delta_line[ix+1])
        deltaY = deltaX*(Ynew[ix] - Ynew[ix+1])
        # Yline.shape/128 - это множитель, т.к. мы ищем номер в массиве, а размер 
        # элемента массива изменился
#        Xc = ix*Yline.shape/128 + Xs + deltaX
        Xc = np.int32(Xnew[ix]) + deltaX
        Yc = Ynew[ix]-deltaY
    return Xc, Yc

def intersection_point2(inline,Xs,Xe,lnorm_float):
    #inline - координаты линии, Xs - start, Xe - end
    #Тут находим начальную и конечную позиции линии границы, которая лежит в рам-
    #ках нашей линиии.
    if (Xs < inline[0,1]):
        strt = 0
    elif (inline[-1,1] <= Xs ):
        return None, None        
    else:
        strt = int( np.argwhere(inline[:,1] == Xs) )
    
    if (Xe > inline[-1,1]):
        stp = inline.shape[0]-1
    elif (Xe <= inline[0,1] ):
        return None, None
    else:
        stp =  int( np.argwhere(inline[:,1] == Xe) )  
        
    if (Xe <= Xs):
        return None, None
    #Находим значения Y и далее его интерпалируем
    Yline = inline[strt:stp+1,0]    
    f = interpolate.interp1d(np.arange(0,stp+1-strt),Yline,fill_value="extrapolate")
    Ynew = f(np.linspace(0,stp-strt,128))
    #Находим разницу между игриками, находим ближайшую точку перед пересечением
    delta_line = Ynew-lnorm_float
    ix = np.where(delta_line[1:] * delta_line[:-1] < 0)[0]
    if (ix.size == 0):
        Xc, Yc = None, None
    elif (ix.size > 1):
        Xc, Yc = None, None
    else:
        deltaX = delta_line[ix]/(delta_line[ix]-delta_line[ix+1])
        deltaY = deltaX*(Ynew[ix] - Ynew[ix+1])
        # Yline.shape/128 - это множитель, т.к. мы ищем номер в массиве а размер 
        # элемента массива изменился
        Xc = ix*Yline.shape/128 + Xs + deltaX
        Yc = Ynew[ix]-deltaY
    return Xc, Yc

def unique_defined(first_branch,second_branch,sz=None):
    #Принимает две ветки значений точек координат. Делает взаимно-однозначную
    #функцию и возвращает ее координаты, а заодно и двумерный булевый массив
    inline_coord = np.concatenate((first_branch,second_branch))
    inline_coord = inline_coord[inline_coord[:,1].argsort()]    
    inline_coord = inline_coord[np.unique(inline_coord[:,1], return_index=True)[1],:]
    if (sz != None):
        inline = np.full((sz,sz), False)
        inline[inline_coord[:,0],inline_coord[:,1]] = True
        return inline_coord, inline
    else:
        return inline_coord, None

def new_dot(e1, e2, o_list, Ev_max):
    # Т.е. я придумал вот такую конструкцию проверки по списку, но она мне не 
    # нарвится
    ti = 0                        #temp i
    en = np.zeros((2,3), dtype=int)
    if 0 in o_list:
        en[0,ti] = e1+1    
        en[1,ti] = e2
        ti = ti+1
    if 1 in o_list:
        en[0,ti] = e1+1    
        en[1,ti] = e2+1
        ti = ti+1
    if 2 in o_list:
        en[0,ti] = e1
        en[1,ti] = e2+1
        ti = ti+1
    if 3 in o_list:
        en[0,ti] = e1-1    
        en[1,ti] = e2+1
        ti = ti+1
    if 4 in o_list:
        en[0,ti] = e1-1    
        en[1,ti] = e2
        ti = ti+1
    if 5 in o_list:
        en[0,ti] = e1-1    
        en[1,ti] = e2-1
        ti = ti+1
    if 6 in o_list:
        en[0,ti] = e1    
        en[1,ti] = e2-1
        ti = ti+1
    if 7 in o_list:
        en[0,ti] = e1+1    
        en[1,ti] = e2-1
        ti = ti+1
    
    tmp = np.zeros((3,1))
    tmp[0] = np.abs( Ev_max[en[0,0]][en[1,0]] )
    tmp[1] = np.abs( Ev_max[en[0,1]][en[1,1]] )
    tmp[2] = np.abs( Ev_max[en[0,2]][en[1,2]] )
#    tmp[0] = np.abs( Ev_max[en[0,0]][en[1,0]] )*np.sqrt((en[0,0]-e1)**2+(en[1,0]-e2)**2)
#    tmp[1] = np.abs( Ev_max[en[0,1]][en[1,1]] )*np.sqrt((en[0,1]-e1)**2+(en[1,1]-e2)**2)
#    tmp[2] = np.abs( Ev_max[en[0,2]][en[1,2]] )*np.sqrt((en[0,2]-e1)**2+(en[1,2]-e2)**2)
    
    max_pixel = (-tmp.T).argsort().T[0] #C .T пришлось изгаляться т.к. argsort()
                                        #так работает
    max_pixel = max_pixel[0].item()
    en1 = np.zeros(1)
    en2 = en1.copy()
    
    if max_pixel == 0:
        en1,en2 = en[:,0]
    if max_pixel == 1:
        en1,en2 = en[:,1]
    if max_pixel == 2:
        en1,en2 = en[:,2]
    return en1, en2

def steger(filepath,filename):
    #filepath - Папка с данными
    #filename - Имя файла в этой папке

    #Важные параметры:
    #Это примерная ширина линии.
    sigma_line = 12
    #Чем меньше эта сигма, тем ближе к  истинной границе располагается нахо-
    #димая граница, но тем выше вероятность всяких разрывов
    sigma_border = 3.5
    #(Длина полулиний)/2 (от inline в сторону одной из нормальей), вдоль которой 
    #ищется максимум
    sig_lenght = 8
    #Отношение максимального и текущего значений собственного векторов, при 
    #при котором построение линии прекращается. Чем больше это значение, тем
    #менее яркие линии будут находиться
    maxrat_max = 4     
    
    A = np.fromfile(filepath+filename,dtype='complex64') # прочитать файл с комплексными данными
    y = 1024
    x = 256
    z = 256
    B = np.reshape(A,(y,x,z))
    C = np.transpose(B, axes = (1,2,0)) # замена осей на (x,z,y)
    dist_size = 500
    distmap = np.zeros((dist_size,256))
    #Тут сохраняются позиции, где толщины !=0, нужно для усреднения (на что делить)
    bool_dist = np.full((dist_size,256), False)
    
    #Выбранные 4 среза для отображения 
    chosen_slices = np.zeros((4,256,256))
    ti = 0
    savelist = np.array([128,256,512,968])
    
    for Y0 in range(0,1000,2):        
        # сечение по "быстрой" оси на линии Y0
        dlog = 20*np.log10(np.abs(C[:,:,Y0+12])+10**-6)
        #первая строчка была -140
        dlog[0][:] = dlog[1][:].sum()/dlog[1][:].shape[0] 
        
        T = np.double(dlog)
        
        Helems = sf.hessian_matrix(T, sigma=sigma_line, 
                                   mode='reflect',order='xy');
        Hxx, Hxy = Helems[0], Helems[1]
        
        Ev = sf.hessian_matrix_eigvals(Helems)
        Ev_max = Ev[1]
        
        sz = 256;
        v  = np.zeros((2,sz,sz))
        v0 = np.zeros((2,sz,sz))
        
        #А вот другие получаются собственные вектора!
        tmpMax = (Hxx-Ev_max)**2 + Hxy**2
        res90 = tmpMax == 0.
        v[1,:,:] = (Hxx-Ev_max)/np.sqrt( tmpMax )
        v[0,:,:] = -Hxy/np.sqrt( tmpMax )
        
        tmpMin = (Hxx-Ev[0])**2 + Hxy**2
        v0[1,:,:] = (Hxx-Ev[0])/np.sqrt( tmpMin )
        v0[0,:,:] = -Hxy/np.sqrt( tmpMin )
        
        v[1,res90] = 1
        v[0,res90] = 0
        
        atan = np.arctan(v[1,:,:]/v[0,:,:])/np.pi*180
        
        
        #добавим, чтобы максимум находился где-нибудь в центре
        inmax = np.full((sz,sz), False)
        inmax[20:236,20:200] = True
        
        e1,e2 = np.unravel_index(np.abs(Ev_max*inmax).argmax(),Ev_max.shape )
        maxEv = np.abs(Ev_max[e1,e2]) #Максимальное значение
        
        tana = v0[1,e1,e2]/v0[0,e1,e2]
        
        e1_s = e1.copy() #e1_start
        e2_s = e2.copy()
            
        sq2 = 1/np.sqrt(2)
            
        M = np.array([[ 1., sq2, 0., -sq2,  0., -sq2,  0.,  sq2 ],
                      [ 0., sq2, 1.,  sq2, -1., -sq2, -1., -sq2 ]])
                      # 0    45  90   135  180  -135  -90   -45
        
        inline = np.full((sz,sz), False) #тут будут пиксели линии
        
        
        inline[e1][e2] = True
        
        nx,ny = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
        n = np.array([nx,ny])
        
        ind = 16
        maxrat = 1
        
        maxiter = 0 #Максимальное количество итераций
        while ( (e1 != ind) and (e1 != sz-ind) and (e2 != ind) 
            and (e2 != sz-ind) and (maxrat<maxrat_max) and (maxiter<500)):
        
            scp = n @ M #scalar product    
            #orientations list 
            o_list = (-scp).argsort()[:3] #индексы трех максимальных значений. Т.к.argsort
                                          #сортирует от меньшего, то вставлен минус
            e1, e2 = stfn.new_dot(e1, e2, o_list, Ev_max)
            maxrat = maxEv/np.abs(Ev_max[e1,e2])
            inline[e1][e2] = True
            tana = v0[1,e1,e2]/v0[0,e1,e2]
            
            nx2,ny2 = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
            if nx*nx2+ny*ny2 >= 0: #Это чтобы в обратную сторону не поворачивало
                nx,ny = nx2,ny2
            else:
                nx,ny = -nx2,-ny2
            n = np.array([nx,ny])
            
            maxiter = maxiter+1
            
        e1 = e1_s
        e2 = e2_s
        maxrat = 1
        tana = v0[1,e1,e2]/v0[0,e1,e2]
        nx,ny = -np.cos(np.arctan(tana)),-np.sin(np.arctan(tana))
        n = np.array([nx,ny])
        
        maxiter = 0 #Максимальное количество итераций
        while ((e1 != ind) and (e1 != sz-ind) and (e2 != ind) 
            and (e2 != sz-ind) and (maxrat<maxrat_max) and (maxiter<500)):
        
            scp = n @ M #scalar product    
            #orientations list 
            o_list = (-scp).argsort()[:3] #индексы трех максимальных значений. Т.к.argsort
                                          #сортирует от меньшего, то вставлен минус
            e1, e2 = stfn.new_dot(e1, e2, o_list, Ev_max)
            maxrat = maxEv/np.abs(Ev_max[e1,e2])
            inline[e1][e2] = True
            tana = v0[1,e1,e2]/v0[0,e1,e2]
            
            nx2,ny2 = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
            if nx*nx2+ny*ny2 >= 0:
                nx,ny = nx2,ny2
            else:
                nx,ny = -nx2,-ny2
            n = np.array([nx,ny])
            
            maxiter = maxiter+1
        
        #Ищем границы
            
        T = dlog        
        
        linecoord = np.argwhere(inline == True)
        
        borders = np.full((sz,sz), False) #тут будут пиксели границ
        
        gT = fil.gaussian(T,sigma=sigma_border)  #Свертка границы
        gTx = np.gradient(gT, axis=0)
        gTx[0:15][:] = 0
        gTy = np.gradient(gT, axis=1)
        gTy[0:15][:] = 0
        gTsqrt = np.sqrt(gTx**2+gTy**2)
        
        tline = np.zeros((4*np.round(sig_lenght),1))
        
        dist = np.zeros(256) #distribution right
        
        for i in range(linecoord.shape[0]):
            xt = linecoord[i,0]
            yt = linecoord[i,1]
            atan = np.arctan(v[1,xt,yt]/v[0,xt,yt])
            cosa = np.cos(atan)
            sina = np.sin(atan)
            rr = round(2*sig_lenght)
            for j in range(0,rr,1):
                xt1 = xt + np.int(np.round(j*cosa))
                yt1 = yt + np.int(np.round(j*sina))
                if (xt1>255 or xt1<0 or yt1>255 or yt1<0):
                    tline[j] = 0
                else:
                    tline[j] = gTsqrt[xt1][yt1]
                
            nm = (-np.abs(tline).T).argsort().T[0] #index of maximum 
            mc1 = xt + np.int(np.round(nm*cosa)), yt + np.int(np.round(nm*sina)) #max indexes
            borders[mc1[0],mc1[1]] = True
            
            for j in range(-rr,0,1):
                xt1 = xt + np.int(np.round(j*cosa))
                yt1 = yt + np.int(np.round(j*sina))
                if (xt1>255 or xt1<0 or yt1>255 or yt1<0):
                    tline[j+rr] = 0
                else:
                    tline[j+rr] = gTsqrt[xt1][yt1]
            
            nm = (-np.abs(tline).T).argsort().T[0] #number of maximum 
            nm = nm - rr
            mc2 = xt + np.int(np.round(nm*cosa)), yt + np.int(np.round(nm*sina))
            #max coordinates
            borders[mc2[0],mc2[1]] = True
            
            dist[yt] = np.sqrt((mc2[0]-mc1[0])**2+(mc2[1]-mc1[1])**2)
        
        #Сохраняем избранные слои с границами
        if Y0 in savelist:
           chosen_slices[ti,:] = T + 120*inline + 120*borders
           ti = ti+1
        
        #boolean массив, где есть толщина перепонки
        bool_dist[np.int16(Y0/2),:] = np.not_equal(dist, 0)
        
        #фильтр низких частот
        fdist = np.fft.fft(dist)
        fdist[18:237] = 0
        dist = np.real( np.fft.ifft(fdist) )
        
        distmap[np.int16(Y0/2),:] = dist
    
    distmap = (distmap*3200/256/1.44).T
        
    #Вычислим стреднюю толщину перепонки, дисперсию, гистограмму
    norm_thick = np.uint8(bool_dist).sum()
    dit = distmap[bool_dist.T] #distmap temp
    av_thickness = dit.sum()/norm_thick
    #среднеквадратичное отклонение
    stdv_thick = np.sqrt( ((dit - av_thickness)**2).sum()/norm_thick )
    
    
    #Выводим карту толщины
    plt.figure(figsize=(12,10))
    plt.subplot(321,title='толщина, мкм') # 3 строки, 2 столбца, 1й график
    plt.imshow(distmap,'gray')
    plt.colorbar()
    plt.xticks([64,128,256,484]) #Это отображаемые слои
    
    #Выводим гистограмму с СКО и СР
    tmpstr = ('Ср. толщина = '+str(np.int16(av_thickness))+
                                    ' \n'+'СКО = '+str(np.int16(stdv_thick))+
                                    '\n Глбн избр = 3200'+
                                    '\n n = 1.44')
    plt.subplot(322, ylabel='кол-во элементов', 
                title=filename)
    plt.hist(dit,bins=100, label=tmpstr)
    plt.yticks([]) #убрали подписи на y'ке
    plt.legend()
    
    #Добавляем субграфики
    plt.subplot(323,title='слой '+str(savelist[0]))
    plt.imshow(chosen_slices[0,:],'gray')
    plt.subplot(324,title='слой '+str(savelist[1]))
    plt.imshow(chosen_slices[1,:],'gray')
    plt.subplot(325,title='слой '+str(savelist[2]))
    plt.imshow(chosen_slices[2,:],'gray')
    plt.subplot(326,title='слой '+str(savelist[3]))
    plt.imshow(chosen_slices[3,:],'gray')
    #plt.text(550, 1.5*10**4, tmpstr, fontsize=12, horizontalalignment='center', 
    #         verticalalignment='center')
    
    #Сохраняем в jpg и закрываем
    plt.savefig(filepath+filename[:-4]+'_.jpg', dpi=600)
    plt.close()
    
    # Сохраняем в .bmp как бы ужимая 4 пикселя в 1
    tmp = np.uint8(np.abs(distmap/4))
    img = Image.fromarray(tmp)
    img.save(filepath+filename[:-3]+'bmp')
    
def steger2(filepath,filename):
    #filepath - Папка с данными
    #filename - Имя файла в этой папке
    A = np.fromfile(filepath+filename,dtype='complex64') # прочитать файл с комплексными данными
    y = 1024
    x = 256
    z = 256
    B = np.reshape(A,(y,x,z))
    C = np.transpose(B, axes = (1,2,0)) # замена осей на (x,z,y)
    dist_size = 500
    distmap = np.zeros((dist_size,256))
    #Тут сохраняются позиции, где толщины !=0, нужно для усреднения (на что делить)
    bool_dist = np.full((dist_size,256), False)
    
    #Выбранные 4 среза для отображения 
    chosen_slices = np.zeros((8,256,256))    
    savelist = np.array([32,64,128,192,256,320,384,448])
    C = stfn.zeroing_spectrum_ends(C) #Убираем выхилы
    
#    t1 = time.process_time()
#    
#    Parallel(n_jobs=4, backend='multiprocessing')(delayed(main_circ)(Y0,distmap,bool_dist,C,chosen_slices,savelist)
#                                                    for Y0 in range(0,1000,2))
#    print(t1 - time.process_time())
    for Y0 in range(0,1000,2):
        main_circ(Y0,distmap,bool_dist,C,chosen_slices,savelist)
    
    
    distmap = (distmap*3200/256/1.44).T
        
    #Вычислим стреднюю толщину перепонки, дисперсию, гистограмму
    norm_thick = np.uint8(bool_dist).sum()
    dit = distmap[bool_dist.T] #distmap temp
    av_thickness = dit.sum()/norm_thick
    #среднеквадратичное отклонение
    stdv_thick = np.sqrt( ((dit - av_thickness)**2).sum()/norm_thick )
    
    
    #Выводим карту толщины
    plt.figure(figsize=(12,10))
    plt.subplot2grid((3,4),(0,0), colspan=2, title='толщина, мкм') # 3 строки, 2 столбца, 1й график
    plt.imshow(distmap,'gray')
    plt.colorbar()
    plt.xticks(savelist) #Это отображаемые слои
    
    #Выводим гистограмму с СКО и СР
    tmpstr = ('Ср. толщина = '+str(np.int16(av_thickness))+
                                    ' \n'+'СКО = '+str(np.int16(stdv_thick))+
                                    '\n Глбн избр = 3200'+
                                    '\n n = 1.44')
    plt.subplot2grid((3,4),(0,2), colspan=2, ylabel='кол-во элементов', 
                title=filename)
    plt.hist(dit,bins=100, label=tmpstr)
    plt.yticks([]) #убрали подписи на y'ке
    plt.legend()
    
    ##Добавляем субграфики
    for i in range(0,2):
        for j in range(0,4):
            plt.subplot2grid((3,4),(1+i,j),title='слой '+str(savelist[4*i+j]))
            plt.imshow(chosen_slices[4*i+j,:],'gray')
            plt.yticks([]) #убрали подписи на y'ке
            plt.xticks([0,50,100,150,200,250]) 
            
    
    #plt.text(550, 1.5*10**4, tmpstr, fontsize=12, horizontalalignment='center', 
    #         verticalalignment='center')
    
    #Сохраняем в jpg и закрываем
    plt.savefig(filepath+filename[:-4]+'.jpg', dpi=400)
    plt.close()
    
    # Сохраняем в .bmp как бы ужимая 4 пикселя в 1
    tmp = np.uint8(np.abs(distmap/4))
    img = Image.fromarray(tmp)
    img.save(filepath+filename[:-3]+'bmp')
#Функция берет 3D массив и убирает из вертикальные выхилы

def main_circ(Y0,distmap,bool_dist,C,chosen_slices,savelist):
    #Важные параметры:
    #Это примерная ширина линии.
    sigma_line = 20
    #Чем меньше эта сигма, тем ближе к  истинной границе располагается нахо-
    #димая граница, но тем выше вероятность всяких разрывов
    sigma_border = 3
    #(Длина полулиний)/2 (от inline в сторону одной из нормальей), вдоль которой 
    #ищется максимум
    sig_lenght = sigma_line
    #Отношение максимального и текущего значений собственного векторов, при 
    #при котором построение линии прекращается. Чем больше это значение, тем
    #менее яркие линии будут находиться
    maxrat_max = 4  
    is_weight = True #Применять веса при построении границ или нет   
    # сечение по "быстрой" оси на линии Y0    
    ctemp = C[:,:,Y0+12]
    ctemp[0][:] = ctemp[1][:]
    dlog = 20*np.log10(np.abs(ctemp)+10**-6)    
    #первая строчка была -140
    #dlog[0][:] = dlog[1][:].sum()/dlog[1][:].shape[0] 
    
    T = np.double(dlog)
    
#    plt.figure(3); plt.imshow(dlog+120*inline+120*borders,'gray')
#    plt.pause(0.1)
    
    Helems = sf.hessian_matrix(T, sigma=sigma_line, 
                               mode='reflect',order='xy');
    Hxx, Hxy = Helems[0], Helems[1]
    
    Ev = sf.hessian_matrix_eigvals(Helems)
    Ev_max = Ev[1]
    
    sz = 256;
    v  = np.zeros((2,sz,sz))
    v0 = np.zeros((2,sz,sz))
    
    #А вот другие получаются собственные вектора!
    tmpMax = (Hxx-Ev_max)**2 + Hxy**2
    res90 = tmpMax == 0.
    v[1,:,:] = (Hxx-Ev_max)/np.sqrt( tmpMax )
    v[0,:,:] = -Hxy/np.sqrt( tmpMax )
    
    tmpMin = (Hxx-Ev[0])**2 + Hxy**2
    v0[1,:,:] = (Hxx-Ev[0])/np.sqrt( tmpMin )
    v0[0,:,:] = -Hxy/np.sqrt( tmpMin )
    
    v[1,res90] = 1
    v[0,res90] = 0
    
#    atan = np.arctan(v[1,:,:]/v[0,:,:])/np.pi*180
    
    
    #добавим, чтобы максимум находился где-нибудь в центре
    inmax = np.full((sz,sz), False)
    inmax[20:236,20:200] = True
    
    e1,e2 = np.unravel_index(np.abs(Ev_max*inmax).argmax(),Ev_max.shape )
    maxEv = np.abs(Ev_max[e1,e2]) #Максимальное значение
    
    tana = v0[1,e1,e2]/v0[0,e1,e2]
    
    e1_s = e1.copy() #e1_start
    e2_s = e2.copy()
        
    sq2 = 1/np.sqrt(2)
        
    M = np.array([[ 1., sq2, 0., -sq2,  0., -sq2,  0.,  sq2 ],
                  [ 0., sq2, 1.,  sq2, -1., -sq2, -1., -sq2 ]])
                  # 0    45  90   135  180  -135  -90   -45
    
    inline = np.full((sz,sz), False) #тут будут пиксели линии
    
    
    inline[e1][e2] = True
    
    nx,ny = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
    n = np.array([nx,ny])
    
    ind = 16
    maxrat = 1
    first_branch  = np.zeros((500,2),dtype=np.int32)
    second_branch = np.zeros((500,2),dtype=np.int32)
    
    maxiter = 0 #Максимальное количество итераций
    while ( (e1 != ind) and (e1 != sz-ind) and (e2 != ind) 
        and (e2 != sz-ind) and (maxrat<maxrat_max) and (maxiter<500)):
    
        scp = n @ M #scalar product    
        #orientations list 
        o_list = (-scp).argsort()[:3] #индексы трех максимальных значений. Т.к.argsort
                                      #сортирует от меньшего, то вставлен минус
        e1, e2 = stfn.new_dot(e1, e2, o_list, Ev_max)
        maxrat = maxEv/np.abs(Ev_max[e1,e2])
        inline[e1][e2] = True
        first_branch[maxiter,:] = e1,e2
        tana = v0[1,e1,e2]/v0[0,e1,e2]
        
        nx2,ny2 = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
        if nx*nx2+ny*ny2 >= 0: #Это чтобы в обратную сторону не поворачивало
            nx,ny = nx2,ny2
        else:
            nx,ny = -nx2,-ny2
        n = np.array([nx,ny])
        
        maxiter = maxiter+1
    
    first_branch = np.delete(first_branch, range(maxiter,500), axis=0)
        
    e1 = e1_s
    e2 = e2_s
    maxrat = 1
    tana = v0[1,e1,e2]/v0[0,e1,e2]
    nx,ny = -np.cos(np.arctan(tana)),-np.sin(np.arctan(tana))
    n = np.array([nx,ny])
    
    maxiter = 0 #Максимальное количество итераций
    while ((e1 != ind) and (e1 != sz-ind) and (e2 != ind) 
        and (e2 != sz-ind) and (maxrat<maxrat_max) and (maxiter<500)):
    
        scp = n @ M #scalar product    
        #orientations list 
        o_list = (-scp).argsort()[:3] #индексы трех максимальных значений. Т.к.argsort
                                      #сортирует от меньшего, то вставлен минус
        e1, e2 = stfn.new_dot(e1, e2, o_list, Ev_max)
        maxrat = maxEv/np.abs(Ev_max[e1,e2])
        inline[e1][e2] = True
        second_branch[maxiter,:] = e1,e2
        tana = v0[1,e1,e2]/v0[0,e1,e2]
        
        nx2,ny2 = np.cos(np.arctan(tana)),np.sin(np.arctan(tana))
        if nx*nx2+ny*ny2 >= 0:
            nx,ny = nx2,ny2
        else:
            nx,ny = -nx2,-ny2
        n = np.array([nx,ny])
        
        maxiter = maxiter+1
    
    second_branch = np.delete(second_branch, range(maxiter,500), axis=0)
    
    #Ищем границы
   
#    linecoord = np.argwhere(inline == True) #Координаты точек линии
    
    borders = np.full((sz,sz), False) #тут будут пиксели границ
    
    gT = fil.gaussian(T,sigma=sigma_border)  #Свертка границы
    gTx = np.gradient(gT, axis=0)
    gTx[0:15][:] = 0
    gTy = np.gradient(gT, axis=1)
    gTy[0:15][:] = 0
    gTsqrt = np.sqrt(gTx**2+gTy**2)
    
       
    dist = np.zeros(256) #distribution right
    
    #===Тут считаем толщну 
    get_thickness(first_branch,dist,borders,v,sig_lenght,gTsqrt,is_weight)
    get_thickness(second_branch,dist,borders,v,sig_lenght,gTsqrt,is_weight)
        
    if Y0 in savelist:
        ti = np.argwhere(savelist == Y0)
        chosen_slices[ti,:] = T + 120*inline + 120*borders  
    
    #boolean массив, где есть толщина перепонки
    bool_dist[np.int16(Y0/2),:] = np.not_equal(dist, 0)
    
    #фильтр низких частот
    fdist = np.fft.fft(dist)
    fdist[18:237] = 0
    dist = np.real( np.fft.ifft(fdist) )
    
    distmap[np.int16(Y0/2),:] = dist

def get_thickness(linecoord,dist,borders,v,sig_lenght,gTsqrt,is_weight):
    mcOld1 = 128,128
    mcOld2 = 128,128
    tline = np.zeros((4*np.round(sig_lenght),1))    
    for i in range(linecoord.shape[0]):
        xt = linecoord[i,0]
        yt = linecoord[i,1]   
        atan = np.arctan(v[1,xt,yt]/v[0,xt,yt])
        cosa = np.cos(atan)
        sina = np.sin(atan)
        rr = round(2*sig_lenght)
        sr = round(sig_lenght/2) #Чтобы граница не была близко к линии
        
        for j in range(sr,rr,1):
            xt1 = xt + np.int(np.round(j*cosa))
            yt1 = yt + np.int(np.round(j*sina))
            if (xt1>255 or xt1<0 or yt1>255 or yt1<0):
                tline[j] = 0
            else:
                if is_weight:
                    #G = gTsqrt[mcOld1[0],mcOld1[1]] * (i != 0)                 #последний множитель нужен на случай первого запуска цикла               
                    deltaS = np.sqrt( (xt1-mcOld1[0])**2+(yt1-mcOld1[1])**2 )  #Расстояние между текущей точкой и предыдущим максимумом
                    #tline[j] = gTsqrt[xt1][yt1] * G/(1+deltaS)
                    tline[j] = gTsqrt[xt1][yt1] * np.exp(-(deltaS-1)*(i != 0)/3)
                else:
                    tline[j] = gTsqrt[xt1][yt1] #/ (1+deltaS*(i != 0))**2
            
        nm = (-np.abs(tline).T).argsort().T[0] #index of maximum 
        mc1 = xt + np.int(np.round(nm*cosa)), yt + np.int(np.round(nm*sina)) #max indexes
        borders[mc1[0],mc1[1]] = True  #mc1 - координаты максимального градиента
        mcOld1 = mc1
        
        tline = tline*0
        for j in range(-rr,-sr,1):
            xt1 = xt + np.int(np.round(j*cosa))
            yt1 = yt + np.int(np.round(j*sina))
            if (xt1>255 or xt1<0 or yt1>255 or yt1<0):
                tline[j+rr] = 0
            else:
                if is_weight:
                    #G = gTsqrt[mcOld2[0],mcOld2[1]] * (i != 0)                 #последний множитель нужен на случай первого запуска цикла               
                    deltaS = np.sqrt( (xt1-mcOld2[0])**2+(yt1-mcOld2[1])**2 )  #Расстояние между текущей точкой и предыдущим максимумом
                    #tline[j+rr] = gTsqrt[xt1][yt1] * G/(1+deltaS)
                    tline[j+rr] = gTsqrt[xt1][yt1] * np.exp(-(deltaS-1)*(i != 0)/3)
                    #expline[j+rr] = np.exp(-(deltaS-1)*(i != 0)/3)
                else:
                    tline[j+rr] = gTsqrt[xt1][yt1] #/ (1+deltaS*(i != 0))**2
            
                
        
        nm = (-np.abs(tline).T).argsort().T[0] #number of maximum 
        nm = nm - rr
        mc2 = xt + np.int(np.round(nm*cosa)), yt + np.int(np.round(nm*sina))
        #max coordinates
        borders[mc2[0],mc2[1]] = True
        mcOld2 = mc2
#        if i >= 59:
#            plt.figure(4); plt.imshow(gTsqrt+3*borders+3*inline+tnorm*0,'gray')
#            plt.pause(0.1)
#            plt.pause(0.1)
        
        dist[yt] = np.sqrt((mc2[0]-mc1[0])**2+(mc2[1]-mc1[1])**2)
    
def zeroing_spectrum_ends(data):
    i2 = np.arange(256) - 128
    gS = 1 - np.roll(np.exp(-(i2/20.)**2),128)
#    data = np.fft.ifft(np.fft.fft(data,axis = 0)
#                                          *gS[:,np.newaxis,np.newaxis],axis =0)
    data = np.fft.ifft(np.fft.fft(data,axis = 0)*gS[:,np.newaxis,np.newaxis],axis=0)
    return(data)