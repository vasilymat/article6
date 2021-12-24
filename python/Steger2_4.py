# -*- coding: utf-8 -*-
"""
Тут я переписываю поиск границ в виде "линк-алгоритма"
Стегер в виде функций.

@author: LAB
"""
import numpy as np
import skimage as si
import skimage.feature as sf
import skimage.filters as fil
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import steger_functions as stfn
from PIL import Image #Чтобы в .bmp сохранять
import time

def steger_line(T,sigma_line,sz,finding_area=1,col=None):
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
    
    
    #добавим, чтобы максимум находился где-нибудь в центре
    inmax = np.full((sz,sz), False)
    inmax[20:236,20:200] = True

    if (col != None):        
        e1,e2 = int( np.unravel_index(np.abs(Ev_max[20:236,col]).argmax()
                                          ,Ev_max[20:236,col].shape )[0] ),col
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



def get_thickness(linecoord,dist,borders,v,sig_lenght,gTsqrt):
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

                    tline[j+rr] = gTsqrt[xt1][yt1] * np.exp(-(deltaS-1)*(i != 0)/3)

                else:
                    tline[j+rr] = gTsqrt[xt1][yt1] #/ (1+deltaS*(i != 0))**2
            
                
        
        nm = (-np.abs(tline).T).argsort().T[0] #number of maximum 
        nm = nm - rr
        mc2 = xt + np.int(np.round(nm*cosa)), yt + np.int(np.round(nm*sina))
        #max coordinates
        borders[mc2[0],mc2[1]] = True
        mcOld2 = mc2
        
        dist[yt] = np.sqrt((mc2[0]-mc1[0])**2+(mc2[1]-mc1[1])**2)
def get_thickness2(xy_l,v,xy_g1,xy_g2,dist,sz):
    Hor_line = np.arange(-64,64)
    for i in range(xy_l.shape[0]):    
    
        xt = xy_l[i,0]
        yt = xy_l[i,1]
        atan = np.arctan(v[1,xt,yt]/v[0,xt,yt])        
        sina,cosa = np.sin(atan),np.cos(atan)
        
#        xlen = 128*sina/2 #Это длина вдоль "нормальной" оси Х, поэтому sina
        
#        #yt - это "нормальная" координата X
#        #Xcn - пространство из 128 точек длиной, равной прокции нормали на ОХ
#        Xcn = np.linspace(strt,fnsh,128,endpoint=False) #Xcontinuous 
        Xcn,lnorm_float = Hor_line*sina,Hor_line*cosa #Xcontinuous 
        Xcn += yt
        lnorm_float += xt
#        
#        #Это уравнение линии. lnorm и X это "дискретные" представления Xcn и lnorm_float
#        lnorm_float = Xcn/tana + xt - 1/tana*yt
        lnorm_float[lnorm_float < 0] = 0
        lnorm_float[lnorm_float > 255] = 255
        Xcn[Xcn < 0] = 0
        Xcn[Xcn > 255] = 255
#        lnorm = np.int32(lnorm_float)[np.newaxis].T
        Xstrt,Xend = int(Xcn[0]), int(Xcn[-1])
                
        Xc1,Yc1 = stfn.intersection_point(xy_g1,Xstrt,Xend,lnorm_float,Xcn)
        Xc2,Yc2 = stfn.intersection_point(xy_g2,Xstrt,Xend,lnorm_float,Xcn)

        if (Xc1 == None) or (Xc2 == None):
            dist[yt] = 0
        else:
            dist[yt] = np.sqrt( (Xc1-Xc2)**2 + (Yc1-Yc2)**2)
    
def main_circ(Y0):
    # сечение по "быстрой" оси на линии Y0
    ctemp = C[:,:,Y0+12]
    ctemp[0][:] = ctemp[1][:]
    dlog = 20*np.log10(np.abs(ctemp)+10**-6)    

    T = np.double(dlog)
    sz = 256
    
    inline, first_branch, second_branch, v, Ev_max = steger_line(T,sigma_line,sz)
    xy_l, inline = stfn.unique_defined(first_branch,second_branch,sz)
    
    #============================Ищем границы==================================
      
    gT = fil.gaussian(T,sigma=3)  #Свертка границы
    gTx = np.gradient(gT, axis=0)
    gTx[0:15][:] = 0
    gTy = np.gradient(gT, axis=1)
    gTy[0:15][:] = 0
    gTsqrt = np.sqrt(gTx**2+gTy**2)
     
    dist = np.zeros(256) #distribution right
    
    sig_s = 3
    first_branch,second_branch = steger_line(gTsqrt,sig_s,sz)[1:3]
    xy_g1, inline_g1 = stfn.unique_defined(first_branch,second_branch,sz)    
    
    t_g = fil.gaussian(inline_g1,sigma=sig_s)
    t_g = t_g < t_g.max()/np.e**2 
    
    first_branch,second_branch = steger_line(gTsqrt,sig_s,sz,t_g)[1:3]
    xy_g2, inline_g2 = stfn.unique_defined(first_branch,second_branch,sz)
    
    get_thickness2(xy_l,v,xy_g1,xy_g2,dist,sz)
    
    if Y0 in savelist:
        ti = np.argwhere(savelist == Y0)
        chosen_slices[ti,:] = T + 120*inline + 120*inline_g1+120*inline_g2
#       plt.figure(3); plt.imshow(dlog+120*inline+120*borders,'gray')
#       plt.pause(0.1)       
    
    #boolean массив, где есть толщина перепонки
    bool_dist[np.int16(Y0/2),:] = np.not_equal(dist, 0)
    
    #фильтр низких частот
    temp_zeros = (dist != 0)
    fdist = np.fft.fft(dist)
    fdist[18:237] = 0
    dist = np.real( np.fft.ifft(fdist) )*temp_zeros
    
    distmap[np.int16(Y0/2),:] = dist
#=============================================================================   
#!! Остановился на том, что нужно свернуть функцию с правильным ядром
#sig = 10
#x,y = np.mgrid[-128:128,-128:128]
#g = (2*np.pi*sig**2)**(-1)*np.exp(-(x**2)/2/sig**2 -(y**2)/2/sig**2 )

#filename = 'C:\\MatkivskiyV\\!Актуальная наука\\статья 6\\end-oct\\Дубилина\\2.dat' #файл лежит в той же директории

#Папка с данными
#filepath = 'C:\\OCT\\ent-oct\\На обработку\\после тимпанопластики\\'
#filepath = 'C:\\MatkivskiyV\\!Актуальная наука\\статья 6\\end-oct\\Дубилина\\'
filepath = 'C:\\MatkivskiyV\\science\\article6\\'
#Имя файла в этой папке
filename = 'Пациент1.dat'
#You can download the file from the link https://disk.yandex.ru/d/mE17vOEMQ6k5-A

#filepath - Папка с данными
#filename - Имя файла в этой папке

#Важные параметры:
#Это примерная ширина линии.
sigma_line = 25
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
if (not ('filename_old' in locals())) or (filename != filename_old):
    A = np.fromfile(filepath+filename,dtype='complex64') # прочитать файл с комплексными данными
    y = 1024
    x = 256
    z = 256
    B = np.reshape(A,(y,x,z))
    C = np.transpose(B, axes = (1,2,0)) # замена осей на (x,z,y)
    C = stfn.zeroing_spectrum_ends(C) #Убираем выхилы
    del A,B
    filename_old = filename


dist_size = 500
distmap = np.zeros((dist_size,256))
#Тут сохраняются позиции, где толщины !=0, нужно для усреднения (на что делить)
bool_dist = np.full((dist_size,256), False)

#Выбранные 4 среза для отображения 
chosen_slices = np.zeros((8,256,256))
ti = 0
savelist = np.array([32,64,128,192,256,320,384,448])*2


t1 = time.process_time()

#Parallel(n_jobs=2, backend='multiprocessing')(delayed(main_circ)(Y0) 
#                                                    for Y0 in range(0,1000,2))

for Y0 in range(0,1000,4):
#for Y0 in range(732,733):
    main_circ(Y0)

print(time.process_time() - t1)

savelist = np.int32(savelist/2)

distmap = (distmap*3200/256/1.44).T
    
#Вычислим стреднюю толщину перепонки, дисперсию, гистограмму
norm_thick = np.uint8(bool_dist).sum()
dit = distmap[bool_dist.T] #distmap temp
av_thickness = dit.sum()/norm_thick
#среднеквадратичное отклонение
stdv_thick = np.sqrt( ((dit - av_thickness)**2).sum()/norm_thick )


#=======================Выводим карту толщины==================================
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
#plt.close()

# Сохраняем в .bmp как бы ужимая 4 пикселя в 1
tmp = np.uint8(np.abs(distmap/4))
img = Image.fromarray(tmp)
img.save(filepath+filename[:-3]+'bmp')
    
    