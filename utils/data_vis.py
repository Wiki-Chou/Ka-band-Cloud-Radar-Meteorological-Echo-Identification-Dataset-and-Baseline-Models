import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

from matplotlib import pyplot as plt
import numpy as np
import datetime
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

import matplotlib as mpl
mpl.rcParams['figure.dpi']=400

import colorsys


# Function to adjust both brightness and saturation
def adjust_brightness_saturation(cmap, brightness_scale=0.8, saturation_scale=0.8):
    """ Adjust brightness and saturation of a colormap.
    Args:
        cmap: Original colormap
        brightness_scale: Factor to darken (0 = black, 1 = original brightness)
        saturation_scale: Factor to reduce saturation (0 = grayscale, 1 = original saturation)
    Returns:
        adjusted_cmap: New colormap with adjusted brightness and saturation
    """
    # Get the colormap colors (in RGBA format)
    colors = cmap(np.arange(cmap.N))
    
    # Initialize array to store adjusted colors
    adjusted_colors = np.zeros_like(colors)
    
    # Loop over all colors in the colormap
    for i, (r, g, b, a) in enumerate(colors):
        # Convert RGB to HLS (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # Adjust lightness (brightness) and saturation
        l = l * brightness_scale  # Adjust brightness
        s = s * saturation_scale  # Adjust saturation
        
        # Convert back to RGB
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
        
        # Store the new color
        adjusted_colors[i] = [r_new, g_new, b_new, a]  # Keep alpha channel unchanged

    # Create a new colormap from the adjusted colors
    adjusted_cmap = mpl.colors.ListedColormap(adjusted_colors)

    return adjusted_cmap

def Visualize(Cloud_mask, Height, Time, Title='', Height_range=(0, 15), Save_path=None, unit='', vmin=0, vmax=30, color='viridis', figsize=(12, 4), size=24, ignore_min=-50, bound=10,half=False):
    # Height=Height[::-1]
    # 字体大小都设置为22
    if half:
        Time=Time[:len(Cloud_mask)//2]
        Cloud_mask=Cloud_mask[:len(Cloud_mask)//2]
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = size
    
    fig, ax = plt.subplots(figsize=figsize,dpi=400)  # 创建图形和轴对象
    # h1为Height_range[0]对应的高度索引，h2为Height_range[1]对应的高度索引
    h1 = np.argmin(np.abs(Height - Height_range[0]))
    h2 = np.argmin(np.abs(Height - Height_range[1]))
    Cloud_mask = Cloud_mask[:, h1:h2][:, ::-1]
    Cloud_mask = Cloud_mask.astype(float)
    Cloud_mask[Cloud_mask < ignore_min] = np.nan

    # 创建 masked array

    ax.set_title(Title, fontsize=size)
    
    # 创建离散的 colorbar
    bounds = np.arange(vmin, vmax + bound, bound)  # 间断点为整数
    
    
    if color == 'viridis':
        # 离散化 len(bounds)-1 个颜色
        cmap = mpl.cm.get_cmap(color)
        # 平均取色 len(bounds)-1 个颜色
        colorlist = [cmap(i) for i in np.linspace(0, 1, len(bounds) +1)]
        cmap = ListedColormap(colorlist)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
    else:
        norm = mpl.colors.BoundaryNorm(bounds, plt.get_cmap(color).N, extend='both')
        cmap=plt.get_cmap(color)
        '''if Title == 'Velocity':
            cmap = adjust_brightness_saturation(cmap, brightness_scale=0.8, saturation_scale=0.6)  # Adjust scales here'''
    

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
    im = ax.imshow(Cloud_mask.T, extent=[0, len(Time), h1, h2], aspect='auto',
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    # cbar顶端标注单位
    cbar.set_label(unit)

    # colorbar标注单位
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Height (km)')
    Height_label = np.arange(Height_range[0], Height_range[1] + 3, 3)
    Height_label = Height_label.astype(int).astype(str)
    # 从h1到h2之间等距离取10个点作为y轴标注
    Height_index = np.linspace(h1, h2 + 3, len(Height_label)).astype(int)

    # y轴标注高度信息
    ax.set_yticks(Height_index)
    ax.set_yticklabels(Height_label)
    if half:
        Time_str = ['00:00', "03:00", "06:00", "09:00", "12:00"]
    else:
        # 将时间信息转化为字符串 并标注在图像上
        Time_str = ['00:00', "06:00", "12:00", "18:00", '24:00']
    ax.set_xticks(np.arange(0, len(Time) + 1, int(len(Time) / 4)))
    ax.set_xticklabels(Time_str)
    ax.set_xlim(0, len(Time))
    ax.tick_params(width=1, length=5)
    # ax.set_ylim(Height_range[0], Height_range[1])
    # plt.savefig(Save_path)
    plt.show()
    return 0

def Visualize_type(Cloud_mask,Height,Time,Title='',Height_range=(0,15),Save_path=None,unit='',vmin=0,vmax=1, 
                   cbar_ticks=[0, 1], cbar_labels=['', ''],size=24,figsize=(9,5),color='Set1',show_cbar=True,half=False):
    if half:
        Time=Time[:len(Cloud_mask)//2]
        Cloud_mask=Cloud_mask[:len(Cloud_mask)//2]
    # 获取 'jet' 色彩映射
    jet = plt.get_cmap(color)
    # 选择 'jet' 平均映射n个点的颜色
    n = len (cbar_ticks)
    colorlist = [jet(int(i)) for i in np.linspace(0, 3, n)]
    colorlist[0]=(1,1,1,1)
    print(colorlist)
    # 创建一个自定义的颜色映射
    cmap = ListedColormap(colorlist)

    
    #Height=Height[::-1]
    #字体大小都设置为22

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.size']=size
    fig, ax = plt.subplots(figsize=figsize,dpi=400)
    #h1为Height_range[0]对应的高度索引，h2为Height_range[1]对应的高度索引
    h1=np.argmin(np.abs(Height-Height_range[0]))
    h2=np.argmin(np.abs(Height-Height_range[1]))
    Cloud_mask=Cloud_mask[:,h1:h2][:,::-1]
    #Cloud_mask[Cloud_mask==0]=np.nan
    im = ax.imshow(Cloud_mask.T,extent=[0,len(Time),h1,h2],aspect='auto',
                cmap=cmap, vmin=vmin, vmax=vmax,interpolation='nearest')
    ax.title.set_text(Title)

    #colorbar标注单位
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Height (km)')
    Height_label=np.arange(Height_range[0],Height_range[1]+3,3)
    Height_label=Height_label.astype(int).astype(str)
    #从h1到h2之间等距离取10个点作为y轴标注
    Height_index=np.linspace(h1,h2+3,len(Height_label)).astype(int)
    #y轴标注高度信息\
    print(Height_index,Height_label)
    ax.set_yticks(Height_index)
    ax.set_yticklabels(Height_label)
    ax.tick_params(width=1, length=4)
    #将时间信息转化为字符串 并标注在图像上
    
    if half:
        Time_str = ['00:00', "03:00", "06:00", "09:00", "12:00"]
    else:
        # 将时间信息转化为字符串 并标注在图像上
        Time_str = ['00:00', "06:00", "12:00", "18:00", '24:00']
    plt.xticks(np.arange(0,len(Time)+1,int(len(Time)/4)),Time_str)

    ax.set_xlim(0,len(Time))


    # 设置颜色条的标签
    # 创建一个新的颜色条
    if show_cbar:
        n_cbar_ticks=len(cbar_ticks)
        gap=(n_cbar_ticks-1)/(n_cbar_ticks*2)
        cbar_ticks=[i*2*gap+gap for i in range(n_cbar_ticks)]
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', ticks= cbar_ticks)
        # 设置颜色条的标签
        cbar.set_ticklabels(cbar_labels)
        #右侧标注每个颜色对应的含义
        #cbar_ticks=[0, 1, 2]
        #cbar_labels=['None', 'Rain', 'Cloud']
        #cbar顶端标注单位
        cbar.set_label(unit)
    

    #plt.ylim(Height_range[0],Height_range[1])
    
    plt.show()
    return 0


def plot_imgs(img,Height_range=(0,15),half=False):
    r=img[:,:,0]
    v=img[:,:,1]
    w=img[:,:,2]
    depo=img[:,:,3]
    Height=np.array(range(len(r[0])))*0.03
    Time=np.array(range(len(r)))
    #
    Visualize(r,Height,Time,Title='Radar reflectivity',unit='dBZ',vmin=-40,vmax=30,Height_range=Height_range,ignore_min=-50,bound=10,half=half)
    Visualize(v,Height,Time,Title='Velocity',unit='m/s',vmin=-2,vmax=2,Height_range=Height_range,color='coolwarm',ignore_min=-15,bound=0.5,half=half)
    Visualize(w,Height,Time,Title='Width',unit='m/s',vmin=0,vmax=0.5,Height_range=Height_range,color='winter',ignore_min=0,bound=0.05,half=half)
    Visualize(depo,Height,Time,Title='LRD',unit='',vmin=-22,vmax=-12,Height_range=Height_range,color='cividis',ignore_min=-50,bound=1,half=half)
    
def plot_mask(img,mask):
    #echo_mask 2-met 1-ca 0-nan
    r=img[:,:,0]
    v=img[:,:,1]
    w=img[:,:,2]
    Height=np.array(range(len(r[0])))*0.03
    Time=np.array(range(len(r)))
    echo_mask=np.full(mask.shape,np.nan)
    echo_mask[r >= -50] = 1
    echo_mask[v >=-15] = 1
    echo_mask[w >=0] = 1
    echo_mask[mask==1]+=1
    echo_mask[:,150:][echo_mask[:,150:]==1]=2
    
    Visualize_type(mask,Height,Time,Title='Echo mask',Height_range=(0,15),Save_path=None,unit='',vmin=0,vmax=1, 
                   cbar_ticks=[0, 1], cbar_labels=['None', 'Meteorological'])
    Visualize_type(echo_mask,Height,Time,Title='Echo mask',Height_range=(0,15),Save_path=None,unit='',vmin=0,vmax=2, 
                   cbar_ticks=[0, 1,2], cbar_labels=['None', 'Clear-air', 'Meteorological'])
    mask[echo_mask==2]=1
    return mask,echo_mask




def plot_img_and_mask(img, mask,true_mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes*2 + 1)
    ax[0].set_title('Radar reflectivity')
    ax[0].imshow(img[:,::-1,0].T, cmap='gray')#,vmin=-50, vmax=30)
    #色条 cbar
    
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i + 1})')
            ax[i+1].imshow(mask[i,:, ::-1].T,vmin=0, vmax=1)
            ax[i+4].set_title(f'True mask (class {i + 1})')
            ax[i+4].imshow(true_mask[:,::-1].T,vmin=0, vmax=1)
            if i ==0:
                print(mask[i,:, :].shape)
                print(true_mask[:, :].shape)
                print(img[:,:,i].shape)
                print(np.sum(mask[i,:, :]))
                print(np.sum(true_mask[:, :]))
    else:
        ax[1].set_title(f'Met echo')
        ax[1].imshow(mask[:,::-1].T,vmin=0, vmax=1)
        ax[2].set_title(f'True mask')
        ax[2].imshow(true_mask[:,::-1].T,vmin=0, vmax=1)
    
    plt.xticks([])
    plt.yticks([])
    plt.show()
