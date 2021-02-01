import numpy as np
import os
import sys

import matplotlib
from matplotlib import pyplot as plt
import brewer2mpl

import xlrd

import matplotlib.lines as mlines

import pylab


'''files = ['results/Whole/NY_Stock_exchange_close_100/combined_NY_Stock_exchange_close_5.0_all_frac.xlsx',\
   'results/Whole/NY_Stock_exchange_high_100/combined_NY_Stock_exchange_high_5.0_all_frac.xlsx',\
    'results/Whole_server/cadata/combined_cadata_0.3_all_frac.xlsx',\
    'results/Whole_server/MSD_no_full/combined_MSD_0.7_all_frac.xlsx',\
    'results/Whole_server/LawSchool/combined_LawSchool_0.04_all_frac.xlsx']'''

"""files = ['results/Faster/cadata/combined_cadata_0.3_all_frac.xlsx',\
    'results/Faster/LawSchool/combined_LawSchool_0.04_all_frac.xlsx',\
    'results/Faster/MSD_50/combined_MSD_0.7_all_frac.xlsx']"""

files = ['results/Faster/NY_Stock_exchange_close_100/combined_NY_Stock_exchange_close_5.0_all_frac.xlsx',\
'results/Faster/NY_Stock_exchange_high_100/combined_NY_Stock_exchange_high_5.0_all_frac.xlsx']

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
color_list = bmap.mpl_colors
color_list = [(0.4, 0.7607843137254902, 0.6470588235294118),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
             (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
             (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
             (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
             (1.0, 0.8509803921568627, 0.1843137254901961),
             (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
             (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

color_list = [(0, 0, 1),(1, 0, 0),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
             (0,0,0),
             (1.0, 0.8509803921568627, 0.1843137254901961),
            (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
            (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
            (0.4, 0.7607843137254902, 0.6470588235294118),
            (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
            (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

def latexify():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amsfonts}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
    plt.rc('axes', linewidth=1)
    plt.rc('font', weight='bold')
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']


sel = 0
x_axis =[]

good_labels = ["Random with Constraints","Ours","Random","Facility","Glister",\
    "Facility with Constraints","Full"]


latexify()

#good_labels = [r'\textbf{'+i+'}' for i in good_labels] 


'''Triage_Alg = mlines.Line2D([], [], color=color_list[0], marker='o',linewidth=4,
                               markersize=14, label=good_labels[0])
Triage_Est = mlines.Line2D([], [], color=color_list[1], marker='o', linewidth=4,
                               markersize=14, label=good_labels[1])
SDG = mlines.Line2D([], [], color=color_list[2], marker='o', linewidth=4,
                               markersize=14, label=good_labels[2])
DG = mlines.Line2D([], [], color=color_list[3], marker='o', linewidth=4,
                               markersize=14, label=good_labels[3])
FA = mlines.Line2D([], [], color=color_list[4], linestyle='dashed',linewidth=6,
                               markersize=14, label=good_labels[4])
NA = mlines.Line2D([], [], color=color_list[5], linestyle='dashed',linewidth=6,
                               markersize=14, label=good_labels[5])

#print(x_axis)

file_no =1

fig = pylab.figure()
figlegend = pylab.figure(figsize=(20,1))

figlegend.legend(ncol=4, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
                 frameon=False,handles=[DG,SDG,Triage_Alg,Triage_Est,FA,NA],
                               handlelength=0.7,fontsize=70,handletextpad=0.3,columnspacing=3.0)

file = "results/Images/Acc_vs_S/legend.pdf"#".pdf"
plt.savefig(file, bbox_inches='tight')'''

markers = ["o","s","v","*","X","x","^","<",">",'+']

for file in files:

    data_name = file.split('/')[-1].split('.')[0]
    
    acc =[]
    wb = xlrd.open_workbook(file)
    sheet = wb.sheet_by_index(0) 

    curr_row = 0
    while curr_row < (sheet.nrows - 1):
        curr_row += 1
        row = sheet.row(curr_row)
        acc.append(row) 
    
    whole_sheet = [[ele.value for ele in each] for each in acc]
    #print(acc)

    time = np.array(whole_sheet[4:8],dtype=np.float32)
    acc = np.array(whole_sheet[-4:],dtype=np.float32)

    time[:,0] = time[:,0]*100
    acc[:,0] = acc[:,0]*100

    #time = np.array(whole_sheet[4:9],dtype=np.float32)
    #acc = np.array(whole_sheet[-5:],dtype=np.float32)

    #print(time)
    print(acc)

    #ax = fig.add_subplot(1,len(files),file_no)

    time_unit = 3600
    time_unit_str = 'hrs.'#"min."
    
    fig, ax = plt.subplots()#(2,len(files))
    

    for i in range(1,acc.shape[1]):#-2):

        clr =0

        #if i == acc.shape[1] -2:
        #   continue
        #print(clr)[0,file_no]
        #ax.plot(acc[:,0],acc[:,i],label=good_labels[i-1],linewidth=3,markersize=5,color=color_list[clr])
        for j in range(acc.shape[0]):
            if i == 2:
                plt.scatter(time[j,i]/time_unit, acc[j,i], label=good_labels[i-1],color=color_list[clr],\
                    marker=markers[i-1]) #+"-"
            else:
                plt.scatter(time[j,i]/time_unit, acc[j,i], label=good_labels[i-1],color=color_list[clr],\
                marker=markers[i-1])
            clr+=1

    plt.ylabel(r'\textbf{Test Error}$\rightarrow$', fontsize=20)
    print(file)

    plt.xlabel(r'\textbf{Time (in '+time_unit_str+r')}$\rightarrow$', fontsize=20,labelpad=15)
    #plt.ylim(top=1,bottom=0)
    #plt.xticks(acc[:,0])
    #print(np.arange(80,97,step=4))
    #plt.yticks(np.arange(80,97,step=4))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax.legend(prop={'size': 18}, frameon=False,handlelength=0.4,loc='center left',\
         bbox_to_anchor=(-0.65, 0.5))

    plt.box(on=True)
    plt.grid(axis='y',linestyle='-', linewidth=1)
    plt.grid(axis='x',linestyle='-', linewidth=1)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    file = "results/Images/Acc_vs_S/"+data_name+"_scatter.png"#".pdf"
    plt.savefig(file, bbox_inches='tight')

    #file_no+= 1

#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')


"""file_no =1
for file in files:

    data_name = file.split('/')[-1].split('.')[0]
    
    acc =[]
    wb = xlrd.open_workbook(file)
    sheet = wb.sheet_by_index(0) 

    curr_row = 0
    while curr_row < (sheet.nrows - 1):
        curr_row += 1
        row = sheet.row(curr_row)
        acc.append(row) 
    
    whole_sheet = [[ele.value for ele in each] for each in acc]
    #print(acc)

    time = np.array(whole_sheet[4:9],dtype=np.float32)
    #acc = np.array(whole_sheet[-5:],dtype=np.float32)

    time[:,0] = time[:,0]*100
    #acc[:,0] = acc[:,0]*100   
    
    #fig, ax = plt.subplots()
    ax = fig.add_subplot(2,len(files),len(files)+file_no)
    clr =0

    time_unit = 3600
    time_unit_str = 'hrs.'#"min."

    for i in range(1,time.shape[1]):
        if i == 4:
            clr+=1
            continue
        ax.plot(time[:,0],time[:,i]/time_unit,label=good_labels[i-1],linewidth=3,markersize=5,color=color_list[clr])
        clr+=1

    plt.ylabel(r'\textbf{Time (in '+time_unit_str+r')}$\rightarrow$', fontsize=20)
    print(file)

    plt.xlabel(r'\textbf{Budget (in \%)}$\rightarrow$', fontsize=20,labelpad=15)
    #plt.ylim(top=1,bottom=0)[1,file_no]
    plt.xticks(acc[:,0])
    #print(np.arange(80,97,step=4))
    #plt.yticks(np.arange(80,97,step=4))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    #ax.legend(prop={'size': 18}, frameon=False,handlelength=0.4,loc='center left',\
    #     bbox_to_anchor=(-0.65, 0.5))

    #ax.legend(ncol=5, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
    #             frameon=False,handles=[DG,SDG,Triage_Alg,Triage_Est,FA,NA],
    #                           handlelength=0.7,fontsize=70,handletextpad=0.3,columnspacing=3.0)

    plt.box(on=True)
    plt.grid(axis='y',linestyle='-', linewidth=1)
    plt.grid(axis='x',linestyle='-', linewidth=1)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    #file = "results/Images/Time_vs_S/"+data_name+".png"#".pdf"
    #plt.savefig(file, bbox_inches='tight')
    file_no+=1"""

'''ax.legend(ncol=8, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
                 frameon=False,handles=[DG,SDG,Triage_Alg,Triage_Est,FA,NA],
                               handlelength=0.7,fontsize=70,handletextpad=0.3,columnspacing=3.0)

file = "results/Images/Acc_vs_S/combined.pdf"#".pdf"
plt.savefig(file, bbox_inches='tight')'''