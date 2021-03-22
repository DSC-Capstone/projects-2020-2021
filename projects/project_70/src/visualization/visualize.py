#!/usr/bin/env python
# Title: Data Science DSC180A (Replication Project), DSC180B (miRNA Overlap Alzheimer's and Parkinson's)
# Section B04: Genetics
# Authors: Saroop Samra (180A/180B), Justin Kang (180A), Justin Lu (180B), Xuanyu Wu (180B)
# Date : 10/23/2020

import os
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.image as mpimg
import logging



def r2(x, y):
	return round(stats.pearsonr(x, y)[0] ** 2, 2)

def lm_corr_plot(x, y, df, title, out_image):
	# cite: https://stackoverflow.com/questions/60358228/how-to-set-title-on-seaborn-jointplot
	# stat_func is causing error on server??
	p = sns.jointplot(x=x, y=y, data=df,kind="reg", line_kws={'color': 'red'})
	p.fig.suptitle(title + " R2 = " + str(r2(df[x], df[y])))
	p.ax_joint.collections[0].set_alpha(0)
	p.fig.tight_layout()
	p.fig.subplots_adjust(top=0.95) # Reduce plot to make room
	plt.savefig(out_image)
	return


def pvalue_histograms(ylim, biofluid_regions, disorders, title, out_image):
	fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(10, 8))

	for ax, col in zip(axes[0], biofluid_regions):
		ax.set_title(col)

	fig.suptitle(title)
	#fig.subplots_adjust(top=0.9) # Reduce plot to make room
	
	#fig.tight_layout()

	colors = ['red', 'blue', 'green']
	col = 0

	k = 0
	labels = ["(a)", "(b)", "(c)", "(d)"]
	offset = [0.08, 0.08, 0.08, 0.08]

	for biofluid_region in biofluid_regions:
		row = 0
		for disorder in disorders:
			filename = "data/out/"+biofluid_region+"/"+disorder+"/lrt.tsv"
			if os.path.exists(filename):
				df = pd.read_csv(filename, sep='\t', index_col=0)
				# remove high pvalues?
				#df = df[df["pvalue"]<0.8]
				df["pvalue"].plot.hist(ax = axes[row,col], color=colors[col], bins=20, ylim=(0,ylim))
				axes[row,col].text(axes[row,col].get_xlim()[0], axes[row,col].get_ylim()[1]+offset[k], labels[k], c='purple', fontsize=16, fontweight='bold', alpha=0.5)

			row+=1
			k+= 1
		col+=1
		
	for ax, row in zip(axes[:,0], disorders):
		ax.set_ylabel(row, rotation=90, size='large')

	plt.savefig(out_image)
	


def process_corrmatrix(out_dir, corrmatrix):
	# Pairwise Spearman correlations of log2 fold gene expression changes between each disorder and CTL in each brain region. 
	# Cite: https://stackoverflow.com/questions/59381273/heatmap-with-circles-indicating-size-of-population

	N = 4
	M = 4
	ylabels = ["Parkinson", "Alzheimer"]*2
	xlabels = ["Alzheimer", "Parkinson"]*2
	biofluid_regions = ["Cerebrospinal", "Serum"]
	# names for positions along x-axis 0..9
	disorders_x = ["Alzheimer","Parkinson"]*2
	biofluid_regions_x = ["Cerebrospinal","Cerebrospinal", "Serum", "Serum"]

	# names for positions along x-axis 0..9
	disorders_y = ["Parkinson", "Alzheimer"]*2
	biofluid_regions_y = ["Serum", "Serum", "Cerebrospinal","Cerebrospinal"]
	
	# size of circles
	s = np.zeros((N,M))

	for y in range(N):
		for x in range(M):
			lrt1 = "data/out/"+biofluid_regions_x[x]+"/"+disorders_x[x]+"/lrt.tsv"
			lrt2 = "data/out/"+biofluid_regions_y[y]+"/"+disorders_y[y]+"/lrt.tsv"
			# Make sure ltr1 exists otherwise zero correlation
			if not os.path.exists(lrt1):
				s[y][x] = 0.5
				continue
			# Make sure ltr2 exists otherwise zero correlation
			if not os.path.exists(lrt2):
				s[y][x] = 0.5
				continue
			if lrt1 != lrt2:
				df_x = pd.read_csv(lrt1, sep='\t', index_col=0)
				df_y = pd.read_csv(lrt2, sep='\t', index_col=0)
				corr = np.abs(df_x["log2FoldChange"].corr(df_y["log2FoldChange"]))
				s[y][x] = corr
			else:
				s[y][x] = 0.0 #Dont set diagnols

	c = np.ones((N, M))

	fig, ax = plt.subplots(figsize=(11,8))

	R = s/s.max()/2
	x, y = np.meshgrid(np.arange(M), np.arange(N))
	circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
	col = PatchCollection(circles)
	col.set(array=s.flatten(), cmap = 'coolwarm')
	ax.add_collection(col)

	ax.set(xticks=np.arange(M), yticks=np.arange(N),
		   xticklabels=xlabels, yticklabels=ylabels)
	ax.set_xticks(np.arange(M+1)-0.5, minor=True)
	ax.set_yticks(np.arange(N+1)-0.5, minor=True)
	ax.grid(which='minor')
	ax.text(0,-0.9, "Cerebrospinal", size=20, color='red')
	ax.text(2,-0.9, "Serum", size=20, color='green')
	ax.text(3.55,2, "Cerebrospinal", size=20, rotation=90, color='red')
	ax.text(3.55,0, "Serum", size=20, rotation=90, color='green')
	
	fig.colorbar(col)
	plt.suptitle(corrmatrix["title"])
	plt.savefig(out_dir + "/corrmatrix.png" )
	return


def visualize_grid_images(biofluid_regions, disorders, image_filename, title, out_image):
	# Cite: https://stackoverflow.com/questions/25862026/turn-off-axes-in-subplots
	fig, axarr = plt.subplots(2, 2, figsize=(15,15))
	fig.suptitle(title, size=25)
	#plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	row = 0
	k = 0
	labels = ["(a)", "(b)", "(c)", "(d)"]
	offset = [0.08, 0.08, 0.08, 0.08]
	for biofluid_region in biofluid_regions:
		col = 0
		for disorder in disorders:
			filename = "data/out/"+biofluid_region+"/"+disorder+"/" + image_filename
			if os.path.exists(filename):
				im = mpimg.imread(filename)
				axarr[row,col].imshow(im, interpolation='bilinear')
			if row == 0 and col == 0:
				axarr[row,col].set_title("Cerebrospinal", size=20, color='red')
			if row == 0 and col == 1:
				axarr[row,col].set_title("Serum", size=20, color='blue')
			if row == 0 and col == 0:
				axarr[row,col].set_ylabel("Parkinson", size=20, color='purple')
			if row == 1 and col == 0:
				axarr[row,col].set_ylabel("Alzheimer", size=20, color='purple')

			axarr[row,col].text(axarr[row,col].get_xlim()[0], axarr[row,col].get_ylim()[1]+offset[k], labels[k], c='purple', fontsize=24, fontweight='bold', alpha=0.5)
			k += 1
			col += 1
		row += 1


	plt.savefig(out_image)
	return

def process_ma_plot(out_dir, ma_plot):
	visualize_grid_images(ma_plot["biofluid_regions"], ma_plot["disorders"],  ma_plot["src_image"], ma_plot["title"], out_dir + "/ma_plot.png")
	return

def process_heat_map(out_dir, heat_map):
	visualize_grid_images(heat_map["biofluid_regions"], heat_map["disorders"],  heat_map["src_image"], heat_map["title"], out_dir + "/heat_map.png")
	return

def process_histogram(out_dir, histogram):
	pvalue_histograms(histogram["ylim"], histogram["biofluid_regions"], histogram["disorders"], histogram["title"], out_dir + "/histogram.png")
	return

def process_normalized_count_plots(out_dir, sra_lm):
	df_norm1 = pd.read_csv(sra_lm["normalized_counts"], sep='\t', index_col=0)
	df_norm2 = pd.read_csv(sra_lm["vst_counts"], sep='\t', index_col=0)
	i = 1
	for sra in sra_lm["sra"]:
		df = pd.DataFrame()
		df['Log Regular Normalized Count'] = np.log(df_norm1[sra])
		df['VST Normalized Count'] = df_norm2[sra]
		title = sra_lm["title"].replace("%sra%", sra)
		out_image = out_dir + "/sra_" + str(i) + ".png"
		lm_corr_plot('VST Normalized Count', 'Log Regular Normalized Count', df, title, out_image)
		i += 1
	return


def process_venn(out_dir, venn):
	from matplotlib_venn import venn3, venn3_unweighted, venn2_unweighted

	plt.clf()
	pvalue_cutoff = venn["pvalue_cutoff"]
	biofluid_regions = venn["biofluid_regions"]
	disorders = venn["disorders"]
	genes = {}
	for biofluid_region in biofluid_regions:
		col = 0
		for disorder in disorders:
			filename = "data/out/"+biofluid_region+"/"+disorder+"/lrt.tsv"
			if os.path.exists(filename):
				df = pd.read_csv(filename, sep='\t', index_col=0)
				# Filter genes with pvalue less than cutoff
				df = df[df["pvalue"] < pvalue_cutoff]
				# Add to list
				if disorder in genes:
					genes[disorder] = genes[disorder] + df.index.tolist()
				else:
					genes[disorder] = df.index.tolist()
			else:
				genes[disorder] = []

	# Find unique genes per disorder
	for disorder in disorders:
		genes[disorder] = set(genes[disorder])

	a = genes[disorders[0]]
	b = genes[disorders[1]]

	intersection = a & b
	a_only = a - intersection
	b_only = b - intersection


	fig = venn2_unweighted(subsets = (len(a - (a&b)), len(b - (a&b)), len((a&b))), set_labels = tuple(disorders), alpha = 0.5)
	plt.text(x=0.05,y=-0.2,s= "\n".join(list(intersection)),color='black', bbox=dict(facecolor='orange', alpha=0.5))
	plt.text(x=-0.84,y=-0.25,s= "\n".join(list(a_only)),color='black', bbox=dict(facecolor='red', alpha=0.5))
	plt.text(x=0.66,y=-0.2,s= "\n".join(list(b_only)),color='black', bbox=dict(facecolor='green', alpha=0.5))
	
	plt.title(venn["title"])
	plt.savefig(out_dir + "/venn.png" )
	return

def process_plot_gene_hist(out_dir, gene_hist):
	# Do this in visualization
	max_genes = gene_hist["max_genes"]
	nbins = gene_hist["nbins"]
	gene_variance = pd.read_csv(out_dir + "/top_genes.tsv", sep="\t", index_col=0)
	fig, ax = plt.subplots()
	gene_variance["Spread"].plot(kind='hist', bins=nbins, alpha = 0.5)
	ax = gene_variance["Spread"][0:max_genes].plot(kind='hist', bins=nbins, alpha = 0.5)
	ax.legend(["All Genes", "Top Genes"])
	ax.set_xlim(0, 600)
	title = plt.suptitle(gene_hist["title"])
	plt.savefig(out_dir + "/top_genes.png")
	return

def process_plot_missing(out_dir, missing):
	# Do this in visualization
	title = missing["title"]
	df_full = pd.read_csv("./data/out/gene_matrix_full.tsv", index_col=0, sep="\t")
	df_core = pd.read_csv("./data/out/features.tsv", sep="\t")
	fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20,18))

	(df_full.isna().sum()*100 / df_full.shape[0]).plot(ax=ax[0], kind='bar')
	t = ax[0].set_title(title + " over all samples", color="red")
	ax[0].get_xaxis().set_visible(False)

	runs = df_core[ df_core["Biofluid"] ==  "Cerebrospinal"]["Run"]
	df2 = df_full[runs]
	(df2.isna().sum()*100 / df2.shape[0]).plot(ax=ax[1], kind='bar')
	t = ax[1].set_title(title + " over all Cerebrospinal Biofluid's", color="red")
	ax[1].get_xaxis().set_visible(False)

	runs = df_core[ (df_core["Biofluid"] ==  "Cerebrospinal") & (df_core["sex"] ==  "male") ]["Run"]
	df2 = df_full[runs]
	(df2.isna().sum()*100 / df2.shape[0]).plot(ax=ax[2], kind='bar')
	t = ax[2].set_title(title + " over all Male Cerebrospinal Biofluid's", color="red")
	ax[2].get_xaxis().set_visible(False)


	runs = df_core[ (df_core["Biofluid"] ==  "Cerebrospinal") & (df_core["sex"] ==  "male")  & (df_core["Disorder"] ==  "Parkinson")]["Run"]
	df2 = df_full[runs]
	(df2.isna().sum()*100 / df2.shape[0]).plot(ax=ax[3], kind='bar')
	t = ax[3].set_title(title + " over all Male Parkinson Cerebrospinal Biofluid's", color="red")
	plt.savefig(out_dir + "/missing.png")
	return


def process_volcano_plot(out_dir, volcano):

	pcutoff = volcano["pcutoff"]
	biofluids = volcano["biofluids"] 
	disorders = volcano["disorders"]
	title = volcano["title"]
	pcutoff = -np.log10(pcutoff)
	fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(10, 8))
	for ax, col in zip(axes[0], biofluids):
		ax.set_title(col, color='purple')
	color_dict = dict({'Down':'blue', 'Up':'red', 'Not Significant': 'gray'})

	k = 0
	labels = ["(a)", "(b)", "(c)", "(d)"]
	offset = [0.02, 0.05, 0.05, 0.05]
	for i in range(2):
		for j in range(2):
			disorder = disorders[i]
			biofluid = biofluids[j]
			legend = False
			if i==0 and j==1:
				legend = 'full'
			df = pd.read_csv(out_dir + "/" + biofluid + "/" + disorder + "/lrt.tsv", sep="\t", index_col=0)
			df["-log_pvalue"] = -np.log10(df["pvalue"])
			df["Type"] = np.where(df["-log_pvalue"] < pcutoff, "Not Significant", np.where(df["log2FoldChange"]<0, "Down", "Up"))
			sns.scatterplot(x='log2FoldChange', y='-log_pvalue', data=df, hue='Type',legend=legend, ax = axes[i,j], palette=color_dict)
			axes[i, j].axhline(pcutoff,color='black',ls='--')
			axes[i, j].text(axes[i,j].get_xlim()[0], axes[i,j].get_ylim()[1]+offset[k], labels[k], c='purple', fontsize=16, fontweight='bold', alpha=0.5)
			
			df = df[df['Type']!='Not Significant']
			df = df.sort_values('-log_pvalue', ascending=False)
			df = df[['Type', 'log2FoldChange', '-log_pvalue']]
			df.to_csv(out_dir + "/" + biofluid + "/" + disorder + "/updown_miRNAs.csv")
			k += 1


	for ax, row in zip(axes[:,0], disorders):
		ax.set_ylabel(row, rotation=90, size='large', color='purple')

	plt.suptitle(title, size=20)
	out_image = out_dir + "/volcano.png"
	plt.savefig(out_image)
	if volcano["show_details"]==1:
		process_volcano_plot_details(out_dir, volcano)
	return

def process_volcano_plot_details(out_dir, volcano):

	pcutoff = volcano["pcutoff"]
	biofluids = volcano["biofluids"] 
	disorders = volcano["disorders"]
	title = volcano["title"]
	pcutoff = -np.log10(pcutoff)
	fig, ax = plt.subplots(figsize=(10,8))
	
	color_dict = dict({'Down':'blue', 'Up':'red', 'Not Significant': 'gray'})

	# Just do one region
	i = 1
	j = 1
	disorder = disorders[i]
	biofluid = biofluids[j]
	legend = 'full'
	df = pd.read_csv(out_dir + "/" + biofluid + "/" + disorder + "/lrt.tsv", sep="\t", index_col=0)
	df["-log_pvalue"] = -np.log10(df["pvalue"])
	df["Type"] = np.where(df["-log_pvalue"] < pcutoff, "Not Significant", np.where(df["log2FoldChange"]<0, "Down", "Up"))
	sns.scatterplot(x='log2FoldChange', y='-log_pvalue', data=df, hue='Type',legend=legend, palette=color_dict)
	plt.axhline(pcutoff,color='black',ls='--')
	for index, row in df.iterrows():
		# skip ones that are too close together
		if index=='mir-22' or index=='mir-186':
			continue
		if row["Type"]=='Down':
			plt.text(x=row['log2FoldChange']+0.02, y=row['-log_pvalue']+0.01, s=index, fontsize=10)
		elif row["Type"]=='Up':
			plt.text(x=row['log2FoldChange']+0.02, y=row['-log_pvalue']+0.01, s=index, fontsize=10)

	plt.suptitle(disorder + " " + biofluid + " Volcano Plot " , size=20)
	out_image = out_dir + "/volcano_details.png"
	plt.savefig(out_image)
	return

def get_cond_tbl(gm, sra, selected_ft):
	# clean up the index
	gm.index = gm.index.str.slice(0,30)
	gm_reind = gm.T
	selected_ft = sra[selected_ft]
	# clean up selected ft
	selected_ft = selected_ft.replace({'sn_depigmentation':{'none':0, 'nan':0, 'mild':1, 'moderate':2, 'severe':3},
									   'Braak score':{'0':0, 'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 
													  'VI':6}}).set_index('Run')
	alz_ft = selected_ft[selected_ft.CONDITION=='Alzheimer\'s Disease'].drop('CONDITION', axis=1)
	park_ft = selected_ft[selected_ft.CONDITION=='Parkinson\'s Disease'].drop('CONDITION', axis=1)

	alz_tbl = alz_ft.merge(gm_reind, left_index=True, right_index=True, how='inner')
	park_tbl = park_ft.merge(gm_reind, left_index=True, right_index=True, how='inner')
	return alz_tbl, park_tbl, alz_ft, park_ft


def get_up_down_martix(up_down_path, gm):
	reg_seq = pd.read_csv(up_down_path)
	reg_seq  = reg_seq.set_index('Unnamed: 0')
	merged_reg = reg_seq.merge(gm, left_index=True, right_index=True)
	up_reg = merged_reg[merged_reg['Type']=='Up'].iloc[:,3:].T
	down_reg = merged_reg[merged_reg['Type']=='Down'].iloc[:,3:].T
	return up_reg, down_reg

def get_seq_overlap(gm, up_down_path):
	biofluid = ['Cerebrospinal', 'Serum']
	condition = ['Alzheimer', 'Parkinson']
	up_overlap = {}
	down_overlap = {}
	all_up = {}
	all_down = {}
	for i in biofluid:
		up_seq = {}
		down_seq = {}
		for j in condition:
			#up_reg, down_reg = get_up_down_martix(up_down_path%(i, j), gm)
			correct_path = up_down_path.replace("%s/%s", i + "/" + j)
			up_reg, down_reg = get_up_down_martix(correct_path, gm)
			up_seq[j] = set(up_reg.columns)
			down_seq[j] = set(down_reg.columns)
		up_overlap[i] = list(set.intersection(up_seq[condition[0]], up_seq[condition[1]]))
		down_overlap[i] = list(set.intersection(down_seq[condition[0]], down_seq[condition[1]]))
		all_up[i] = up_seq
		all_down[i] = down_seq
	return all_up, all_down, up_overlap, down_overlap

def get_corr_df(num_fts, fts_names, tbl):
	corr_df = pd.DataFrame()
	for i in range(num_fts):
		corr = tbl[tbl.columns[num_fts:]].apply(lambda x: x.corr(tbl.iloc[:,i]))
		corr_df[fts_names[i]] = corr
	return corr_df

def process_all_box(plot_path, box_all):
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)
	gm = pd.read_csv(box_all['gm_path'], sep='\t', index_col=0)
	sra = pd.read_csv(box_all['sra_path'])
	up_down_path = box_all['up_down_path']
	alz_tbl, park_tbl, _, _ = get_cond_tbl(gm, sra, box_all['selected_ft'])
	_, _, reg, _ = get_seq_overlap(gm, up_down_path)
	biofluid = ['Cerebrospinal', 'Serum']
	for i in biofluid:
		alz_gm = alz_tbl[list(reg[i])]
		park_gm = park_tbl[list(reg[i])]
		gm_arr = [alz_gm, park_gm]
		cols = ['Alzheimer\'s Disease','Parkinson\'s Disease']
		fig, axes= plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
		ind = 0
		for cond_gm in gm_arr:
			cond_gm.boxplot(ax = axes.flatten()[ind])
			ind = ind+1
		for ax, col in zip(axes, cols):
			ax.set_ylabel(col, rotation=90, size='large')
			ax.set_xticklabels(cond_gm.columns, rotation=90)
		fig.tight_layout()
		fig.subplots_adjust(top=0.9)
		fig.suptitle('Distribution of the overlapping sequence in %s'%i)
		fig.savefig(os.path.join(plot_path, 'overlapping_distr_%s.png'%i))

def process_regulated_corr(out_dir, reg_corr):
	gm = pd.read_csv(reg_corr['gm_path'], sep='\t', index_col=0)
	gm_reind = gm.T
	sra = pd.read_csv(reg_corr['sra_path'])
	up_down_path = reg_corr['up_down_path']
	_, _, alz_ft, park_ft = get_cond_tbl(gm, sra, reg_corr['selected_ft'])
	all_up, all_down, up_overlap, down_overlap = get_seq_overlap(gm, up_down_path)
	reg_li = [all_up, all_down, up_overlap]
	reg_name = ['up-regulated', 'down-regulated', 'overlapping']
	biofluid = ['Cerebrospinal', 'Serum']
	for reg_ind in range(len(reg_li)):
		reg = reg_li[reg_ind]
		for i in biofluid:
			if reg_ind == 2:
				alz_tbl = alz_ft.merge(gm_reind[reg[i]], left_index=True, right_index=True, how='inner')
				park_tbl = park_ft.merge(gm_reind[reg[i]], left_index=True, right_index=True, how='inner')
			else:
				alz_tbl = alz_ft.merge(gm_reind[reg[i]['Alzheimer']], left_index=True, right_index=True, how='inner')
				park_tbl = park_ft.merge(gm_reind[reg[i]['Parkinson']], left_index=True, right_index=True, how='inner')
			tbl_arr = [alz_tbl, park_tbl]
			cols = ['Alzheimer\'s Disease','Parkinson\'s Disease']
			num_fts = reg_corr['num_fts']
			fts_names = reg_corr['fts_names']
			if reg == all_down and i=='Cerebrospinal':
					fig = plt.figure(figsize=(8, 6))
					corr_df = get_corr_df(num_fts, fts_names, alz_tbl)
					fig = sns.heatmap(corr_df)
					fig.set_xticklabels(fts_names)
					fig.set_ylabel('Alzheimer\'s Disease')
					plt.title('Correlation between the %s sequence and basic numerical features in %s'%(reg_name[reg_ind],i))
					plt.savefig(os.path.join(out_dir, '%s_corr_%s.png'%(reg_name[reg_ind],i)),bbox_inches='tight')
			else: 
				fig, axes= plt.subplots(nrows=1, ncols=2, figsize=(8, 10))
				ind = 0
				for tbl in tbl_arr:
					corr_df = get_corr_df(num_fts, fts_names, tbl)
					sns.heatmap(corr_df, ax = axes.flatten()[ind])
					axes.flatten()[ind].set_xticklabels(fts_names)
					ind = ind+1
				for ax, col in zip(axes, cols):
					ax.set_ylabel(col, rotation=90, size='large')
				fig.tight_layout()
				fig.subplots_adjust(top=0.9)
				fig.suptitle('Correlation between the %s sequence and basic numerical features in %s'%(reg_name[reg_ind],i))
				fig.savefig(os.path.join(out_dir, '%s_corr_%s.png'%(reg_name[reg_ind],i)))


					  

def process_plots(out_dir, plot_path, gene_hist, missing_plot, sra_lm, ma_plot, heat_map, histogram, corrmatrix, venn, volcano, box_all, reg_corr, verbose):
		
	if verbose:
		logging.info("# ---------------------------------------------------")
		logging.info("# Visualize")

	# Process SRA LM Plots of Normalized Plots
	if gene_hist["enable"] == 1:
		process_plot_gene_hist(out_dir, gene_hist)
	# Process SRA LM Plots of Normalized Plots
	if missing_plot["enable"] == 1:
		process_plot_missing(out_dir, missing_plot)
	# Process SRA LM Plots of Normalized Plots
	if sra_lm["enable"] == 1:
		process_normalized_count_plots(out_dir, sra_lm)
	# Process MA Plot
	if ma_plot["enable"] == 1:
		process_ma_plot(out_dir, ma_plot)
	# Process Heat Map Plot
	if heat_map["enable"] == 1:
		process_heat_map(out_dir, heat_map)
	# Process Histogram Plot
	if histogram["enable"] == 1:
		process_histogram(out_dir, histogram)
	# Process Corr Matrix Plot
	if corrmatrix["enable"] == 1:
		process_corrmatrix(out_dir, corrmatrix)
	# Process Corr Matrix Plot
	if venn["enable"] == 1:
		process_venn(out_dir, venn)

	if volcano["enable"] == 1:
		process_volcano_plot(out_dir, volcano)
		
	if box_all["enable"] == 1:
		process_all_box(plot_path, box_all)
	
	if reg_corr["enable"] == 1:
		process_regulated_corr(out_dir, reg_corr)

	if verbose:
		logging.info("# Finished")
		logging.info("# ---------------------------------------------------")
	return

