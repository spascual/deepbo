n_datapts_list='50 100' # 250 500 750 1000 1250 1500 2000 2250 2500 2750'
id_list='3 4'
for item in $id_list; do 
	DGP_sample_data=DGPS_$item
	for n_datapts in $n_datapts_list; do 
		echo $DGP_sample_data
		echo $n_datapts
		python thesis_work/REG_DGP_sampling.py $DGP_sample_data $n_datapts &
		
	done	
done