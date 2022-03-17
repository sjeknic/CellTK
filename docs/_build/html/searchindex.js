Search.setIndex({docnames:["arrays","ext","index","opers","pipes","pro","quick","seg","tra"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,sphinx:56},filenames:["arrays.rst","ext.rst","index.rst","opers.rst","pipes.rst","pro.rst","quick.rst","seg.rst","tra.rst"],objects:{"celltk.core":[[0,0,0,"-","arrays"],[4,0,0,"-","orchestrator"],[4,0,0,"-","pipeline"]],"celltk.core.arrays":[[0,1,1,"","ConditionArray"],[0,1,1,"","ExperimentArray"]],"celltk.core.arrays.ConditionArray":[[0,2,1,"","add_metric_slots"],[0,2,1,"","filter_cells"],[0,2,1,"","generate_mask"],[0,2,1,"","interpolate_nans"],[0,2,1,"","load"],[0,2,1,"","propagate_values"],[0,2,1,"","remove_parents"],[0,2,1,"","remove_short_traces"],[0,2,1,"","reshape_mask"],[0,2,1,"","save"],[0,2,1,"","set_condition"],[0,2,1,"","set_position_id"],[0,2,1,"","set_time"]],"celltk.core.arrays.ExperimentArray":[[0,2,1,"","filter_cells"],[0,2,1,"","generate_mask"],[0,2,1,"","interpolate_nans"],[0,2,1,"","load"],[0,2,1,"","load_condition"],[0,2,1,"","merge_conditions"],[0,2,1,"","plot_by_condition"],[0,2,1,"","predict_peaks"],[0,2,1,"","remove_empty_sites"],[0,2,1,"","remove_short_traces"],[0,2,1,"","save"],[0,2,1,"","set_conditions"],[0,2,1,"","set_time"]],"celltk.core.orchestrator":[[4,1,1,"","Orchestrator"]],"celltk.core.orchestrator.Orchestrator":[[4,2,1,"","add_operations"],[4,2,1,"","build_experiment_file"],[4,2,1,"","load_operations_from_yaml"],[4,2,1,"","run"],[4,2,1,"","save_condition_map_as_yaml"],[4,2,1,"","save_operations_as_yaml"],[4,2,1,"","save_pipelines_as_yamls"],[4,2,1,"","update_condition_map"]],"celltk.core.pipeline":[[4,1,1,"","Pipeline"]],"celltk.core.pipeline.Pipeline":[[4,2,1,"","add_operations"],[4,2,1,"","load_from_yaml"],[4,2,1,"","run"],[4,2,1,"","save_as_yaml"],[4,2,1,"","save_operations_as_yaml"]],"celltk.extract":[[1,1,1,"","Extractor"]],"celltk.extract.Extractor":[[1,2,1,"","extract_data_from_image"]],"celltk.process":[[5,1,1,"","Processor"]],"celltk.process.Processor":[[5,2,1,"","align_by_cross_correlation"],[5,2,1,"","gaussian_filter"],[5,2,1,"","gaussian_laplace_filter"],[5,2,1,"","histogram_matching"],[5,2,1,"","inverse_gaussian_gradient"],[5,2,1,"","make_edge_potential_image"],[5,2,1,"","rolling_ball_background_subtract"],[5,2,1,"","sobel_edge_detection"],[5,2,1,"","unet_predict"],[5,2,1,"","wavelet_background_subtract"],[5,2,1,"","wavelet_noise_subtract"]],"celltk.segment":[[7,1,1,"","Segmenter"]],"celltk.segment.Segmenter":[[7,2,1,"","adaptive_thres"],[7,2,1,"","agglomeration_segmentation"],[7,2,1,"","binary_fill_holes"],[7,2,1,"","clean_labels"],[7,2,1,"","constant_thres"],[7,2,1,"","dilate_to_cytoring_celltk"],[7,2,1,"","expand_to_cytoring"],[7,2,1,"","find_boundaries"],[7,2,1,"","level_set_mask"],[7,2,1,"","misic_predict"],[7,2,1,"","morphological_acwe"],[7,2,1,"","morphological_geodesic_active_contour"],[7,2,1,"","otsu_thres"],[7,2,1,"","random_walk_segmentation"],[7,2,1,"","regional_extrema"],[7,2,1,"","remove_nuc_from_cyto"],[7,2,1,"","unet_predict"],[7,2,1,"","watershed_ift_segmentation"]],celltk:[[1,0,0,"-","extract"],[5,0,0,"-","process"],[7,0,0,"-","segment"],[8,0,0,"-","track"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method"},terms:{"0":[0,1,5,6,7],"01":7,"06_segmentation_and_shape_analysi":7,"1":[1,4,5,7],"10":6,"100":[5,7],"1000":[5,7],"12":7,"16":0,"1d":0,"2":[0,1,5,7],"20":7,"256":7,"2d":0,"3":[0,1,5,6,7],"4":[1,7],"45":7,"5":[0,5,6],"50":7,"5d":0,"6":5,"64":7,"7":7,"8":6,"80":7,"85":7,"975":7,"99":7,"boolean":0,"case":6,"catch":0,"class":[0,1,2,4,5,7],"default":[0,1,4,6],"do":[4,6],"final":6,"function":[0,2,4,5,6,7],"import":[0,2,7],"int":[0,4],"new":[0,4,6,7],"return":[0,1,4,5,7],"true":[0,1,4,5,6,7],A:[0,4],By:6,For:[0,4,6],If:[0,2,4],In:6,Is:7,It:[0,2,6],No:[6,7],Not:4,That:2,Then:7,There:[2,7],These:2,__init__:[],__str__:4,__wrapped__:7,_arr:0,_close_border_hol:7,_format:0,_output_id:1,_split_kei:4,_threshold_dispatcher_funct:7,a1:4,abl:0,abov:[6,7],abs_val:0,accept:7,ad:[2,6],adaptive_thr:7,add:[0,1,4,5,6,7],add_function_to_oper:6,add_metric_slot:0,add_oper:[4,6],add_operations_to_pipelin:[],addit:0,adjac:7,after:7,agglom_max:7,agglom_min:7,agglomer:7,agglomeration_segment:7,algorithm:[0,6],align:4,align_by_cross_correl:5,align_with:5,all:[0,4,5,6,7],allow:[0,1],alon:[2,5],along:[0,5],alpha:5,alreadi:6,also:[0,6],am:2,among:2,amount:7,an:[0,1,2,6],analysi:[2,4],analyz:[2,6],ani:[0,2,6],aniostrop:7,anoth:7,anyth:2,appli:[0,5,7],ar:[0,1,2,4,5,6,7],arbitrari:0,area:7,arg:[0,7],arr:0,arrai:[1,2,4],arrari:0,array_fold:4,assign:7,associ:1,assum:0,attribut:7,auto:7,avg:5,ax:1,axi:0,background:5,bacteri:2,balloon:7,base:[0,5],basearrai:1,batch:[5,7],becaus:7,befor:[4,5,7],being:0,below:7,best:6,beta:[5,7],better:7,between:[0,7],bia:5,bin:5,binary_fill_hol:7,blur:[5,7],bool:[0,4],border:7,both:[0,2,5],bound:4,boundari:7,brighter:7,buffer:7,build:[2,4,6],build_experiment_fil:[0,4],built:0,cach:1,calcul:[2,7],call:[4,6,7],callabl:4,can:[0,2,5,6,7],cast:0,categori:5,cell:[0,1,2],cell_index:0,celltk:[0,1,4,5,6,7],center:7,chang:[0,1],channel0001:6,channel000:[4,6],channel001:4,channel:[0,1,4,6],check:[0,5],checkerboard:7,classmethod:[0,4],clean:[6,7],clean_before_match:7,clean_label:[6,7],clear_bord:7,clone:2,clue:7,colleagu:6,collect:[0,2,4],color:0,com:2,compact:7,concaten:0,cond_map:4,cond_map_onli:4,condit:[0,1,4],condition_map:[0,4],conditionarrai:[0,4,6],config:0,configur:4,confirm:5,conflict:2,connect:[1,7],consecut:[],consid:7,consist:7,constant:[6,7],constant_thr:[6,7],consum:6,contain:[2,4],contour:5,control:4,convert:0,copi:7,core:[0,4],correct:5,could:[5,7],countour:5,coupl:2,creat:[0,2,6,7],crop:5,current:4,custom:7,cyto:1,cyto_mask:7,cytoplasm:7,cytor:7,cytoring_above_adaptive_thr:7,cytoring_above_buff:7,cytoring_above_thr:7,d4:6,data:[0,2,4,6],data_fram:[1,6],datafram:1,db1:5,db4:5,defin:[0,4],delet:0,dep:2,depend:2,deriv:5,design:[0,4],desir:0,determin:7,dict:[0,4],dictat:4,dictionoari:4,did:5,differ:[4,7],difflib:0,diffus:7,digit:0,dilat:7,dilate_to_cytoring_celltk:7,dimens:0,dimension:0,dir:0,directli:[0,7],directori:6,disk:7,dist:7,distanc:7,doesn:0,don:7,done:7,draw:7,dtype:5,e:[0,4],each:[0,4,6,7],easi:6,effici:4,either:0,ellipsi:0,empti:0,en:5,ensur:0,eros:7,err_estim:0,estim:[0,5],etc:[0,1,7],even:[],everyth:6,exampl:[4,6],exclud:4,exist:0,exp:5,expand:[0,7],expand_to_cytor:7,expans:7,experi:[0,4],experiment:6,experimentarrai:[0,4],explanatori:1,ext:6,extens:4,extern:7,extra:6,extract:[1,2,6],extract_data_from_imag:1,extractor:[0,2,3,6],fals:[0,4,5,6,7],faset:5,faster:[5,7],few:6,fewer:0,field:5,figur:[0,7],file:[0,1,4,6],file_extens:4,file_loc:[],fill_bord:7,fill_hol:7,filter:[0,2,5,7],filter_cel:0,filter_util:0,find:[2,6,7],find_boundari:7,first:[0,6,7],fist:6,fitc:[1,6],fix:[0,2],flat:[5,7],float32:5,flow:5,fname:4,focu:4,folder:[4,6],follow:[2,6,7],force_rerun:[1,6],format:6,found:[4,6],fraction:7,frame:[0,1,4,5],frame_rng:[0,4],from:[0,4,5,6,7],fully_connect:7,further:2,gaussian:[5,7],gaussian_filt:5,gaussian_laplace_filt:5,gener:[0,4],generate_mask:0,get:[0,7],git:2,github:2,given:4,go:[2,6],good:[0,2],greater:0,greyscal:7,group:[0,4],guid:2,h5:7,h:5,ha:5,handl:[0,7],happen:7,have:[0,4,6,7],hdf5:[0,4,6],help:7,here:[0,7],hint:0,histogram:5,histogram_match:5,hit:6,hold:6,how:[4,7],howev:7,html:[5,7],http:[2,5,7],i:[0,2,4,7],identifi:[0,4,6],imag:[1,2,4,5,6,7],image_fold:4,img_as_uint:7,img_dtyp:[],implement:[4,5,7],in_plac:7,includ:[2,4,6],increas:4,increment:7,index:[2,4],indic:0,individu:0,info:1,inform:2,initi:[0,6],inner:7,input:[1,5,7],insert:4,insid:5,instead:[0,7],int32:7,intens:5,interpol:0,interpolate_nan:0,interv:0,invers:7,inverse_gaussian_gradi:5,io:5,issu:2,iter:7,job_control:4,jobcontrol:4,just:7,k1:5,k2:5,katharina:6,keep_label:7,kei:0,kernel:5,kernel_radiu:7,kind:0,kit_sch_ge_track:6,know:6,kwarg:[0,1,5,6,7],label:[0,6,7],lambda1:7,lambda2:7,laplac:5,larg:7,larger:2,last:4,layout_spec:0,length:0,less:0,let:6,level:5,level_set_mask:7,levelset:7,light:7,like:[0,2,6,7],line:0,lineag:1,linear:[0,7],link_n4biasfieldcorrection_doc:5,list:0,live:2,load:[0,4],load_condit:0,load_from_yaml:4,load_operations_from_yaml:4,locat:[0,1,4],loeffler:6,log:4,log_fil:4,lot:5,made:[6,7],mahota:7,mak:0,make:[0,5,7],make_edge_potential_imag:5,mammalian:2,mani:4,map:[0,4],margin:7,mask:[0,1,4,5,6,7],mask_fold:4,master:5,match:[0,5,7],match_pt:5,match_str:4,max_length:7,max_radiu:7,maxima:7,meant:0,median_int:1,merg:0,merge_condit:0,metadata:6,method:[4,5,7],metric:[0,1],microcoloni:2,microscopi:2,might:[0,5],min_radiu:[6,7],min_siz:7,min_trace_length:[0,1,6],minima:7,minimum:[0,5],misic:7,misic_predict:7,misicv2:7,mode:[5,7],model:[0,5],model_path:7,modul:2,more:[0,2,5,7],morph:7,morphological_acw:7,morphological_chan_ves:7,morphological_geodesic_active_contour:7,most:6,msak:0,much:7,multidimension:5,multipl:[4,5,6,7],must:0,mutual:2,n4:5,n_core:4,name:[0,1,4,6],nan:0,nansaf:5,nbin:7,ndarrai:[0,1,5,7],ndimag:7,need:[0,5,6,7],neg:7,neighbor:7,newtyp:[1,5,7],next:6,nodep_requir:2,noise_level:5,non:0,none:[0,1,4,5,7],note:7,now:[0,6,7],np:0,nuc:[0,1,6],nuc_mask:7,nuclear:[6,7],nuclei:[6,7],number:[0,5],numpi:5,o:7,object:7,onc:2,onli:[0,4,5,7],open:[2,7],open_s:7,oper:[2,4,6],oper_output:[],optic:5,option:[0,1,4,5,6,7],orchestr:[0,4,6],order:4,org:7,organ:4,orient:5,other:[0,2],otherwis:[0,4,5],otsu:7,otsu_thr:7,our:6,out:[4,7],outlin:7,output:[1,4,5,6,7],output_fold:[4,6],overwrit:[0,4],overwritten:[0,4,7],ovewrit:4,own:6,packag:2,pad:1,page:2,paper:7,par_daught:1,paramet:[0,1,4],parent:[0,1],parent_daught:0,parent_fold:[4,6],parent_track:1,part:2,pass:[0,4,5,6],path:[0,4,6],peak:2,per:[],percentil:0,phase_cross_correl:5,pip:2,pipe:6,pipelin:[2,7],pipeline_dict:[],pixel:7,pleas:2,plot:2,plot_by_condit:0,po:0,popul:1,pos_id:0,posit:4,position_id:1,position_map:4,possibl:7,pre:7,predict:5,predict_peak:0,preprocess:7,preserv:7,prevent:7,probabl:7,problem:2,process:[5,7],processor:[2,3,6],prop_to:0,propag:0,propagate_valu:0,provid:[0,1,4],publish:2,pypi:2,quickstart:2,r:2,radiu:[5,7],rais:0,random:7,random_walk:7,random_walk_segment:7,rang:4,raw:4,re:7,readi:6,readthedoc:5,realli:[],reason:5,recalcul:[],reciproc:5,ref_fram:5,refer:[5,7],regi:5,region:[0,1,6,7],regional_extrema:7,regionprop:[1,7],registr:5,rel:7,relabel:[6,7],relative_thr:7,remov:[0,1,7],remove_empty_sit:0,remove_nuc_from_cyto:7,remove_par:[0,1],remove_short_trac:0,renam:0,repeat:6,requir:[2,6,7],rerun:5,rescal:5,reshape_mask:0,resolv:2,result:0,retrun:4,right:[0,7],ring:7,ringwidth:7,roi:[5,7],roll:5,rolling_ball_background_subtract:5,room:0,round:7,row:0,run:[0,2,4,5,6,7],run_multiple_pipelin:[],s1:5,s:[0,2,6,7],same:[4,5,6,7],save:[0,1,4,6],save_a:6,save_arrai:[],save_as_yaml:4,save_condition_map_as_yaml:4,save_imag:[],save_master_df:4,save_operations_as_yaml:4,save_pipelines_as_yaml:4,scipi:7,search:2,second:5,see:[0,2],seed:7,seed_min_s:7,seed_thr:7,seg:6,seg_thr:7,segment:[0,2,3,6],select:7,self:[0,1],separ:7,sequenti:7,set:[2,4,7],set_condit:0,set_position_id:0,set_tim:0,shape:5,shift:5,should:[0,2,4,5,6,7],show:0,sigma:[5,7],sigmoid:5,similar:7,simpl:[6,7],simpleitk:[5,7],simpli:6,simplifi:2,singl:[0,4,6],site:[0,4],site_0:4,site_2:6,sitk:7,size:7,sjeknic:2,skimag:7,skip:[4,6],skip_fram:[1,4],slice:0,slower:7,small:7,smooth:[5,7],so:[6,7],sobel:[5,7],sobel_edge_detect:5,some:[0,5],someth:0,specif:4,specifi:4,speed:7,spie2019_cours:7,square_s:7,stack:5,stand:[2,5],start:7,step:[2,7],still:2,store:[0,4,6],str:[0,4],strategi:[],string:[0,6],structur:[0,7],subfold:[],svg:0,symmetr:5,t:[0,7],taka:7,take:0,tell:6,test:[5,7],text:4,tf:0,than:[0,7],thei:[6,7],them:[0,4,5,6,7],thi:[0,4,5,6,7],thing:0,think:7,those:[2,6],thre:[5,6,7],threshold:[5,6,7],tif:4,time:[0,1,5,6],time_plot:0,titl:0,tol:7,too:[4,7],tool:2,tra:6,track:[0,1,2,4,5,6],track_fold:4,tracker:[2,3,6],transfer:7,translat:5,tritc:[0,1,6],tupl:[0,4],two:[4,7],txt:2,type:[0,1,4,5,7],typeerror:0,underli:0,unet:6,unet_nuc:6,unet_predict:[5,6,7],union:4,uniq:0,uniqu:[0,4,6],until:2,up:[2,7],updat:0,update_condition_map:4,upeak:0,upeak_example_weight:0,upscal:[],us:[0,1,2,4,5,6,7],user:0,v:5,val:[],val_match:7,valu:[0,1,5,7],valueerror:0,verbos:4,version:7,voronoi:7,w:7,wa:[],wai:[0,2],walk:[0,7],want:[0,6],watersh:7,watershed_dist:7,watershed_ift:7,watershed_ift_segment:7,wavelet:5,wavelet_background_subtract:5,wavelet_noise_subtract:5,we:6,weight:[5,6],weight_path:[0,5,6,7],were:7,when:[0,7],where:[0,4],which:6,whole:5,why:7,within:7,work:[0,2,7],would:[2,6],wrapper:7,write:7,x_label:0,x_limit:0,y_label:0,y_limit:0,yaml:4,yaml_fold:4,yaml_path:4,yet:0,you:[2,4,6],your:6},titles:["Arrays","Extractor","Welcome to CellTK!","Operations","Pipelines","Processor","Quickstart Guide","Segmenter","Tracker"],titleterms:{"class":6,"import":6,arrai:0,celltk:2,content:[2,3],extractor:1,guid:6,indic:2,instal:2,oper:3,pipelin:[4,6],processor:5,quickstart:6,segment:7,set:6,tabl:2,todo:[0,1,4,5,7],tracker:8,up:6,welcom:2}})