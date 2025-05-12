##################################
# Fig 1 (left, middle) + Fig S.1 #
#      Generate 1d plots         #
##################################

mkdir -p img/1d_plots
for kernel in 'rbf' 'matern1-2'  'matern5-2';
do
  for curve in 2 3;
  do
    python onedim_curve_fitting.py --kernel $kernel  --curve $curve  --style styles/vr.mpl  --save_fig "img/1d_plots/$kernel-curve$curve.pdf"
  done
done
#############################################
#           Generate SNR plots  (Linear)    #
#             Fig S.3 (b/c)                 #
#############################################
mkdir -p out/snr
python error_vs_snr.py --kernel 'linear' --dataset 'linear'  --n_reps 20 --n_points 20 --dont_plot_figure --csv_file "out/snr/linear.csv"

python error_vs_snr.py --kernel 'linear' --load  --csv_file "out/snr/linear.csv" --save_fig "img/snr/linear.pdf" --style styles/vr.mpl


#############################################
#      Generate SNR plots (Nonlinear)       #
#             Fig S.3 (a)                   #
#############################################
mkdir -p out/snr
for dataset in 'sine_1d' 'squarewave';
do
  python error_vs_snr.py --kernel 'rbf' --dataset $dataset  --n_reps 10 --n_points 10  --dont_plot_figure --csv_file "out/snr/rbf-$dataset.csv"
done


mkdir -p img/snr
for dataset in 'sine_1d' 'squarewave';
do
  python error_vs_snr.py --kernel 'rbf'  --dataset $dataset  --style styles/vr.mpl  --load --csv_file "out/snr/rbf-$dataset.csv" --save_fig "img/snr/rbf-$dataset.pdf"
done



####################################
# Fig 1 (right) + Fig S.2 + Tab. 3 #
#      Fig  error vs sample size   #
####################################
mkdir -p out
mkdir -p out/error_vs_sample_size

for method in 'kr_cv' 'akr';
do
  for kernel in 'rbf' 'matern1-2'  'matern5-2';
  do
    python error_vs_sample_size.py --kernel $kernel --estimate $method --n_points 10 --n_rep 5 --csv_file "out/error_vs_sample_size/$kernel--$method.csv"
  done
done


mkdir -p img
mkdir -p img/error_vs_sample_size
rm out/error_vs_sample_size.csv
for method in 'kr_cv' 'akr';
do
  for kernel in 'rbf' 'matern1-2'  'matern5-2';
  do
    python error_vs_sample_size.py  --kernel $kernel --estimate $method --load --csv_file "out/error_vs_sample_size/$kernel--$method.csv" \
           --save_fig "img/error_vs_sample_size/$kernel--$method.png"  --style styles/vr.mpl \
           --save_summary "out/error_vs_sample_size.csv"
  done
done

python styles/print_mytable.py


##################################
#             Fig X              #
#      Compute performance        #
##################################

# Now running on hyperion: seff
python get_performance.py --dont_plot_figure  # now running on hyperion
python get_performance.py --load_data --csv_file "out/performance_regr.csv" --style styles/vr.mpl


##############################
# Fig X error vs sample size #
##############################
# now running on hyperion: seff
for method in 'akr' 'kr_cv';
do
  python error_vs_sample_size_linear.py  --csv_file "out/error_vs_sample_size/linear--$method.csv" --dont_plot_figure --estimate $method --max_log_range 2 --n_reps 10
done;

python error_vs_sample_size_linear.py  --csv_file "out/error_vs_sample_size/linear--akr.csv" --load

#######################
# Plot grant Figuress #
#######################

python onedim_curve_fitting.py --curve 2 --kernel 'matern5-2'
python onedim_curve_fitting.py --curve 3 --kernel 'matern5-2'
python error_vs_sample_size.py  --kernel 'matern5-2' --estimate 'akr' --load --csv_file "out/error_vs_sample_size/matern5-2--akr.csv" --save_fig "img/error_vs_sample_size/matern5-2--akr.png" --style styles/vr.mpl