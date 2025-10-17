# Add .. to python path
export PYTHONPATH=../:$PYTHONPATH


##################################
# Fig 1 (left, middle) + Fig S.1 #
#      Generate 1d plots         #
##################################

mkdir -p img/1d_plots
for kernel in 'rbf' 'matern1-2' 'matern3-2'  'matern5-2';
do
  for curve in 2 3;
  do
    python scripts/onedim_curve_fitting.py --kernel $kernel  --curve $curve  --style "styles/one_third_page.mpl"  --save_fig "img/1d_plots/$kernel-curve$curve.pdf"
  done
done

#############################################
#           Generate SNR plots  (Linear)    #
#             Fig S.3 (b/c)                 #
#############################################
mkdir -p out/snr
python scripts/error_vs_snr.py --kernel 'linear' --dataset 'linear'  --n_reps 20 --n_points 20 --dont_plot_figure --csv_file "out/snr/linear.csv"
python scripts/error_vs_snr.py --kernel 'linear' --load  --csv_file "../out/snr/linear.csv" --save_fig "../img/snr/linear.pdf" --style "styles/one_third_page.mpl"

#############################################
#      Generate SNR plots (Nonlinear)       #
#             Fig S.3 (a)                   #
#############################################

mkdir -p out/snr
for dataset in 'sine_1d' 'squarewave';
do
  python scripts/error_vs_snr.py --kernel 'rbf' --dataset $dataset  --n_reps 10 --n_points 10  --dont_plot_figure --csv_file "out/snr/rbf-$dataset.csv"
done

mkdir -p img/snr
for dataset in 'sine_1d' 'squarewave';
do
  python scripts/error_vs_snr.py --kernel 'rbf'  --dataset $dataset  --style "styles/one_third_page.mpl"  --load --csv_file "out/snr/rbf-$dataset.csv" --save_fig "img/snr/rbf-$dataset.pdf"
done

####################################
# Fig 1 (right) + Fig S.2 + Tab. S1 #
#      Fig  error vs sample size   #
####################################

mkdir -p out
mkdir -p out/error_vs_sample_size

for kernel in 'rbf' 'matern1-2' 'matern3-2'  'matern5-2';
do
  echo "Running $kernel"
  python scripts/error_vs_sample_size.py --kernel $kernel --n_points 10 --n_rep 5 --csv_file "out/error_vs_sample_size/$kernel--akr.csv"  "out/error_vs_sample_size/$kernel--kr_cv.csv" --max_log_range 3
done

mkdir -p img
mkdir -p img/error_vs_sample_size
rm out/error_vs_sample_size.csv

for kernel in 'matern1-2' 'matern3-2' 'matern5-2' 'rbf'  ;
do
  python scripts/error_vs_sample_size.py  --kernel $kernel --load --csv_file "out/error_vs_sample_size/$kernel--akr.csv"  "out/error_vs_sample_size/$kernel--kr_cv.csv" \
         --save_fig "img/error_vs_sample_size/$kernel.pdf"  --style styles/one_third_page.mpl \
         --save_summary "out/error_vs_sample_size.csv"
done



##################################
#        Fig 2  + Table 3        #
#      Compute performance       #
##################################

python scripts/get_performance.py --dont_plot_figure --csv_file "out/performance_regr_short.csv"  # now running on hyperion
python scripts/get_performance.py --load_data --figure_dir "../img" --csv_file "../out/performance_regr.csv" --style ../styles/wrapfig.mpl

python styles/print_mytable.py

##################################
#             Rebuttal            #
#    Performance for fixed size   #
##################################

python scripts/get_performance.py --setting rebuttal --dont_plot_figure --csv_file "../out/performance_rebutall.csv"  # now running on hyperion: getp

#############################################
#                 Rebuttal                  #
#        Generate delta/lambda plots        #
#                                           #
#############################################

mkdir -p out/delta-lambda
mkdir -p img/delta-lambda
python scripts/error_vs_delta.py --kernel 'rbf'  --dataset sine_1d  --style styles/vr.mpl --csv_file "out/delta-lambda/rbf-sine_1d.csv" --save_fig "img/delta-lambda/rbf-sine_1d"


