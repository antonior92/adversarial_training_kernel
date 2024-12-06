
mkdir -p out
mkdir -p out/error_vs_sample_size

# now running on hyperion: seff
for method in 'kr_cv' 'akr';
do
  for kernel in 'rbf' 'matern1-2'  'matern5-2';
  do
    python error_vs_sample_size.py --kernel $kernel --estimate $method --n_points 10 --n_rep 5 --csv_file "out/error_vs_sample_size/$kernel--$method.csv"
  done
done


mkdir -p img
mkdir -p img/error_vs_sample_size
for method in 'kr_cv' 'akr';
do
  for kernel in 'rbf' 'matern1-2'  'matern5-2';
  do
    python error_vs_sample_size.py  --kernel $kernel --estimate $method --load --csv_file "out/error_vs_sample_size/$kernel--$method.csv" --save_fig "img/error_vs_sample_size/$kernel--$method.png"
  done
done