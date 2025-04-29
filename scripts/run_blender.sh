
CONFIGFILE=$1;
shift

if [ $# -eq 0 ]; then
    echo "Error: a config file is required."
    exit
fi
if [ ! -f "$CONFIGFILE" ]; then
    echo "Error: $CONFIGFILE does not exist."
    exit
fi
source $CONFIGFILE

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--test)
      test=True
      shift # past argument
      ;;    
    -v|--video)
      video=True
      shift # past argument
      ;;
    -g|--gui)
      gui=True
      shift # past argument
      ;;
    -m|--model)
      model="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

test_mode=''

if [ $gui ]; then
    # test_mode='--test --gui'
    test_mode=${test_mode}'--gui'
fi

if [ $test ]; then
    test_mode=${test_mode}' --test'
fi


if [[ $model == 'nerf' ]]; then
    # density_thresh=1
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py \
    $data_dir \
    --workspace ${name} \
    --iters ${iters} \
    --bound ${bound} \
    --offset ${offset} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --no_bg \
    --density_thresh ${density_thresh} \
    -O \
    --dt_gamma 0 \
    $test_mode
elif [[ $model == 'extract' ]]; then
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
    $data_dir \
    $nerf_model \
    -O \
    --bound ${bound} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh}  \
    --extract_palette \
    --use_normalized_palette
elif [[ $model == 'palette' ]]; then
    # density_thresh=1
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
    $data_dir \
    $nerf_model \
    -O \
    --iters 2500 \
    --bound ${bound} \
    --scale ${scale} \
    --offset ${offset} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    --random_size ${random_size} \
    --use_initialization_from_rgbxy \
    --use_normalized_palette \
    --dt_gamma 0 \
    --datatype "blender" \
    $test_mode
elif [[ $model == 'distill' ]]; then
    # density_thresh=1
    # tic=$(date +%s)
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_distill.py \
    $data_dir \
    $palette_model \
    -O \
    -s ${style} \
    --iters 500 \
    --stage1_iters 150 \
    --bound ${bound} \
    --scale ${scale} \
    --offset ${offset} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    --random_size ${random_size} \
    --dt_gamma 0 \
    --datatype "blender" \
    $test_mode
    toc=$(date +%s)
    # echo "Time elapsed: $((toc - tic)) seconds on " >> ${palette_model}/time.txt
else
    echo "Invalid model. Options are: nerf, extract, palette"
fi