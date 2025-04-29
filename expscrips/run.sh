cd ..
dataset=${1:-"vortex"}

gui=false
test=false

bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m nerf  
bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m extract  
bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m palette 

if ([ "$gui" = true ] && [ "$test" = false ]); then
    bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m distill -g
elif ([ "$gui" = false ] && [ "$test" = false ]); then
    bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m distill
elif ([ "$gui" = false ] && [ "$test" = true ]); then
    bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m distill -t
elif ([ "$gui" = true ] && [ "$test" = true ]); then
    bash scripts/run_blender.sh scripts/configs/${dataset}.sh -m distill -g -t
fi
