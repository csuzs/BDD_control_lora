for yaml_dir in bdd_bycicle_synth_data/p*; do
    python train.py --data ${yaml_dir} --batch 300 --name "only_bycicles_synth_lr001/${yaml_dir}" --lr=0.001 --workers 12 --epochs 250 --imgsz 640 --classes "7"
done