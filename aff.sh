for x in DRAW4-dual-shared-keep DRAW4-dual-shared-stopgrad DRAW4-dual-shared DRAW4-dual-shared DRAW4-noctx DRAW4-shared-attentiononly DRAW4-shared-fixedbiasonly DRAW4dual-pascal DRAW5-dual-shared DRAW5-shared-nocenter above-below conv-block-blur vanilla-vgg
 
do
python affinity.py --trial=128 $x COCO arch 7 --timestep=29999
done
