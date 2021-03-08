#!/bin/bash
cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
cls=${cls_lst[0]}
python -m train.train_linemod_pvn3d --cls ${cls} -checkpoint train_log/linemod/checkpoints/ape/ape_pvn3d_best.pth.tar
