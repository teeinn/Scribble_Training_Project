# Resume training
#python3 tree_train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 4 --epochs 300 --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 2 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --resume /home/qisens/2020.3~/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/experiment_9/checkpoint_epoch_232.pth.tar

# Start training
python3 tree_train_withdensecrfloss.py --backbone resnet --lr 0.007 --workers 4 --epochs 100 --batch-size 8 --checkname resnet --eval-interval 2 --dataset tree --save-interval 1 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100



