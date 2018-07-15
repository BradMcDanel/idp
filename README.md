## Training MLP with different coefficient functions

```
python train.py --lr 0.01 --coeff-type all-one --output models/all-one.pth
python train.py --lr 0.01 --coeff-type linear --output models/linear.pth
python train.py --lr 0.01 --coeff-type harmonic --output models/harmonic.pth
```

## Evaluating classification accuracy over different IDP settings
```
python sweep_idp.py
```
