# Yolo_v1_scratch

There are some results on the test data.

![r2](https://user-images.githubusercontent.com/75822824/144744641-91a93a42-12e4-4395-bb1c-31755668a768.png)
![r1](https://user-images.githubusercontent.com/75822824/144744613-600b52da-c0f6-4fd3-9269-1cc3d9740883.png)
![r3](https://user-images.githubusercontent.com/75822824/144744614-4ed22722-27c1-4e04-82fb-1c8b171ef947.png)

An inplace operation in the loss function.

//loss for the width and height is SUM( (root(h) - root(h^))^2 + (root(w) - root(w^))^2 )

box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
torch.abs(box_predictions[..., 2:4]) + 1e-6) 

//width and height may have negative values in starting
//to avoide the error we has use absolute value and using sign because after sqrt the sign will diminish.

box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4]) # target will not have negative (ofcourse).



All I learned this from Aladdin Persson
