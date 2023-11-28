# Denoising Medical Image
Denoising Medical Image with Difference GAN

## 💡 Discription
```Diffrence GAN``` refers to a GAN that trains adversarially with a difference map to be denoised like a Target (ndct).  
Since Input (qdct) and Target (ndct) are similar, Output (pred) can also look like Input, but considering diff, there is an effect of making it closer to Target.

### The network pipeline.  
![diff_gan](https://github.com/SkiddieAhn/HW-Denoising-Image/assets/52392658/8e8b8d87-fb47-4419-8a18-88eea2904a52)

## 📖 Results
|                       |PSNR    |SSIM   |
|:--------------:|:-----------:|:-----------:|
| **ndct-qdct**  |    33.78    |0.8532|
| **ndct-pred**  |    37.43    | 0.9225 |
| **percentage**  |   **+10.8%**   | **+8.12%**|

### Visualization  
![image](https://github.com/SkiddieAhn/HW-Denoising-Image/assets/52392658/31ddfe03-b4f9-4458-aa09-8c987d27e057)
### Heatmap
![image](https://github.com/SkiddieAhn/HW-Denoising-Image/assets/52392658/ff3b79ab-d6b1-4f05-86ca-399f4ee5ffa9)


