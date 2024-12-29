out_path=$"datasets/cifar10_w"
cd $out_path
gdown https://drive.google.com/uc?id=18RKtnxfKGiFTXvDsT4nd9ve04bUt2J3H
gdown https://drive.google.com/uc?id=1yz56SW9NJzqcon-2Jj2Slnb9ANPHbC28
gdown https://drive.google.com/uc?id=1YSGQtxcmpnSiHsveAyR5TR2l_5vCzxKl
gdown https://drive.google.com/uc?id=1b9E7W_gu1845DADALJmB9UulofsoCDoW
gdown https://drive.google.com/uc?id=1aT1zlgEQxuL2FjlZTIZpXDyVgWhUy4Fi
gdown https://drive.google.com/uc?id=1fIVQa0Ma04B2n-_RdwOqQkt2rV7Zdo9U
gdown https://drive.google.com/uc?id=1p3zzK5M4elIXAbcLKNXXSIu3Uiu9SxL5
gdown https://drive.google.com/uc?id=1ITAS0ESiLfP9oRkT3OXoYByI5iDbB9fd
gdown https://drive.google.com/uc?id=113T3SMgIBfZJIw7XxZaXoOfIIUyPU4WL
gdown https://drive.google.com/uc?id=1WravuqPV1UrSEaoP6bAlu7ZvxwbNHMpX
gdown https://drive.google.com/uc?id=1zH8zW1ri6bL4Zgb0lsdZM-EeM_aTZYHO
gdown https://drive.google.com/uc?id=1UMAFveY2gTJcebEn56OUDzhoCtc6uUi7 
gdown https://drive.google.com/uc?id=1cQyStZe0u-t9sjjUitqioP5OO9Qfalga

# get downloaded files and extract them in the same directory
for file in `ls *.zip`
do
    unzip $file
    rm $file
done