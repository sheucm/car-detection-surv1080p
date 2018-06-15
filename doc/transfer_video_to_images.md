# How to transfer video to images

Output a single frame from the video into an image file:
```` bash
ffmpeg -i input.avi -ss 00:00:14.435 -vframes 1 out.jpg
````

Output one image every second, named out1.png, out2.png, out3.png, etc.
```` bash
ffmpeg -i input.avi -vf fps=1 out%d.jpg
````

Output one image every minute, named img001.jpg, img002.jpg, img003.jpg, etc.
```` bash
ffmpeg -i input.avi -vf fps=1/60 img%03d.jpg
````

Output one image every ten minutes:
```` bash
ffmpeg -i test.flv -vf fps=1/600 thumb%04d.bmp
````

