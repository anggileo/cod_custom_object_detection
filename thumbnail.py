#membuat thumbnail
import glob
from PIL import Image
for infile in glob.glob("*.jpg"):
	im = Image.open(infile)
	im.thumbnail((28,28), Image.ANTIALIAS)
	if infile[0:2] != "asu":
		im.save("asu" + infile, "JPEG")
