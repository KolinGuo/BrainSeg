import java.nio.file.*;
import java.io.*;

import qupath.lib.display.ChannelDisplayInfo;
import qupath.lib.display.ImageDisplay;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.color.ColorDeconvolutionStains;
import qupath.lib.color.StainVector;

// Get current QuPath viewer and image data
QuPathViewer viewer = getCurrentViewer();
ImageDisplay display = viewer.getImageDisplay();
ImageData imageData = getCurrentImageData();
ImageServer server = imageData.getServer();
String imagePathStr = server.getURIs()[0];

// Get imagePath and dirPath for saving
Path imagePath = Paths.get(new URI(imagePathStr));
String imageName = imagePath.getFileName().toString();
String dirName = "stain_contrast";
Path dirPath = Paths.get(imagePath.getParent().toString(), dirName);
// Create directory if not exists
File dir = new File(dirPath.toString());
dir.mkdirs();
// Create filePath for saving
Path filePath = Paths.get(dirPath.toString(), imageName.split("\\.")[0]+".txt");
print("Processing " + imageName + " and saved to " + filePath.toString());

// Get stain vectors
ColorDeconvolutionStains stains = imageData.getColorDeconvolutionStains();
StainVector stain1 = stains.getStain(1);
StainVector stain2 = stains.getStain(2);
StainVector stain3 = stains.getStain(3);
print("Color deconvolution stains: ");
print("\t"+stain1.getName() + ": \t" + stain1.getArray());
print("\t"+stain2.getName() + ": \t" + stain2.getArray());
print("\t"+stain3.getName() + ": \t" + stain3.getArray());

int bk_red = stains.getMaxRed();
int bk_green = stains.getMaxGreen();
int bk_blue = stains.getMaxBlue();
print("Background RGB: \t["+bk_red+", "+bk_green+", "+bk_blue+"]");

// Get current selected channel's contrast (min_display and max_display)
ChannelDisplayInfo channel = display.selectedChannels()[0];
double min_display = channel.getMinDisplay();
double max_display = channel.getMaxDisplay();
print("Selected channel: " + channel.toString());
print("\t[Min, Max]: \t[" + min_display.toString() + ", " + max_display.toString() + "]");

// Write to file
File f = new File(filePath.toString());
f.createNewFile();    // if file already exists will do nothing 
out = new BufferedWriter(new FileWriter(f, false));

// Write WSI name
out.write(imageName + "\n");
out.write("Color deconvolution stains: " + "\n");
out.write(stain1.getName() + ": " + stain1.getArray() + "\n");
out.write(stain2.getName() + ": " + stain2.getArray() + "\n");
out.write(stain3.getName() + ": " + stain3.getArray() + "\n");
out.write("Background RGB: ["+bk_red+", "+bk_green+", "+bk_blue+"]\n");
out.write("Selected channel: " + channel.toString() + "\n");
out.write("[Min, Max]: [" + min_display.toString() + ", " + max_display.toString() + "]" + "\n");

// Close the BufferedWriter
if(out != null) {
    out.close();
}