package neureka.devices.storage;

import neureka.Tsr;
import neureka.dtype.DataType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

public class JPEGHead implements FileHead<JPEGHead, Number>
{
    private final String _fileName;

    public JPEGHead( String fileName )
    {
        _fileName = fileName;
        try {
            _loadHead( fileName );
        } catch( Exception e ) {
            System.err.print("Failed reading IDX file!");
        }
    }

    private void _loadHead( String fileName )
    {

        File found = new File(fileName);

        BufferedImage image = null;

        try
        {
            image = ImageIO.read( found );

            Raster data = image.getData();
            int height = data.getHeight();
            int width = data.getWidth();
            int[] value = new int[ width * height * 3 ];
            data.getPixels(0, 0, width, height, value);

            //ImageIO.write(image, "jpg", new File("I:/output.jpg"));
            //ImageIO.write(image, "png", new File("I:/output.png"));
            //ImageIO.write(image, "gif", new File("I:/output.gif"));
            //ImageIO.write(image, "bmp", new File("I:/output.bmp"));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        System.out.println("done");

    }

    @Override
    public Tsr<Number> load() throws IOException {
        return null;
    }

    @Override
    public JPEGHead free() throws IOException {
        return null;
    }

    @Override
    public int getValueSize() {
        return 0;
    }

    @Override
    public int getDataSize() {
        return 0;
    }

    @Override
    public int getTotalSize() {
        return 0;
    }

    @Override
    public String getFileName() {
        return null;
    }

    @Override
    public DataType getDataType() {
        return null;
    }

    @Override
    public int[] getShape() {
        return new int[0];
    }

    @Override
    public Storage store( Tsr<Number> tensor ) {
        return null;
    }

    @Override
    public Storage restore( Tsr<Number> tensor ) {
        return null;
    }
}
