package neureka.acceleration.storage;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/*
    How the IDX file format is being read:
    The IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.

    The basic format according to http://yann.lecun.com/exdb/mnist/ is:
    magic number
    size in dimension 1
    size in dimension 2
    size in dimension 3
    ....
    size in dimension N
    data

    The magic number is four bytes long. The first 2 bytes are always 0.

    The third byte codes the type of the data:
    0x08: unsigned byte
    0x09: signed byte
    0x0B: short (2 bytes)
    0x0C: int (4 bytes)
    0x0D: float (4 bytes)
    0x0E: double (8 bytes)

    The fourth byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

    The sizes in each dimension are 4-byte integers (big endian, like in most non-Intel processors).

    The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.




 */



public class MNISTLoader
{
    private final String TRAINING_LABLES = "Data/train-labels.idx1-ubyte";
    private final String TRAINING_IMAGES = "Data/train-images.idx3-ubyte";

    private final String TEST_LABELS = "Data/t10k-labels.idx1-ubyte";
    private final String TEST_IMAGES = "Data/t10k-images.idx3-ubyte";

    public MNISTLoader(){ }


    public double[][] getTrainingsImages(){
        try {
            return normalize(readImagesAsBytes(TRAINING_IMAGES));
        }catch (Exception e){
            return null;
        }
    }

    public  double[][] getTrainingsLabels(){
        try {
            return readLabels(TRAINING_LABLES);
        }catch (Exception e){
            return null;
        }
    }

    public double[][] readTestImages(){
        try {
            return normalize(readImagesAsBytes(TEST_IMAGES));
        }catch (Exception e){
            return null;
        }
    }

    public double[][] readTestLabels(){
        try {
            return readLabels(TEST_LABELS);
        }catch (Exception e){
            return null;
        }
    }

    public double[][] readImagesNormalized(String file) throws IOException {
        return normalize(readImagesAsBytes(file));
    }



    public byte[] __readImagesAsBytes(String file) throws IOException
    {
        FileInputStream f = null;
        try {
            f = new FileInputStream(file);
        } catch (FileNotFoundException e) {
            System.err.println("File: " + file + " not found.");
            return null;
        }
        NumberReader numre = new NumberReader(f);

        int zeros = numre.readIntegerInByteNumber((byte) 2);
        assert zeros == 0;

        int dtype = numre.readIntegerInByteNumber((byte) 1);
        // TODO : interpret dtype

        int rank = numre.readIntegerInByteNumber((byte)1);
        int[] shape = new int[rank];

        int size = 1;
        for ( int i = 0; i < rank; i++ ) {
            shape[i] = numre.readIntegerInByteNumber((byte)4);
            size *= shape[i];
        }

        byte[] data = new byte[size];

        assert f.read(data) == data.length;
        f.close();
        return data;
    }

    public byte[][] readImagesAsBytes(String file) throws IOException
    {
        int magicNumber; // 0-3 byte
        int size;        // 4-7 byte
        int rows;        // 8-11
        int columns;     // 12-15
        byte[][] images; // every next byte is one px, every img contains rows*columns px, there are size amount of img

        FileInputStream f = null;
        try {
            f = new FileInputStream(file);
        } catch (FileNotFoundException e) {
            System.err.println("File: " + file + " not found.");
            return null;
        }
        final byte[] integer = new byte[4];

        assert f.read(integer) == 4;
        magicNumber = _byteArrayToInt(integer);

        if(magicNumber != 2051) throw new IOException("Not a valid file");

        assert f.read(integer) == 4;
        size = _byteArrayToInt(integer);

        assert f.read(integer) == 4;
        rows = _byteArrayToInt(integer);

        assert f.read(integer) == 4;
        columns = _byteArrayToInt(integer);

        images = new byte[size][rows*columns];

        int nrImg = 0;

        while( nrImg < size ) {
            byte[] image = new byte[rows*columns];
            assert f.read(image) == image.length;
            images[nrImg] = image;
            if(images[nrImg][0] == -1) return images;

            nrImg++;
        }
        f.close(); // added
        return images;
    }

    private double[][] readLabels(String file) throws IOException
    {
        int magicNumber;
        int length;
        double[][] labels;
        byte[] integer = new byte[4];
        FileInputStream f = new FileInputStream(new File(file));

        assert f.read(integer) == 4;
        magicNumber = _byteArrayToInt(integer);

        if(magicNumber != 2049){
            throw new RuntimeException("Wrong File");
        }
        f.read(integer);
        length = _byteArrayToInt(integer);

        labels = new double[length][10];
        for(int i = 0; i < labels.length; i++){
            double[] label = new double[10];
            label[f.read() & 0xff] = 1;
            labels[i] = label;
        }

        f.close(); // added
        return labels;

    }

    private int _byteArrayToInt(byte[] b) { // This views the given bytes as unsigned!
        if (b.length == 4) return b[0] << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8 | (b[3] & 0xff);
        else if (b.length == 2) return 0x00 << 24 | 0x00 << 16 | (b[0] & 0xff) << 8 | (b[1] & 0xff);
        else if (b.length == 1) return b[0] & 0xFF;
        return 0;
    }

    public static String printDigit(double[] img){
        return imageToString(img);
    }

    public double[][] normalize(byte[][] img){
        double[][] dImg = new double[img.length][img[0].length];
        for(int i = 0; i < img.length; i++){
            for(int j = 0; j < img[i].length; j++){
                dImg[i][j] = (double)( (int) img[i][j]& 0xff)/255;
            }
        }
        return dImg;
    }

    private double[] normalize(byte[] img){
        double[] dImg = new double[img.length];
        for(int i = 0; i < img.length; i++){
            dImg[i] = (double)( (int) img[i]& 0xff)/255;
        }
        return dImg;
    }

    public static String imageToString(double[] img){
        StringBuilder s = new StringBuilder();
        for(int i = 0; i < img.length; i++)
        {
            if(img[i] < .1) s.append(" ");
            else if(img[i] < .5) s.append(".");
            else if (img[i] < .7) s.append("*");
            else s.append("#");
            if(i%28 == 27) s.append("\n");
        }
        return s.toString();
    }



}
