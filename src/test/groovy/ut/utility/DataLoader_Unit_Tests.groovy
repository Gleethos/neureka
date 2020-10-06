package ut.utility

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.storage.IDXHead
import neureka.acceleration.storage.MNISTLoader
import spock.lang.Specification

class DataLoader_Unit_Tests extends Specification
{
    /*
    def 'Test MNIST loading.'(){
        given :
            Neureka.instance().reset()
            MNISTLoader m = new MNISTLoader()

        when :
            byte[][] b = m.readImagesAsBytes("Data/train-images.idx3-ubyte")
            double[][] d = m.normalize(b)

        then :
            m.printDigit(d[0]).digest("md5") == "88aa2c56cc2304779175e7a8ff382426"
    }

    def 'Test reading IDX file format.'(){

        given :
        Neureka.instance().reset()
        IDXHead idx = new IDXHead("Data/train-images.idx3-ubyte")

        when :
            Tsr t = idx.load()

        then :
            t != null
            t.toString().contains("(60000x28x28)")


    }
    */

    /*
    def 'some tests'(){
        given :
            byte b = 1
            int i = 0xFF

        when :
            b = b << 7

        then :
            b == (byte) -128
            (((int)b) & 0xFF) == 128
            i == 255
            (0..255).each {
                String s1 = String.format(
                        "%8s",
                        Integer.toBinaryString(((byte)it) & 0xFF)
                ).replace(' ', '0');
                String s2 = String.format(
                        "%8s",
                        Integer.toBinaryString(((byte)it))
                ).replace(' ', '0');
                //Integer.toBin
                println(it+" : "+s1+" : "+s2+" : "+((byte)it))
            }
    }
    */


}
