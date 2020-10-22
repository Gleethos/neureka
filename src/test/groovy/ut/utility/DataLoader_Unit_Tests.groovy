package ut.utility

import neureka.Neureka
import neureka.Tsr
import neureka.devices.storage.IDXHead
import neureka.dtype.NumericType
import neureka.dtype.custom.F64
import neureka.dtype.custom.UI8
import spock.lang.Specification

class DataLoader_Unit_Tests extends Specification
{

    def setup() {
        File dir = new File( "build/test-can" )
        if ( ! dir.exists() ) dir.mkdirs()
    }

    def 'Test writing IDX file format.'(
        Tsr<?> tensor, Class<NumericType<?,?>> type, String filename, String expected
    ) {

        given : 'Neureka settings are being reset.'
            Neureka.instance().reset()

        when : 'A new IDX file handle for the given filename is being instantiated.'
            IDXHead idx = new IDXHead(tensor, "build/test-can/"+filename)

        then : 'The file will then exist at the following path: '
            new File("build/test-can/"+filename).exists()

        when : 'The "load" method is being called in order to load the tensor into memory.'
            Tsr loaded = idx.load()

        then : 'The loaded tensor is as expected...'
            loaded != null
            loaded.toString() == expected
            loaded.getDataType().getTypeClass() == type

        where : 'The following paths and file names are being used for testing : '
            tensor                  | type      | filename         || expected
            new Tsr([2, 4], -2..4)  | F64.class | "test.idx3-ubyte"|| "(2x4):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -2.0]"
            new Tsr([2, 4], 2)      | F64.class | "test2.idx"      || "(2x4):[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]"
    }


    def 'Test reading IDX file format.'(
            String filename, String expected
    ) {

        given :
            Neureka.instance().reset()
            StringBuilder s = new StringBuilder()

        when :
            IDXHead idx = new IDXHead( "src/test/resources/idx/" + filename )
            Tsr loaded = idx.load()
            int i = 0
            loaded.forEach({ e ->
                def norm = (double)( (int) e& 0xff) / 255
                if ( norm < 0.1 ) s.append(" ")
                else if ( norm < 0.5 ) s.append( "." )
                else if ( norm < 0.7 ) s.append( "*" )
                else s.append( "#" )
                if ( ( i ) % 28==27 ) s.append( "\n" )
                i ++
            })

        then :
            loaded != null
            s.toString().digest('md5') == expected
            loaded.getDataType().getTypeClass() == UI8.class

        and :
            idx.valueSize == 28 * 28
            idx.valueSize == 28 * 28 * 1 // 1 := ubyte
            idx.fileName == filename
            idx.totalSize == 28 * 28 * 1 + 16
            idx.dataType == loaded.dataType

        where :
            filename             || expected
            "MNIST-sample-1.idx" || "88aa2c56cc2304779175e7a8ff382426"
            "MNIST-sample-2.idx" || "f9611a2e2283e8a241276068f29102b8"
            "MNIST-sample-3.idx" || "d7a3f3454a5b4047517cbc584fbfc8f4"
    }

}
