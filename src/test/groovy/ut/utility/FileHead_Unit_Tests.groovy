package ut.utility

import neureka.Neureka
import neureka.Tsr
import neureka.devices.storage.CSVHead
import neureka.devices.storage.IDXHead
import neureka.devices.storage.JPEGHead
import neureka.dtype.DataType
import neureka.dtype.NumericType
import neureka.dtype.custom.F64
import neureka.dtype.custom.I16
import neureka.dtype.custom.UI8
import neureka.utility.TsrAsString
import spock.lang.Specification

class FileHead_Unit_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
            This specification covers the expected functionality of
            various "FileHead" implementations.
            Such implementations ought to be able to save tensors to
            a given directory in the file format that they represent.
            Functionalities like : "store", "restore" and "free" must
            behave as expected.
            (For more information take a look a the "FileHead" & "Storage" interface)
        """
    }

    def setup() {
        Neureka.instance().reset() 

        File dir = new File( "build/test-can" )
        if ( ! dir.exists() ) dir.mkdirs()
    }

    def 'Test writing IDX file format.'(
        Tsr<?> tensor, Class<NumericType<?,?,?,?>> type, String filename, String expected
    ) {
        given:
            Neureka.instance().settings().view().asString = TsrAsString.configFromCode("dgc")

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
            tensor                  | type      | filename          || expected
            new Tsr([2, 4], -2..4)  | F64.class | "test.idx3-ubyte" || "(2x4):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -2.0]"
            new Tsr([2, 4], 2)      | F64.class | "test2.idx"       || "(2x4):[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]"
    }


    def 'Test reading IDX file format.'(
            String filename, String expected
    ) {

        given : 'A variable for storing a hash :'
            def hash = ""

        when : 'The given idx file is being loaded by the "IDXHead" class into a new tensor...'
            IDXHead idx = new IDXHead( "build/resources/test/idx/" + filename )
            Tsr loaded = idx.load()
        and : '... this new tensor is then hashed ...'
            loaded.forEach( e -> hash = ( hash + e ).digest("md5") )
            /*
            // Use the following for an ASCII respresentation of the tensor :
            int i = 0
            loaded.forEach({ e ->
                def norm = (double)( (int) e& 0xff) / 255
                if ( norm < 0.1 ) print(" ")
                else if ( norm < 0.5 ) print( "." )
                else if ( norm < 0.7 ) print( "*" )
                else s.append( "#" )
                if ( ( i ) % 28==27 ) print( "\n" )
                i ++
            })
            */
        then : 'The hash is as expected.'
            hash == expected
        and : 'The loaded tensor has the expected data type.'
            loaded.dataType.getTypeClass() == I16.class
            loaded.dataType == DataType.of( I16.class )
        and : 'It contains the correct array type.'
            loaded.data instanceof short[]

        and : 'The "IDXHead" instance has the expected state :'
            idx.valueSize == 28 * 28
            idx.valueSize == 28 * 28 * 1 // 1 := ubyte
            idx.fileName == filename
            idx.location.endsWith( filename )
            idx.totalSize == 28 * 28 * 1 + 16
            idx.dataType != loaded.dataType
            idx.dataType == DataType.of( UI8.class ) // The underlying data is unsigned byte! (Not supported by JVM)


        where : 'The following files and the expected hashes of their data were used :'
            filename             || expected
            "MNIST-sample-1.idx" || "c74e87c7a93605e7a1660ec9e17dcf9f"
            "MNIST-sample-2.idx" || "4a57297981456a467a302c8738b3ac50"
            "MNIST-sample-3.idx" || "87eade8bb5659d324030f4e84f6745e7"
    }


    def 'The FileDevice component "JPEGHead" can read JPG file formats and load them as tensors.'(
            String filename, List<Integer> shape, String expected
    ) {
        given :
            def hash = ""

        when :
            JPEGHead jpg = new JPEGHead( "build/resources/test/jpg/" + filename )
            Tsr loaded = jpg.load()
            loaded.forEach(e -> hash = ( hash + e ).digest('md5') )
            /*
            // Use the following code to get an ASCII representation of the image (from the loaded tensor):
            int i = 0
            float pixel = 0
            loaded.forEach({ e ->
                def norm = (double)( (int) e& 0xff) / (255*3)
                pixel += norm
                if (  ( i ) % 3==2 ) {
                    if (pixel < 0.1) print(" ")
                    else if ( pixel < 0.2 ) print("`")
                    else if ( pixel < 0.5 ) print(".")
                    else if ( pixel < 0.7 ) print("*")
                    else print("#")
                    if ((i) % (loaded.shape(1)*3) == (loaded.shape(1)*3-1)) print("\n")
                    pixel = 0
                }
                i ++
            })
            */

        then :
            loaded != null
            !loaded.isVirtual()
            loaded.size() == shape.inject( 1, {prod, value -> prod * value} )
            loaded.getDataType().getTypeClass() == I16.class // Auto convert! (stored as I16)
            hash == expected

        and :
            jpg.shape == shape as int[]
            jpg.valueSize == shape.inject( 1, {prod, value -> prod * value} )
            jpg.totalSize == shape.inject( 1, {prod, value -> prod * value} ) //28 * 28 * 1 + 16
            jpg.location.endsWith( filename )
            jpg.dataType == DataType.of( UI8.class )
            loaded.dataType == DataType.of( I16.class )

        where : 'The following jpg files with their expected shape and hash were used.'
            filename           || shape          | expected
            "small.JPG"        || [260, 410, 3]  | "b0e336b03f2ead7297e56b8ca050f34d"
            "tiny.JPG"         || [10, 46, 3]    | "79bf5dd367b5ec05603e395c41dafaa7"
            "super-tiny.JPG"   || [3, 4, 3]      | "a834038d8ddc53f170fa426c76d45df2"
    }

    def 'The FileDevice component "CSVHead" can read CSV file formats and load them as tensors.'(
            String filename, Map<String, Object> params, int byteSize, List<Integer> shape, String expected
    ) {
        given :
            def hash = ""

        when :
            CSVHead csv = new CSVHead( "build/resources/test/csv/" + filename, params )
            Tsr loaded = csv.load()
            loaded.forEach( e -> hash = ( hash + e ).digest('md5') )
            println(loaded)
        then :
            loaded != null
            !loaded.isVirtual()
            loaded.size() == shape.inject( 1, {prod, value -> prod * value} )
            loaded.getDataType().getTypeClass() == String.class // Auto convert! (stored as String)
            hash == expected

        and :
            csv.shape == shape as int[]
            csv.valueSize == shape.inject( 1, {prod, value -> prod * value} )
            csv.totalSize == byteSize
            csv.dataSize == byteSize
            csv.location.endsWith( filename )
            csv.dataType == DataType.of( String.class )
            loaded.dataType == DataType.of( String.class )

        where : 'The following jpg files with their expected shape and hash were used.'
            filename      | params                  || byteSize | shape    | expected
            "biostats.csv"| [:]                     || 753      | [19, 5]  | "96aacf13cc7be153945722b31632b36a"
            "biostats.csv"| [firstRowIsLabels:true] || 702      | [18, 5]  | "ddece2f325da089d191ef2efb1df3ba4"
            "biostats.csv"| [firstColIsIndex:true]  || 639      | [19, 4]  | "c2bc090f87304b3844b83fa7bf2bba38"
    }


}
