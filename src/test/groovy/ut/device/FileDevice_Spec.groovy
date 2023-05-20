package ut.device

import neureka.Data
import neureka.Neureka
import neureka.Shape
import neureka.Tensor
import neureka.devices.Device
import neureka.devices.file.*
import neureka.devices.host.CPU
import neureka.dtype.DataType
import neureka.dtype.custom.F64
import neureka.dtype.custom.UI8
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("FileDevice, Storing Tensors in Files")
@Narrative('''

    The `FileDevice` class, one of many implementations of the `Device` interface, 
    represents a file directory which can store and load tensors as files (`idx`, `jpg`, `png`...).
    
''')
@Subject([FileDevice, Device])
class FileDevice_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
                    This specification covers the behavior of the `FileDevice`
                    class, which enables the persistence of tensor data.      
            """
    }

    def setup()
    {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 15
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'A file device stores tensors in idx files by default.'(
        String path, String filename
    ) {
        given : 'A new tensor is being created for testing.'
            var a = Tensor.of([2, 4], [5, 4, -7, 3, -2, 6, -4, 3 ])
        and : 'A file device instance is being accessed for a given path.'
            var device = FileDevice.at( path )

        expect : 'Initially the device does not store our newly created tensor.'
            !device.contains(a)

        when : 'Tensor `a` is being stored in the device...'
            device.store( a, filename )

        then : 'The expected file is being created at the given path.'
            new File( path + '/' + filename + '.idx' ).exists()

        and : 'Tensor "a" does no longer have a value (stored in RAM).'
            a.mut.data.getOrNull() == null

        when : 'Freeing the tensor...'
            device.free( a )

        then : 'The file will have been deleted!'
            !new File( path + '/' + filename + '.idx' ).exists()


        where : 'The following parameters have been used:'
            path             | filename
            "build/test-can" | "tensor_2x4_"
    }

    def 'A file device stores tensors in various file formats.'(
            String path, String filename, int[] shape, Class<FileHandle<?,Number>> fileHandleClass, Class<?> dataTypeClass
    ) {
        given : 'A new tensor is being created for testing.'
            var a = Tensor.of( Shape.of(shape), -8d..8d )
        and : 'A String representation of the shape.'
            var shapeStr = String.join('x',(shape as List<Integer>).collect {String.valueOf(it)})
        and : 'A file device instance is being accessed for a given path.'
            var device = FileDevice.at( path )

        expect : 'Initially the device does not store our newly created tensor.'
            !device.contains(a)

        when : 'Tensor "a" is being stored in the device...'
            if ( filename != null ) device.store( a, filename ) else device.store( a )

        then : 'The expected file is being created at the given path.'
            new File( path + '/' + filename ).exists()
                    ||
            new File( path ).listFiles().any {
                it.name.startsWith('tensor_' + shapeStr + '_f64_') && it.name.endsWith('.idx')
            }

        and : 'Tensor "a" does no longer have a value (stored in RAM).'
            a.mut.data.getOrNull() == null

        and : 'The tensor is now of the expected data-type.'
            a.dataType == DataType.of( dataTypeClass )

        and : 'The device contains a "FileHandle" instances of the expected type.'
            device.fileHandleOf( a ).class == fileHandleClass

        when : 'Freeing the tensor...'
            device.free( a )

        then : 'The file will be deleted!'
            !new File( path + '/' + filename ).exists()
            !new File( path ).listFiles().any {it.name.startsWith('tensor_' + shapeStr + '_f64_') }

        where : 'The following parameters are being used:'
            path             | filename            |  shape  || fileHandleClass  | dataTypeClass
            "build/test-can" | "tensor_2x4x3_.idx" | [2,4,3] || IDXHandle.class  | F64.class
            "build/test-can" | "tensor_2x4x3_.jpg" | [2,4,3] || JPEGHandle.class | UI8.class
            "build/test-can" | "tensor_5x3x4_.png" | [5,3,4] || PNGHandle.class  | UI8.class
            "build/test-can" | null                | [2,4,3] || IDXHandle.class  | F64.class
            "build/test-can" | "tensor_4x3_.csv"   | [4,3]   || CSVHandle.class  | String.class
    }


    def 'The file device can load known files in a directory.'()
    {
        given: 'A file device instance is being accessed for a simple test path.'
            def device = FileDevice.at( 'build/resources/test/csv' )
        expect : 'The device contains the expected number of files and it tells as how many files could be loaded.'
            device.loadable.toSet() == ['biostats-without-head.csv', 'biostats.csv'].toSet() // If this fails: consider deleting the build folder!!
            device.loaded == []

        when : 'We load a file...'
            def t = device.load('biostats-without-head.csv').get()

        then : 'The device reports said file as loaded.'
            device.loadable == ['biostats.csv']
            device.loaded == ['biostats-without-head.csv']
            t.toString({ NDPrintSettings it ->
                        it.setHasSlimNumbers(false)
                          .setIsScientific(true)
                          .setIsCellBound(false)
                          .setIsMultiline(true)
                          .setCellSize(15)
                    }
            ) == '(18x5):[\n' +
                 '   (        a       )(       b       )(       c       )(       d       )(       e        ):( biostats-without-head )\n' +
                 '   [      "Alex"    ,           "M"  ,         41     ,           74   ,          170    ]:( 0 ),\n' +
                 '   [      "Bert"    ,           "M"  ,         42     ,           68   ,          166    ]:( 1 ),\n' +
                 '   [      "Carl"    ,           "M"  ,         32     ,           70   ,          155    ]:( 2 ),\n' +
                 '   [      "Dave"    ,           "M"  ,         39     ,           72   ,          167    ]:( 3 ),\n' +
                 '   [      "Elly"    ,           "F"  ,         30     ,           66   ,          124    ]:( 4 ),\n' +
                 '   [      "Fran"    ,           "F"  ,         33     ,           66   ,          115    ]:( 5 ),\n' +
                 '   [      "Gwen"    ,           "F"  ,         26     ,           64   ,          121    ]:( 6 ),\n' +
                 '   [      "Hank"    ,           "M"  ,         30     ,           71   ,          158    ]:( 7 ),\n' +
                 '   [      "Ivan"    ,           "M"  ,         53     ,           72   ,          175    ]:( 8 ),\n' +
                 '   [      "Jake"    ,           "M"  ,         32     ,           69   ,          143    ]:( 9 ),\n' +
                 '   [      "Kate"    ,           "F"  ,         47     ,           69   ,          139    ]:( 10 ),\n' +
                 '   [      "Luke"    ,           "M"  ,         34     ,           72   ,          163    ]:( 11 ),\n' +
                 '   [      "Myra"    ,           "F"  ,         23     ,           62   ,           98    ]:( 12 ),\n' +
                 '   [      "Neil"    ,           "M"  ,         36     ,           75   ,          160    ]:( 13 ),\n' +
                 '   [      "Omar"    ,           "M"  ,         38     ,           70   ,          145    ]:( 14 ),\n' +
                 '   [      "Page"    ,           "F"  ,         31     ,           67   ,          135    ]:( 15 ),\n' +
                 '   [      "Quin"    ,           "M"  ,         29     ,           71   ,          176    ]:( 16 ),\n' +
                 '   [      "Ruth"    ,           "F"  ,         28     ,           65   ,          131    ]:( 17 )\n' +
                 ']'

        when :
            t = device.load('biostats.csv', [firstRowIsLabels:true]).get()

        then :
            device.loadable == []
            device.loaded == ['biostats-without-head.csv', 'biostats.csv']
            t.toString({ NDPrintSettings it ->
                        it.setHasSlimNumbers(false)
                                .setIsScientific(true)
                                .setIsCellBound(false)
                                .setIsMultiline(true)
                                .setCellSize(15)
                    }) == '(18x5):[\n' +
                    '   (      "Name"    )(        "Sex"  )(      "Age"    )(  "Height (in)")( "Weight (lbs)" ):( biostats )\n' +
                    '   [      "Alex"    ,           "M"  ,         41     ,           74   ,          170    ]:( 0 ),\n' +
                    '   [      "Bert"    ,           "M"  ,         42     ,           68   ,          166    ]:( 1 ),\n' +
                    '   [      "Carl"    ,           "M"  ,         32     ,           70   ,          155    ]:( 2 ),\n' +
                    '   [      "Dave"    ,           "M"  ,         39     ,           72   ,          167    ]:( 3 ),\n' +
                    '   [      "Elly"    ,           "F"  ,         30     ,           66   ,          124    ]:( 4 ),\n' +
                    '   [      "Fran"    ,           "F"  ,         33     ,           66   ,          115    ]:( 5 ),\n' +
                    '   [      "Gwen"    ,           "F"  ,         26     ,           64   ,          121    ]:( 6 ),\n' +
                    '   [      "Hank"    ,           "M"  ,         30     ,           71   ,          158    ]:( 7 ),\n' +
                    '   [      "Ivan"    ,           "M"  ,         53     ,           72   ,          175    ]:( 8 ),\n' +
                    '   [      "Jake"    ,           "M"  ,         32     ,           69   ,          143    ]:( 9 ),\n' +
                    '   [      "Kate"    ,           "F"  ,         47     ,           69   ,          139    ]:( 10 ),\n' +
                    '   [      "Luke"    ,           "M"  ,         34     ,           72   ,          163    ]:( 11 ),\n' +
                    '   [      "Myra"    ,           "F"  ,         23     ,           62   ,           98    ]:( 12 ),\n' +
                    '   [      "Neil"    ,           "M"  ,         36     ,           75   ,          160    ]:( 13 ),\n' +
                    '   [      "Omar"    ,           "M"  ,         38     ,           70   ,          145    ]:( 14 ),\n' +
                    '   [      "Page"    ,           "F"  ,         31     ,           67   ,          135    ]:( 15 ),\n' +
                    '   [      "Quin"    ,           "M"  ,         29     ,           71   ,          176    ]:( 16 ),\n' +
                    '   [      "Ruth"    ,           "F"  ,         28     ,           65   ,          131    ]:( 17 )\n' +
                    ']'
    }


    def 'A tensor loaded from a file device can be loaded again.'( Shape shape, Data data )
    {
        given: 'A file device instance is being accessed for a simple test path.'
            def device = FileDevice.at( 'build/resources/test/idx2' )
        and : 'We create a simple tensor which we want to save.'
            var t = Tensor.of(shape, data)
        expect :
            device.directory == 'build/resources/test/idx2'
            t.getDevice() === CPU.get()
            device.loadable.toSet() == [].toSet() // If this fails: consider deleting the build folder!!
            device.loaded == []

        when : 'We save the tensor to the device.'
            device.store( t, 'my-tensor-file.idx' )

        then : 'The device contains the expected tensor.'
            device.has( t )
            t.isOutsourced()
            t.getDevice() === device
            device.loadable.toSet() == [].toSet() // If this fails: consider deleting the build folder!!
            device.loaded == []

        when : 'We load the tensor from the device.'
            device.restore( t )

        then : 'The tensor is restored correctly.'
            !device.has( t )
            !t.isOutsourced()
            t.getDevice() === CPU.get()
            device.loadable.toSet() == [].toSet() // If this fails: consider deleting the build folder!!
            device.loaded == []

        when : 'We store the tensor again.'
            device.store( t, 'my-tensor-file.idx' )

        then : 'The tensor is stored correctly.'
            device.has( t )
            device.loadable.toSet() == [].toSet() // If this fails: consider deleting the build folder!!
            device.loaded == []

        cleanup : 'We delete the file again.'
            new File( 'build/resources/test/idx2/my-tensor-file.idx' ).delete()

        where : 'We use the following shapes and data arrays:'
            shape          |  data
            Shape.of(2, 3) |  Data.of( 1, 2, 3, 4, 5, 6 )
            Shape.of(3, 2) |  Data.of( -1, -2, -3, -4, -5, -6 )
            Shape.of(4)    |  Data.of( 1.1, 2.2, 3.3, 4.4 )
            Shape.of(5)    |  Data.of( 0.3f, 0.4f, 0.5f, 0.6f, 0.7f )
            Shape.of(2)    |  Data.of( 42L, -7L )
    }

}
