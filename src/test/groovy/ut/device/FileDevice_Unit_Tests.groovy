package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.file.FileDevice
import neureka.devices.file.FileHead
import neureka.devices.file.heads.*
import neureka.dtype.DataType
import neureka.dtype.custom.F64
import neureka.dtype.custom.I16
import spock.lang.Specification

class FileDevice_Unit_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
                <h2> FileDevice Behavior </h2>
                <br> 
                <p>
                    This specification covers the behavior of the "FileDevice"
                    class, which enables the persistence of tensor data.           
                </p>
            """
    }

    def setup()
    {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors = "dgc"
    }

    def 'A file device stores tensors in idx files by default.'(
        String path, String filename
    ) {
        given : 'A new tensor is being created for testing.'
            Tsr a = Tsr.of([2, 4], [ 5, 4, -7, 3, -2, 6, -4, 3 ])
        and : 'A file device instance is being accessed for a given path.'
            def device = FileDevice.at( path )

        expect : 'Initially the device does not store our newly created tensor.'
            !device.contains(a)

        when : 'Tensor "a" is being stored in the device...'
            device.store( a, filename )

        then : 'The expected file is being created at the given path.'
            new File( path + '/' + filename + '.idx' ).exists()

        and : 'Tensor "a" does no longer have a value (stored in RAM).'
            a.data == null

        when : 'Freeing the tensor...'
            device.free( a )

        then : 'The file will have been deleted!'
            !new File( path + '/' + filename + '.idx' ).exists()


        where : 'The following parameters have been used:'
            path             | filename
            "build/test-can" | "tensor_2x4_"
    }

    def 'A file device stores tensors in various file formats.'(
            String path, String filename, int[] shape, Class<FileHead<?,Number>> fileHeadClass, Class<?> dataTypeClass
    ) {
        given : 'A new tensor is being created for testing.'
            Tsr a = Tsr.of( shape, -8..8 )
        and : 'A String representation of the shape.'
            def shapeStr = String.join('x',(shape as List<Integer>).collect {String.valueOf(it)})
        and : 'A file device instance is being accessed for a given path.'
            def device = FileDevice.at( path )

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
            a.data == null

        and : 'The tensor is now of the expected data-type.'
            a.dataType == DataType.of( dataTypeClass )

        and : 'The device contains a "FileHead" instances of the expected type.'
            device.fileHeadOf( a ).class == fileHeadClass

        when : 'Freeing the tensor...'
            device.free( a )

        then : 'The file will be deleted!'
            !new File( path + '/' + filename ).exists()
            !new File( path ).listFiles().any {it.name.startsWith('tensor_' + shapeStr + '_f64_') }

        where : 'The following parameters are being used:'
            path             | filename            |  shape  || fileHeadClass  | dataTypeClass
            "build/test-can" | "tensor_2x4x3_.idx" | [2,4,3] || IDXHead.class | F64.class
            "build/test-can" | "tensor_2x4x3_.jpg" | [2,4,3] || JPEGHead.class | I16.class
            "build/test-can" | null                | [2,4,3] || IDXHead.class  | F64.class
            "build/test-can" | "tensor_4x3_.csv"   | [4,3]   || CSVHead.class  | String.class
    }


    def 'The file device can load known files in a directory.'()
    {
        given :
            def device = FileDevice.at( 'build/resources/test/csv' )
        expect :
            device.loadable.toSet() == ['biostats-without-head.csv', 'biostats.csv'].toSet() // If this fails: consider deleting the build folder!!
            device.loaded == []

        when :
            def t = device.load('biostats-without-head.csv')

        then :
            device.loadable == ['biostats.csv']
            device.loaded == ['biostats-without-head.csv']
            t.toString('fp') == '(18x5):[\n' +
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
            t = device.load('biostats.csv', [firstRowIsLabels:true])

        then :
            device.loadable == []
            device.loaded == ['biostats-without-head.csv', 'biostats.csv']
            t.toString('fp') == '(18x5):[\n' +
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


}
