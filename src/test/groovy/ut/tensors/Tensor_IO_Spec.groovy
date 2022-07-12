package ut.tensors


import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("The Tensor state Input and Output Specification")
@Narrative('''

    Tensors are complicated data structures with a wide range of different possible states.
    They can host elements of different types residing on many kinds of different devices.
    Here we want to define some basic stuff about how a tensor can be instantiated
    and how we can read from and write to the state of a tensor.
    Here we also specify how a tensor can be converted to another tensor of a different data type!
                    
''')
class Tensor_IO_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2>Tensor Instantiation and IO Tests</h2>
            <p>
                This specification covers some basic behaviour related to
                tensor instantiation and modification.
                This includes the instantiation of tensors with custom seeds, shapes and values...
                Included are also tests covering static factory methods.
            </p>
        """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
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

    def 'Tensors can be instantiated with String seed.'()
    {
        given : 'Three seeded 2D tensors are being instantiated.'
            Tsr t1 = Tsr.of([2, 3], "I am a seed! :)")
            Tsr t2 = Tsr.of([2, 3], "I am a seed! :)")
            Tsr t3 = Tsr.of([2, 3], "I am also a seed! But different. :)")

        expect : 'Equal seeds produce equal values.'
            t1.toString() == t2.toString()
            t1.toString() != t3.toString()
    }

    def 'Smart tensor constructors yield expected results.'()
    {
        given : 'Three scalar tensors.'
            Tsr a = Tsr.of(3)
            Tsr b = Tsr.of(2)
            Tsr c = Tsr.of(-1)

        when : Tsr t = Tsr.of("1+", a, "*", b)
        then : t.toString().contains("7.0")
        when : t = Tsr.of("1", "+", a, "*", b)
        then : t.toString().contains("7.0")
        when : t = Tsr.of("(","1+", a,")", "*", b)
        then : t.toString().contains("8.0")
        when : t = Tsr.of("(","1", "+", a,")", "*", b)
        then : t.toString().contains("8.0")
        when : t = Tsr.of("(", c, "*3)+", "(","1+", a,")", "*", b)
        then : t.toString().contains("5.0")
        when : t = Tsr.of("(", c, "*","3)+", "(","1+", a,")", "*", b)
        then : t.toString().contains("5.0")
        when : t = Tsr.of("(", c, "*","3", ")+", "(","1+", a,")", "*", b)
        then : t.toString().contains("5.0")

        when : t = Tsr.of([2, 2], [2, 4, 4])
        then : t.toString().contains("(2x2):[2.0, 4.0, 4.0, 2.0]")
        when : t = Tsr.of([2], [3, 5, 7])
        then :
            t.toString().contains("(2):[3.0, 5.0]")
            t.getItemsAs( double[].class ).length==2

        // Now the same with primitive array ! :
        when : t = Tsr.of(new int[]{2, 2}, new double[]{2, 4, 4})
        then : t.toString().contains("(2x2):[2.0, 4.0, 4.0, 2.0]")
        when : t = Tsr.of(new int[]{2}, new double[]{3, 5, 7})
        then :
            t.toString().contains("(2):[3.0, 5.0]")
            t.getItemsAs( double[].class ).length==2
    }

    def 'Indexing after reshaping works as expected.'()
    {
        given : 'We are using the legacy view for tensors where bracket types are swapped, just because...'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)

        and : 'A new tensor instance with the shape (4x3).'
            Tsr t1 = Tsr.of([4, 3], 1d..12d)

        when : 'Recording the index behavior before and after a reshape operation...'
            var t1_ioi_1 = t1.indexOfIndices(new int[]{2, 1})
            var t1_ioi_2 = t1.indexOfIndices(new int[]{1, 2})
            var t1_indices = t1.indicesOfIndex(5)

            Tsr t2 = Function.of(" [ 1, 0 ]:( I[0] ) ")(t1)
            var t2_ioi_1 = t2.indexOfIndices(new int[]{1, 2})
            var t2_idx = t2.indicesOfIndex(7)

            var t1_ioi_3 = t1.indexOfIndices(t1.indicesOfIndex(7)) // Element 7 '8.0' is at index 7!
            var t2_ioi_2 =  t2.indexOfIndices(t2.indicesOfIndex(7)) // Element 7 '11.0' is at index 10!

        then : 'These recorded values are as one would expect.'
            t1_ioi_1 == 7
            t1_ioi_2 == 5
            t1_indices[0] == 1
            t1_indices[1] == 2

            t2_ioi_1 == 7
            t2_idx[0] == 1
            t2_idx[1] == 3

            t1_ioi_3 == 7 // Element 7 '8.0' is at index 7!
            t2_ioi_2 == 10 // Element 7 '11.0' is at index 10!

            t1.toString().contains("[4x3]:(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)")
            t2.toString().contains("[3x4]:(1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0)")
    }


    def 'Passing String seed to tensor produces expected values.'()
    {
        when : Tsr r = Tsr.of([2, 2], "jnrejn")
        then : r.toString().contains("0.02600, -2.06129, -0.48373, 0.94884")
        when : r = Tsr.of([2, 2], "jnrejn2")
        then : !r.toString().contains("0.02600, -2.06129, -0.48373, 0.94884")
    }

    def 'Tensor value type can not be changed by passing float or double arrays to it.'()
    {
        given : 'We are using the legacy view for tensors where bracket types are swapped, just because...'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and : 'A new tensor instance.'
            Tsr x = Tsr.of(3)

        when : 'Setting the value of the tensor...'
            float[] value32 = new float[1]
            value32[0] = 5
            x.setItems(value32)

        then : '...the tensor will change as expected.'
            !(x.getItems() instanceof float[])
            !(x.unsafe.data instanceof float[])
            !(x.data instanceof float[])
            x.getItemsAs( float[].class )[ 0 ]==5.0f
            x.getItemsAs( double[].class )[0]==5.0d

        when : 'Doing the same with double array...'
            double[] value64 = new double[1]
            value64[0] = 4.0
            x.setItems(value64)

        then : '...once again the tensor changes as expected.'
            x.getItems() instanceof double[]
            x.unsafe.data instanceof double[]
            x.data instanceof double[]
            x.getItemsAs( float[].class )[ 0 ]==4.0f
            x.getItemsAs( double[].class )[0]==4.0d

            x.isLeave()
            !x.isBranch()
            !x.isOutsourced()
            !x.isVirtual()
            !x.isSlice()
            !x.isSliceParent()
            !x.belongsToGraph()
            x.getDevice() !=null
            x.getDevice() instanceof CPU
            x.rank()==1
            !x.rqsGradient()
            x.size()==1

        when : x.unsafe.toType( Float.class )
        then : x.items instanceof float[]

        when :
            value64 = new double[1]
            value64[0] = 7.0
            x.setItems(value64)

        then :
            !(x.getItems() instanceof double[])
            !(x.unsafe.data instanceof double[])
            !(x.data instanceof double[])
            x.getItemsAs( float[].class )[ 0 ]==7.0f
            x.getItemsAs( double[].class )[0]==7.0d
    }

    def 'We turn a tensor into a scalar value or string through the "as" operator!'()
    {
        given : 'A tensor of 3 floats:'
            var t = Tsr.ofFloats().vector(42, 42, 42)

        expect : 'We can now turn the tensor int other data types!'
            (t as Integer) == 42
            (t as Double) == 42
            (t as Short) == 42
            (t as Byte) == 42
            (t as Long) == 42
        and : 'Also use it instead of the "toString" method.'
            (t as String) == t.toString()
    }

    @IgnoreIf({ data.device == 'GPU' && !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'We can re-populate a tensor of shorts from a single scalar value!'(
         String device, Class<?> type
    ) {
        given : 'A tensor of 3 numbers:'
            var t = Tsr.of(type).vector(42, 666, 73)
        and : 'We store the tensor on the given device, to ensure that it work there as well.'
            t.to(device)

        when : 'We call the "setValue" method with a scalar value passed to it...'
            t.setItems(5)
        then : 'The value of the tensor will be an array of 3.'
            t.items == [5, 5, 5]
        and : 'We now expect the tensor to be virtual, because it stores only a single type of value.'
            t.isVirtual()

        where :
            device | type
            'CPU'  | Byte
            'CPU'  | Integer
            'CPU'  | Long
            'CPU'  | Double
            'CPU'  | Short
            'CPU'  | Float
            'GPU'  | Float
    }


    @IgnoreIf({ data.device == 'GPU' && !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'We can manipulate the underlying data array of a tensor through the unsafe API.'(
            String device, Class<?> type
    ) {
        given : 'A tensor of 3 numbers:'
            var t = Tsr.of(type).vector(42, 66, 73)
        and : 'We store the tensor on the given device, to ensure that it work there as well.'
            t.to(device)

        when : 'We create a slice which should internally reference the same data as the slice parent.'
            var s = t[1]
        then : 'The slice has the expected state!'
            s.isSlice()
            s.items == [66]
            s.data  == [42, 66, 73]

        when : 'We call the "setData" method with a scalar value passed to it...'
            s.unsafe.setDataAt(1, -9)

        then : 'The change will be reflected in the slice...'
            s.items == [-9]
        and : 'Also in the slice parent!'
            t.items == [42, -9, 73]
        and : 'Both tensors should have the same data array!'
            s.data  == [42, -9, 73]
            t.data  == [42, -9, 73]

        where :
            device | type
            'CPU'  | Double
            'CPU'  | Float
            'CPU'  | Byte
            'CPU'  | Short
            //'CPU'  | Integer
            //'CPU'  | Long
            //'GPU'  | Float
    }

    @IgnoreIf({ data.device == 'GPU' && !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'When we try to manipulate the underlying data array of a virtual tensor then it will become actual.'(
            String device, Class<?> type
    ) {
        given : 'A tensor of 3 numbers:'
            var t = Tsr.of(type).vector(1, 1, 1)
        and : 'We store the tensor on the given device, to ensure that it work there as well.'
            t.to(device)

        expect :
            t.isVirtual()
            t.items == [1, 1, 1]
            t.data == [1]

        when :
            t.at(2).set(42)

        then :
            !t.isVirtual()
            t.items == [1, 1, 42]
            t.data == [1, 1, 42]

        where :
            device | type
            'CPU'  | Double
            'CPU'  | Float
            'CPU'  | Byte
            'CPU'  | Short
            //'CPU'  | Integer
            //'CPU'  | Long
            //'GPU'  | Float
    }


    def 'Tensors value type can be changed by calling "toType(...)".'()
    {
        given : 'We are using the legacy view for tensors where bracket types are swapped, just because...'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Tsr x = Tsr.of(3)

        when : x.unsafe.toType( Float.class )
        then :
            x.getItems() instanceof float[]
            x.unsafe.data instanceof float[]
            x.data instanceof float[]
            x.getItemsAs( float[].class )[ 0 ]==3.0f

        when : x.unsafe.toType( Double.class )
        then :
            x.getItems() instanceof double[]
            x.unsafe.data instanceof double[]
            x.data instanceof double[]
            x.getItemsAs( float[].class )[ 0 ]==3.0f
    }

    def 'Vector tensors can be instantiated via factory methods.'(
        def data, Class<?> type, List<Integer> shape
    ) {
        given :
            Tsr<?> t = Tsr.of(data)

        expect :
            t.itemClass == type
        and :
            t.shape() == shape
        and :
            t.unsafe.data == data
            t.data == data
        and :
            t.items == data

        where :
            data                        ||  type  | shape
            new double[]{1.1, 2.2, 3.3} || Double | [ 3 ]
            new float[]{-0.21, 543.3}   || Float  | [ 2 ]
            new boolean[]{true, false}  || Boolean| [ 2 ]
            new short[]{1, 2, 99, -123} || Short  | [ 4 ]
            new long[]{3, 8, 4, 2, 3, 0}|| Long   | [ 6 ]
            new int[]{66, 1, 4, 42, -40}|| Integer| [ 5 ]
    }

    def 'A tensor produced by a function has expected properties.'()
    {
        given : 'We are using the legacy view for tensors where bracket types are swapped, just because...'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and : 'A simple scalar tensor containing the number "4".'
            Tsr x = Tsr.of(4)

        when: Tsr y = Tsr.of("th(I[0])", x)

        then:
            y.isBranch()
            !y.isLeave()
            y.belongsToGraph()
            x.belongsToGraph()
            y.toString().contains("[1]:(0.99932)")
    }

    def 'A tensor produced by the static "Tsr.Create.newRandom(shape)" has expected "random" value.'()
    {
        given : 'We are using the legacy view for tensors where bracket types are swapped, just because...'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)

        when : 'Creating a simple shape array...'
            int[] shape = new int[1]
            shape[0] = 4
        and : '...and passing it to the "newRandom" factory method to produce tensor x...'
            Tsr x = Tsr.ofRandom(Double, shape)

        then : '...the newly created variable x is as expected!'
            x.toString().contains("[4]:(-1.04829, -0.40245, -0.04347, -1.4921)")
        when : 'Again using the "andSeed" method with a long seed...'
            x = Tsr.ofDoubles().withShape(shape).andSeed(106605040595L)
        then : '...the newly created variable x is as expected!'
            x.toString().contains("[4]:(0.22266, 0.65678, -0.83154, 0.68019)")

        when : 'Again using the "andSeed" method with a long seed and with float as data type...'
            x = Tsr.ofFloats().withShape(shape).andSeed(106605040595L)
        then : '...the newly created variable x is as expected!'
            x.toString().contains("[4]:(0.22266, 0.65678, -0.83154, 0.68019)")
    }


    void 'Tensor values can be manipulated via static method calls within the "Tsr.IO" class.'()
    {
        given : 'We are using the legacy view for tensors where bracket types are swapped, just because...'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and : 'Two tensors which will be used for testing IO.'
            var t = Tsr.of([2, 2], [
                    1.0d, 4.0d,
                    2.0d, 7.0d,
            ])
            var v = Tsr.of([2, 2], [
                    1.0d, -1.0d,
                    1.0d, -1.0d
            ])

        when : t += v
        then : t.toString().contains("[2x2]:(2.0, 3.0, 3.0, 6.0)")

        when : t.setItemAt( 2, 6.0 as double )
        then : t.toString().contains("[2x2]:(2.0, 3.0, 6.0, 6.0)")

        when :
            int[] indices = new int[2]
            indices[1] = 1
            t.setItemAt(t.indexOfIndices(indices), -6.0 as double)
        then :
            t.toString().contains("[2x2]:(2.0, -6.0, 6.0, 6.0)")
            t.getItemAt(indices) ==-6.0d

        when :
            indices[0] = 1
            t[indices].timesAssign(-1d)

        then : t.toString().contains("[2x2]:(2.0, -6.0, 6.0, -6.0)")

        when : t[3].timesAssign(-2d)
        then : t.toString().contains("[2x2]:(2.0, -6.0, 6.0, 12.0)")

        when : t[indices] = 0d
        then : t.toString().contains("[2x2]:(2.0, -6.0, 6.0, 0.0)")

        when : t[2] = 99d
        then : t.toString().contains("[2x2]:(2.0, -6.0, 99.0, 0.0)")

        when : t[2].minusAssign(99d)
        then : t.toString().contains("[2x2]:(2.0, -6.0, 0.0, 0.0)")

        when : 'Modifying the first index of the indices array...'
            indices[0] = 0
        and : 'Using this new indices array for IO...'
            t[indices].minusAssign(-9d)
        then : 'The underlying data will have changed.'
            t.toString().contains("[2x2]:(2.0, 3.0, 0.0, 0.0)")

        when : t -= Tsr.of([2, 2], [1d, 2d, 3d, 4d])
        then : t.toString().contains("[2x2]:(1.0, 1.0, -3.0, -4.0)")
    }

    def 'The tensor data array can be modified by targeting them with an index.'(
            Class<Object> type, int[] shape, Object data, Object element, Object expected
    ) {
        given :
            var t = Tsr.of(type).withShape(shape).andFill(data)
        when :
            t.unsafe.setDataAt( 1, element )
        then :
            t.getDataAt( 1 ) == element
        and :
            t.unsafe.data == expected
            t.data == expected

        when :
            t = Tsr.of(type).withShape(shape).andFill(data)
        and :
            t.setItemAt( 1, element )
        then :
            t.getItemAt( 1 ) == element
        and :
            t.unsafe.data == expected
            t.data == expected

        where :
            type     | shape | data                             | element        || expected
            Float    | [2,2] | [-42, 24, 9, 3, -34] as float[]  | 0.032 as float || [-42.0, 0.032, 9.0, 3.0] as float[]
            Double   | [2,2] | [-42, 24, 9, 3, -34] as double[] | 0.032 as double|| [-42.0, 0.032, 9.0, 3.0] as double[]
            Byte     | [2,2] | [-42, 24, 9, 3, -34] as byte[]   | 1 as byte      || [-42, 1, 9, 3] as byte[]
            Short    | [2,2] | [-42, 24, 9, 3, -34] as short[]  | 1 as short     || [-42, 1, 9, 3] as short[]
            Long     | [2,2] | [-42, 24, 9, 3, -34] as long[]   | 1 as long      || [-42, 1, 9, 3] as long[]
            Integer  | [2,2] | [-42, 24, 9, 3, -34] as int[]    | 1 as int       || [-42, 1, 9, 3] as int[]
            Boolean  | [2,1] | [false, true, false] as boolean[]| false          || [false, false] as boolean[]
            Character| [2,1] | ['a', 'b', 'c'] as char[]        | 'x' as char    || ['a', 'x'] as char[]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Adding OpenCL device to tensor makes tensor be "outsourced" and contain the Device instance as component.'()
    {
        given : 'Neureka can access OpenCL (JOCL).'
            Device gpu = Device.get("gpu")
            Tsr t = Tsr.of([3, 4, 1], 3)

        expect : 'The following is to be expected with respect to the given :'
            !t.has(Device.class)
            !t.isOutsourced()
            !gpu.has(t)

        when : 'The tensor is being added to the OpenCL device...'
            t.to(gpu)

        then : 'The now "outsourced" tensor has a reference to the device and vice versa!'
            t.has(Device.class)
            t.isOutsourced()
            gpu.has(t)
    }

}
