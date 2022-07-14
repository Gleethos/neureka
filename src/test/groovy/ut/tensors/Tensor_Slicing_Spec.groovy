package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.args.Arg
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Tensors within Tensors")
@Narrative('''

    ND-Array data structures can be "sliced" in the sense
    that one can create a subset view of the underlying data inside a tensor
    through a new tensor instance...
    This can be a tedious and complicated procedure.
    Therefore a tensor should expose a various user friendly API for slicing which
    are also fit for various languages.
    This specification covers these APIs for tensor slicing.
                    
''')
class Tensor_Slicing_Spec extends Specification
{
    def setupSpec() {
        reportHeader """
                <h2> Tensor Slicing </h2>
                <br> 
                <p>
                    This specification covers the behavior of tensors when being sliced 
                    on multiple different device types using the SliceBuilder API.           
                </p>
            """
    }

    def setup() {
        Neureka.get().reset()
        Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
    }

    def 'When Slicing only one axis using the SliceBuilder API, the other axes will be sliced implicitly.' (
            Device device
    ) {
        given : 'A device could be found.'
            if ( device == null ) return

        and: 'The found device is also supported (Which might not always be the case for the OpenCLDevice).'
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        and : 'A 3 dimensional tensor which will be sliced.'
            Tsr<Double> t = Tsr.of([2, 4, 3], -3d..7d)

        and : 'Which will be placed on a given device:'
            t.to(device)

        when : 'Slicing axis 1 of the tensor using the "from" & "to" methods...'
            Tsr s = t.slice()
                        .axis(1).from(1).to(2)
                        .get() // Note: Axis 0 and 2 will be sliced implicitly if not specified!

        then : 'This will result in a slice which has 4 axis entries less than the original tensor.'
            s.shape().sum() == t.shape().sum() - 2

        and : 'This new slice will be displayed as follows when printed (with adjusted indent):'
            s.toString().replace('\n', '\n'+" "*20) ==
                """(2x2x3):[
                       [
                          [   0.0 ,   1.0 ,   2.0  ],
                          [   3.0 ,   4.0 ,   5.0  ]
                       ],
                       [
                          [   1.0 ,   2.0 ,   3.0  ],
                          [   4.0 ,   5.0 ,   6.0  ]
                       ]
                    ]"""

        and : 'As already shown by the printed view, the tensor as the expected shape:'
            s.shape() == [2, 2, 3]

        where: 'This works both on the GPU as well as CPU of course.'
            device << [Device.get('gpu'), CPU.get() ]

    }

    def 'The "at" method and the "from" / "to" methods can be mixed when slicing a tensor.' (
            Device device
    ) {
        given : 'A device could be found.'
            if ( device == null ) return

        and: 'The found device is also supported (Which might not always be the case for the OpenCLDevice).'
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        and : 'A 3 dimensional tensor which will be sliced.'
            Tsr<Double> t = Tsr.of([3, 3, 4], -11d..3d)

        and : 'Which will be placed on a given device:'
            t.to(device)

        when : 'Slicing the tensor using both "at", "from"/"to" and an implicit full ranged slice for axis 1...'
            Tsr s = t.slice()
                        .axis(0).at(1)
                        // Note: Axis 1 will be sliced implicitly if not specified!
                        .axis(2).from(1).to(2)
                        .get()

        then : 'This will result in a slice which has 4 axis entries less than the original tensor.'
            s.shape().sum() == t.shape().sum() - 4

        and : 'This new slice will be displayed as follows when printed (with adjusted indent):'
            s.toString().replace('\n', '\n'+" "*20) ==
                """(1x3x2):[
                       [
                          [   2.0 ,   3.0  ],
                          [  -9.0 ,  -8.0  ],
                          [  -5.0 ,  -4.0  ]
                       ]
                    ]"""

        and : 'The "at" method sliced a single axis point whereas the "from" & "to" sliced from 1 to 2.'
            s.shape() == [1, 3, 2]

        where: 'This works both on the GPU as well as CPU of course.'
            device << [Device.get('gpu'), CPU.get() ]

    }



    def 'A tensor can be sliced by passing ranges in the form of primitive arrays.' (
            Device device
    ) {
        given : 'A device could be found.'
            if ( device == null ) return

        and: 'The found device is also supported (Which might not always be the case for the OpenCLDevice).'
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        and : 'A 3 dimensional tensor which will be sliced.'
            Tsr t = Tsr.of([3, 3, 4], -11..3)

        and : 'Which will be placed on a given device:'
            t.to(device)

        when : 'Slicing the tensor using primitive int arrays...'
            Tsr s = t.getAt(
                        new int[]{1},    // Axis 0
                        new int[]{0, 2}, // Axis 1
                        new int[]{1, 2}  // Axis 2
                    )

        then : 'This will result in a slice which has 4 axis entries less than the original tensor.'
            s.shape().sum() == t.shape().sum() - 4

        and : 'This new slice will be displayed as follows when printed (with adjusted indent):'
            s.toString().replace('\n', '\n'+" "*20) ==
                """(1x3x2):[
                       [
                          [   2.0 ,   3.0  ],
                          [  -9.0 ,  -8.0  ],
                          [  -5.0 ,  -4.0  ]
                       ]
                    ]"""

        and : 'The the slice will have the following shape'
            s.shape() == [1, 3, 2]

        where: 'This works both on the GPU as well as CPU of course.'
            device << [Device.get('gpu'), CPU.get() ]

    }



    def 'A tensor can be sliced by passing ranges in the form of lists (Groovy ranges).' (
            Device device
    ) {
        given : 'A device could be found.'
            if ( device == null ) return

        and: 'The found device is also supported (Which might not always be the case for the OpenCLDevice).'
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        and : 'A 3 dimensional tensor which will be sliced.'
            Tsr t = Tsr.of([3, 3, 4], -11..3)

        and : 'Which will be placed on a given device:'
        t.to(device)

        when : 'Slicing the tensor using lists of integers generated by the Groovy range operator..'
            Tsr s = t[1, 0..2, 1..2]

        then : 'This will result in a slice which has 4 axis entries less than the original tensor.'
            s.shape().sum() == t.shape().sum() - 4

        and : 'This new slice will be displayed as follows when printed (with adjusted indent):'
            s.toString().replace('\n', '\n'+" "*20) ==
                """(1x3x2):[
                       [
                          [   2.0 ,   3.0  ],
                          [  -9.0 ,  -8.0  ],
                          [  -5.0 ,  -4.0  ]
                       ]
                    ]"""

        and : 'The the slice will have the following shape'
            s.shape() == [1, 3, 2]

        where: 'This works both on the GPU as well as CPU of course.'
            device << [Device.get('gpu'), CPU.get() ]

    }


    def 'The slice builder also supports slicing with custom step sizes.' (
            Device device
    ) {
        given : 'A device could be found.'
            if ( device == null ) return

        and: 'The found device is also supported (Which might not always be the case for the OpenCLDevice).'
            if ( device instanceof OpenCLDevice && !Neureka.get().canAccessOpenCLDevice() ) return

        and : 'A 3 dimensional tensor which will be sliced.'
            Tsr<Double> t = Tsr.of([3, 3, 4], -11d..3d)

        and : 'Which will be placed on a given device:'
            t.to(device)

        when : 'Slicing the tensor using lists of integers generated by the Groovy range operator..'
            Tsr s = t.slice()
                        .axis(0).at(0)
                        .axis(1).at(0)
                        .axis(2).from(0).to(3).step(2)
                        .get()

        then : 'This will result in a slice which has 4 axis entries less than the original tensor.'
            s.shape().sum() == t.shape().sum() - 6

        and : 'This new slice will be displayed as follows when printed (with adjusted indent):'
            s.toString() == """(1x1x2):[
   [
      [  -11.0,  -9.0  ]
   ]
]"""

        and : 'The the slice will have the following shape'
            s.shape() == [1, 1, 2]

        where: 'This works both on the GPU as well as CPU of course.'
            device << [Device.get('gpu'), CPU.get() ]

    }


    def 'Slicing is also a Function with autograd support!'(
    ) {
        given :
            var t = Tsr.ofBytes().withShape(4, 4).andFill(-1, 7, 3).setRqsGradient(true)
        and :
            int[] newShape  = [2, 2]
            int[] newOffset = [1, 1]
            int[] newSpread = [1, 1]

        when :
            Tsr<Object> slice = Function.of("slice(I[0])", true)
                                    .with(Arg.Shape.of(newShape),Arg.Offset.of(newOffset),Arg.Stride.of(newSpread))
                                    .call(t)
        then :
            slice.toString() == "(2x2):[\n" +
                                "   [   3.0 ,  -1.0  ],\n" +
                                "   [  -1.0 ,   7.0  ]\n" +
                                "]"

        when :
           slice.backward(-8)

        then :
           t.toString() == "(4x4):[\n" +
                           "   [  -1.0 ,   7.0 ,   3.0 ,  -1.0  ],\n" +
                           "   [   7.0 ,   3.0 ,  -1.0 ,   7.0  ],\n" +
                           "   [   3.0 ,  -1.0 ,   7.0 ,   3.0  ],\n" +
                           "   [  -1.0 ,   7.0 ,   3.0 ,  -1.0  ]\n" +
                           "]:g:[\n" +
                           "   [   0.0 ,   0.0 ,   0.0 ,   0.0  ],\n" +
                           "   [   0.0 ,  -8.0 ,  -8.0 ,   0.0  ],\n" +
                           "   [   0.0 ,  -8.0 ,  -8.0 ,   0.0  ],\n" +
                           "   [   0.0 ,   0.0 ,   0.0 ,   0.0  ]\n" +
                           "]"
    }


    def 'Normal slicing will try to do autograd.'()
    {
        given :
            var t = Tsr.ofBytes().withShape(5).andFill(-1, 7, 3).setRqsGradient(true)

        when :
            var s = t[2..3]
        and :
            s.backward(42)

        then :
            t.toString() == "(5):[  -1.0 ,   7.0 ,   3.0 ,  -1.0 ,   7.0  ]:g:[   0.0 ,   0.0 ,  42.0 ,  42.0 ,   0.0  ]"

    }

    def 'We can avoid autograd when slicing by using the "detached" instead of the "get" method.'()
    {
        given :
            var t = Tsr.ofBytes().withShape(4).andFill(-1, 7, 3).setRqsGradient(true)

        when :
            t.slice()
                .axis(0).from(0).to(1)
                .detached()
                .backward(73)

        then :
            t.toString() == "(4):[  -1.0 ,   7.0 ,   3.0 ,  -1.0  ]:g:[null]"

        when :
            t.slice()
                .axis(0).from(0).to(1)
                .get()
                .backward(73)

        then :
            t.toString() == "(4):[  -1.0 ,   7.0 ,   3.0 ,  -1.0  ]:g:[  73.0 ,  73.0 ,   0.0 ,   0.0  ]"
    }

}
