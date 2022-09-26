package ut.device.internal

import neureka.Neureka
import neureka.Tsr
import neureka.backend.api.Algorithm
import neureka.backend.api.DeviceAlgorithm
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionParser
import neureka.devices.opencl.CLContext
import neureka.devices.opencl.KernelCaller
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.utility.CLFunctionCompiler
import neureka.view.NDPrintSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Turning functions into kernels.")
@Narrative('''

    Neureka parses mathematical expressions into an AST representation
    hidden behind the Function interface...
    This feature does not exist without reason, we can use
    this abstract syntax tree to compile to OpenCL kernels
    for optimal execution speed!

''')
class CLFunctionCompiler_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            Specified below are strict tests for covering the ability of 
            OpenCL devices to be able produce optimized functions given
            a normal function instance created from a String...
        """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
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

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'The OpenCLDevice produces a working optimized Function (internally using the CLFunctionCompiler).'()
    {
        given : 'We get the first available OpenCLDevice we can find in the CLContext!'
            def device = Neureka.get().backend().get(CLContext.class).platforms[0].devices[0]
        and : 'Three scalar test tensors which will be used as inputs to the optimized function.'
            Tsr<Double> t1 = Tsr.of(-2d).to(device)
            Tsr<Double> t2 = Tsr.of(5d).to(device)
            Tsr<Double> t3 = Tsr.of(2d).to(device)

        and : 'A test function which will be the optimization target for this test.'
            def funToBeOptimized = Function.of("i2 + (i0 / i1)") // 2 + (-2 / 5)

        when : 'We instruct the device to produce an optimized Function based on the provided test function...'
            Function optimized = device.optimizedFunctionOf(funToBeOptimized, "my_test_fun")

        then : 'Initially we expect that the device does not contain the "ad hoc" kernel with the following signature...'
            !device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

        when : 'We test the optimized function by calling it with three arguments...'
            Tsr result = optimized( t1, t2, t3 )

        then : '...the result should look as follows:'
            result.toString() == "(1):[1.6]"

        and : 'We expect that the device has an underlying kernel with the following name:'
            device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")
    }

    /* // WIP
    def 'The OpenCLDevice produces an optimized Function for slices.'() {

        given : 'This system supports OpenCL'
        if ( !Neureka.get().canAccessOpenCL() ) return
        and : 'We get the first available OpenCLDevice we can find in the CLContext!'
        def device = Neureka.get().context().get(CLContext.class).platforms[0].devices[0]
        and : 'Three scalar test tensors which will be used as inputs to the optimized function.'
        Tsr<Double> t1 = Tsr.of([[1, 3, 2],[4, -2, 5]])[0..1, 1..2]
        t1.set(device)
        Tsr<Double> t2 = Tsr.of(5).set(device)
        Tsr<Double> t3 = Tsr.of(2).set(device)

        and : 'A test function which will be the optimization target for this test.'
        def funToBeOptimized = Function.of("i2 + (i0 / i1)") // 2 + (-2 / 5)

        when : 'We instruct the device to produce an optimized Function based on the provided test function...'
        Function optimized = device.optimizedFunctionOf(funToBeOptimized, "my_test_fun")

        then : 'Initially we expect that the device does not contain the "ad hoc" kernel with the following signature...'
        !device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

        when :
        Tsr result = optimized( t1, t2, t3 )

        then :
        result.toString() == "(1):[1.6E0]"

        and :
        device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

    }
     */

    def 'The CLFunctionCompiler produces an operation which properly integrates to the backend.'() {

        given : 'A mocked OpenCLDevice which allows us to test the compiler without OpenCL dependency.'
            var mockDevice = Mock(OpenCLDevice)
        and : 'Three simple scalar tensors (of doubles) which we will keep in RAM (not outsource to a device).'
            Tsr<Number> t1 = Tsr.of( 1d )
            Tsr<Number> t2 = Tsr.of(-2d )
            Tsr<Number> t3 = Tsr.of( 5d )
        and : 'A simple test function which will serve as the basis for the optimization.'
            var funToBeOptimized = Function.of("i2 - (i0 / i1)")
        and : 'Finally we instantiate the compiler which uses the mocked device and the test function for optimization.'
            var compiler = new CLFunctionCompiler(
                                            mockDevice,
                                            funToBeOptimized,
                                            "test_fun"
                                        )

        when : 'We instruct the compiler to compile an optimized kernel in the form of an Operation instance'
            Operation resultOperation = compiler.optimize()

        then : 'This resulting operation will not be null!'
            resultOperation != null

        when : 'We create a new cloned context from the current one with the added test operation...'
            var context = Neureka.get().backend().clone().addOperation(resultOperation)
        and : '... a context runner ...'
            var run = context.runner()
        and : 'We create a function based on our optimized operation...'
            Function fun = run {
                            new FunctionParser( Neureka.get().backend() )
                                        .parse(resultOperation, 3, true)
                        }

        then : 'This function should of course not be null!'
            fun != null
        and : 'The function should look as follows when represented as a String:'
            fun.toString() == "test_fun(I[0], I[1], I[2])"
        and : 'The context now stores the newly created operation.'
            context.getOperation("test_fun") == resultOperation
        when : 'Querying the new operation for an algorithm...'
            var foundAlgorithm = ( DeviceAlgorithm ) resultOperation
                                                        .getAlgorithmFor(
                                                            ExecutionCall.of(t1, t2, t3)
                                                                            .running(resultOperation)
                                                                            .on(mockDevice)
                                                        )
        then : 'We expect this algorithm to exist!'
            foundAlgorithm != null
        and : 'This algorithm is expected to host an implementation for our mocked device.'
            foundAlgorithm.getImplementationFor(mockDevice.getClass()) != null

        when: 'Calling this function using the previously created scalars...'
            fun( t1, t2, t3 )

        then : """
                This will lead to an exception because these tensors are not members of the mocked OpenCLDevice instance.
                All of them are still residing in RAM without being member of any device...
                Therefore in order for this to work we need to fake the membership of these tensors!
        """
            var exception = thrown(IllegalStateException)
            exception.message == "No suitable implementation found for operation 'test_fun', algorithm 'generic_algorithm_for_test_fun' and device type 'CPU'."

        when : 'We set the mocked device as components of our three scalar tensors...'
            t1.to(mockDevice)
            t2.to(mockDevice)
            t3.to(mockDevice)

        then : """
                This will require the mocked OpenCLDevice to notify the tensors that they are not already members.
                This will cause the tensor to add themselves to the device after which
                the tensors will ask the device again if they are now their members!
        """
            (0.._) * mockDevice.has(t1) >>> [false, true] // doesn't have it, then storing it, then has it!
            (0.._) * mockDevice.has(t2) >>> [false, true]
            (0.._) * mockDevice.has(t3) >>> [false, true]
        and : 'The update method is being called on the device because it becomes the component of 3 tensors!'
            (1.._) * mockDevice.update(_) >> true

        and : 'Finally the tensors are outsourced members of our mocked OpenCLDevice (Even though they are technically still in RAM).'
            t1.isOutsourced()
            t2.isOutsourced()
            t3.isOutsourced()

        when : 'We set a dummy implementation so that the real implementation does not get called'
            foundAlgorithm.setImplementationFor(mockDevice.getClass(), (call)->call.input(0))
        and : 'We call the function again...'
            fun( t1, t2, t3 )

        then : """
                We will register that the Operation created by the CLFunctionCompiler managed to 
                integrate well with the Function backend (calculus package) and eventually
                dispatch an execution call to our mocked OpenCLDevice.
        """
            1 * mockDevice.approve({ ExecutionCall<OpenCLDevice> call ->
                call.device == mockDevice &&
                call.operation == resultOperation
            })

    }


    def 'The CLFunctionCompiler produces the expected "ad hoc" kernel.'()
    {
        given : 'A mocked OpenCLDevice which allows us to test the compiler without OpenCL dependency.'
            var mockDevice = Mock(OpenCLDevice)
        and : 'A mocked KernelCaller which allows us to check if the OpenCL backend is being called properly.'
            var mockCaller = Mock(KernelCaller)
        and : 'A test function which will be the optimization target for this test.'
            var funToBeOptimized = Function.of("i2 + (i0 / i1)")
        and : 'Finally we create the function compiler!'
            var compiler = new CLFunctionCompiler(
                                    mockDevice,
                                    funToBeOptimized,
                                    "test_fun"
                                )

        when : 'We instruct the compiler to produce an optimized operation based on the provided test function...'
            Operation resultOperation = compiler.optimize()
        and : 'We query this new operation for a any algorithm... (There should really only be a single one)'
            Algorithm algorithm = resultOperation.getAlgorithm(Algorithm.class)

        then : 'The returned algorithm should of course not be null...'
            algorithm != null
        and : '...as well as an OpenCLDevice specific implementation within the algorithm.'
            algorithm.getImplementationFor(OpenCLDevice.class) != null

        when : 'We call the OpenCLDevice specific implementation by passing a well populated ExecutionCall.'
            algorithm
                .getImplementationFor(OpenCLDevice.class)
                .run(
                        ExecutionCall.of(Tsr.of(0d), Tsr.of(1d), Tsr.of(2d), Tsr.of(3d))
                                        .running(resultOperation)
                                        .on(mockDevice) as ExecutionCall<OpenCLDevice>
                )

        then : 'We expect that the implementation first checks with an optimized kernel already exists...'
            1 * mockDevice.hasAdHocKernel("test_fun_F64\$1_F64\$1_F64\$1_F64\$1") >> false
        and : 'The implementation will then also build and pass an "adHoc" kernel to the mocked device.'
            1 * mockDevice.compileAndGetAdHocKernel("test_fun_F64\$1_F64\$1_F64\$1_F64\$1",
                    """
    int _i_of_idx_on_tln( int* cfg, int rank ) // cfg: [ 0:shape | 1:translation | 2:mapping | 3:indices | 4:strides | 5:offset ]
    {
        int* offset      = ( cfg + rank * 5 );
        int* strides     = ( cfg + rank * 4 );
        int* indices     = ( cfg + rank * 3 );
        int* translation = ( cfg + rank     );
        int i = 0;
        for ( int ii = 0; ii < rank; ii++ ) {
            i += ( indices[ ii ] * strides[ ii ] + offset[ ii ] ) * translation[ ii ];
        }
        return i;
    }

    int _i_of_i( int i, int* cfg, int rank ) // cfg: [ 0:shape | 1:translation | 2:mapping | 3:indices | 4:strides | 5:offset ]
    {
        int* indices    = ( cfg + rank * 3 );
        int* indicesMap = ( cfg + rank * 2 );
        for( int ii = 0; ii < rank; ii++ ) {
            indices[ ii ] = ( i / indicesMap[ ii ] ); // is derived from the shape of a tensor. Translates scalar index to dim-Index
            i %= indicesMap[ ii ];
        }
        return _i_of_idx_on_tln( cfg, rank );
    }

    __kernel void test_fun_F64\$1_F64\$1_F64\$1_F64\$1(
        __global double* arg0, __global double* arg1, __global double* arg2, __global double* arg3
    ) {                                                                                     
        int cfg0[] = {1,1,1,0,1};
        int cfg1[] = {1,1,1,0,1};
        int cfg2[] = {1,1,1,0,1};
        int cfg3[] = {1,1,1,0,1};                                                                                          
        unsigned int i = get_global_id( 0 );                                              
        double v1 = arg1[_i_of_i(i, cfg1, 1)];
        double v2 = arg2[_i_of_i(i, cfg2, 1)];
        double v3 = arg3[_i_of_i(i, cfg3, 1)];                                                                                          
        arg0[_i_of_i(i, cfg0, 1)] = (v1 + (v2 / v3));                         
    }                                                                                     

""") >> mockCaller
        and : 'We expect that the caller receives 4 inputs, 1 output tensor and the 3 function arguments.'
            4 * mockCaller.pass(_)
        and : 'Finally the caller will receive a dispatch call with a work size of 1 (because the tensors are scalars). '
            1 * mockCaller.call(1)

    }

}
