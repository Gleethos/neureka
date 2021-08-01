package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.backend.api.Algorithm
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import neureka.devices.opencl.KernelCaller
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.utility.CLFunctionCompiler
import spock.lang.Specification

class CLFunctionCompiler_Unit_Tests extends Specification {

    def 'The CLFunctionCompiler produces an operation which properly integrates to the backend.'() {

        given : 'A mocked OpenCLDevice which allows us to test the compiler without OpenCL dependency.'
            def mockDevice = Mock(OpenCLDevice)
        and : 'Three simple scalar tensors (of doubles) which we will keep in RAM (not outsource to a device).'
            Tsr<Number> t1 = Tsr.of(1)
            Tsr<Number> t2 = Tsr.of(-2)
            Tsr<Number> t3 = Tsr.of(5)
        and : 'A simple test function which will serve as the basis for the optimization.'
            def funToBeOptimized = Function.of("i2 - (i0 / i1)")
        and : 'Finally we instantiate the compiler which uses the mocked device and the test function for optimization.'
            def compiler = new CLFunctionCompiler(
                                            mockDevice,
                                            funToBeOptimized,
                                            "test_fun"
                                        )

        when : 'We instruct the compiler to compile an optimized kernel in the form of an Operation instance'
            Operation resultOperation = compiler.optimize()

        then : 'This resulting operation will not be null!'
            resultOperation != null

        when : 'We create a new cloned context from the current one with the added test operation...'
            def context = Neureka.get().context().clone().addOperation(resultOperation)
        and : '... a context runner ...'
            def run = context.runner()
        and : 'We create a function based on our optimized operation...'
            Function fun = run {
                            new FunctionBuilder(Neureka.get().context())
                                        .build(resultOperation, 3, true)
                        }

        then : 'This function should of course not be null!'
            fun != null
        and : 'The function should look as follows when represented as a String:'
            fun.toString() == "test_fun(I[0], I[1], I[2])"

        when : 'Calling this function using the previously created scalars...'
            fun( t1, t2, t3 )

        then : """
                This will lead to an exception because these tensors are not members of the mocked OpenCLDevice instance.
                All of them are still residing in RAM without being member of any device...
                Therefore in order for this to work we need to fake the membership of these tensors!
        """
            def exception = thrown(IllegalStateException)
            exception.message == "No suitable implementation found for algorithm 'generic_algorithm_for_test_fun' and device type 'HostCPU'."

        when : 'We set the mocked device as components of our three scalar tensors...'
            t1.set(mockDevice)
            t2.set(mockDevice)
            t3.set(mockDevice)
            t1.setIsOutsourced(true)
            t2.setIsOutsourced(true)
            t3.setIsOutsourced(true)

        then : """
                This will require the mocked OpenCLDevice to notify the tensors that they are not already members.
                This will cause the tensor to add themselves to the device after which
                the tensors will ask the device again if they are now their members!
        """
            (0.._) * mockDevice.has(t1) >>> [false, true] // doesn't have it, then storing it, then has it!
            (0.._) * mockDevice.has(t2) >>> [false, true]
            (0.._) * mockDevice.has(t3) >>> [false, true]

        and : 'Finally the tensors are outsourced members of our mocked OpenCLDevice (Even though they are technically still in RAM).'
            t1.isOutsourced()
            t2.isOutsourced()
            t3.isOutsourced()

        when : 'We call the function again...'
            fun( t1, t2, t3 )


        then : """
                We will register that the Operation created by the CLFunctionCompiler managed to 
                integrate well with the Function backend (calculus package) and eventually
                dispatch an execution call to our mocked OpenCLDevice.
        """
            1 * mockDevice.execute({ ExecutionCall<OpenCLDevice> call ->
                call.device == mockDevice &&
                call.operation == resultOperation
            })

    }


    def 'The CLFunctionCompiler produces the expected "ad hoc" kernel.'() {

        given : 'A mocked OpenCLDevice which allows us to test the compiler without OpenCL dependency.'
            def mockDevice = Mock(OpenCLDevice)
        and : 'A mocked KernelCaller which allows us to check if the OpenCL backend is being called properly.'
            def mockCaller = Mock(KernelCaller)
        and : 'A test function which will be the optimization target for this test.'
            def funToBeOptimized = Function.of("i2 + (i0 / i1)")
        and : 'Finally we create the function compiler!'
            def compiler = new CLFunctionCompiler(
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
                        ExecutionCall.builder()
                                        .tensors(Tsr.of(0), Tsr.of(1), Tsr.of(2), Tsr.of(3))
                                        .operation(resultOperation)
                                        .algorithm(algorithm)
                                        .device(mockDevice)
                                        .build() as ExecutionCall<OpenCLDevice>
                )

        then : 'We expect that the implementation first checks with an optimized kernel already exists...'
            1 * mockDevice.hasAdHocKernel("test_fun_F64\$1_F64\$1_F64\$1_F64\$1") >> false
        and : 'The implementation will then also build and pass an "adHoc" kernel to the mocked device.'
            1 * mockDevice.compileAdHocKernel(
                    "test_fun_F64\$1_F64\$1_F64\$1_F64\$1",
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

""")
        and : 'After the kernel has been compiled we expect the implementation to '
            1 * mockDevice.getAdHocKernel("test_fun_F64\$1_F64\$1_F64\$1_F64\$1") >> mockCaller
        and : 'We expect that the caller receives 4 inputs, 1 output tensor and the 3 function arguments.'
            4 * mockCaller.passRaw(_)
        and : 'Finally the caller will receive a dispatch call with a work size of 1 (because the tensors are scalars). '
            1 * mockCaller.call(1)

    }


}
