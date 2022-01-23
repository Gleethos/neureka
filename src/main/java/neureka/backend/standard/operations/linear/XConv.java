package neureka.backend.standard.operations.linear;

import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.Fun;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.operations.ConvUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class XConv extends AbstractOperation
{
    public XConv()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "mul_conv"    )
                        .setOperator(         "x"    )
                        .setArity(            2          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        Convolution convolution = ConvUtil.getConv();

        setAlgorithm(
            Convolution.class,
            convolution
                .setImplementationFor(
                    CPU.class,
                    Convolution.implementationForCPU()
                            .with(Fun.F64F64ToF64.triple(
                                    ( a, b ) -> a * b,
                                    ( a, b ) -> b, // Deriving at input 0
                                    ( a, b ) -> a  // deriving input 1
                            ))
                            .with(Fun.F32F32ToF32.triple(
                                    ( a, b ) -> a * b,
                                    ( a, b ) -> b, // Deriving at input 0
                                    ( a, b ) -> a  // deriving input 1
                            ))
                            .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    CLImplementation.compiler()
                            .arity( 3 )
                            .kernelSource( convolution.getKernelSource() )
                            .activationSource( "value = src1 * src2;\n" )
                            .differentiationSource( "value += handle * drain;\n" )
                            .kernelPostfix( this.getFunction() )
                            .execution(
                                call -> {
                                    int offset = ( call.getTsrOfType( Number.class, 0 ) != null ) ? 0 : 1;
                                    int gwz = ( call.getTsrOfType( Number.class, 0 ) != null ) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                    call.getDevice()
                                        .getKernel(call)
                                        .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                        .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                        .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                        .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                        .pass( call.getValOf( Arg.DerivIdx.class ) )
                                        .call( gwz );
                                }
                            )
                            .build()
                )
        );

    }


    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" x ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
