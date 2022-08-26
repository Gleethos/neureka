package neureka.backend.main.operations.linear;

import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.operations.ConvUtil;
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
                        .setIdentifier(         "mul_conv"    )
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
                            .kernelPostfix( this.getIdentifier() )
                            .execution(
                                call -> {
                                    int offset = ( call.input( Number.class, 0 ) != null ) ? 0 : 1;
                                    int gwz = ( call.input( Number.class, 0 ) != null ) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                                    call.getDevice()
                                        .getKernel(call)
                                        .passAllOf( call.input( Number.class, offset ) )
                                        .passAllOf( call.input( Number.class, offset + 1 ) )
                                        .passAllOf( call.input( Number.class, offset + 2 ) )
                                        .pass( call.input( Number.class, 0 ).rank() )
                                        .pass( call.getValOf( Arg.DerivIdx.class ) )
                                        .call( gwz );

                                    return call.input( 0 );
                                }
                            )
                            .build()
                )
        );

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
